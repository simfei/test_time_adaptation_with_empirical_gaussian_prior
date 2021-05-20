import tensorflow as tf
import numpy as np
import model as model
import train_config as config
import utils
import data_hcp as data_hcp
from data_aug import augment_images
import h5py


def iterate_minibatches_images(images, batch_size, state='prior'):
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    for b_i in range(n_images // batch_size):
        if b_i + batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i * batch_size:(b_i + 1) * batch_size])
        x = images[batch_indices, ...]

        if config.da_ratio > 0:
            if state == 'prior':
                x = augment_images(images=x,
                                   data_aug_ratio=config.da_ratio,
                                   sigma=config.sigma,
                                   alpha=config.alpha,
                                   trans_min=config.trans_min,
                                   trans_max=config.trans_max,
                                   rot_min=config.rot_min,
                                   rot_max=config.rot_max,
                                   scale_min=config.scale_min,
                                   scale_max=config.scale_max,
                                   gamma_min=config.gamma_min,
                                   gamma_max=config.gamma_max,
                                   brightness_min=config.brightness_min,
                                   brightness_max=config.brightness_max,
                                   noise_min=config.noise_min,
                                   noise_max=config.noise_max)

        x = np.expand_dims(x, axis=-1)
        yield x


def pred_features(images, feature_type):
    with tf.Graph().as_default():
        # images = np.expand_dims(images, axis=-1)
        if images.ndim == 3:
            images = np.expand_dims(images, axis=0)
        images_pl = tf.placeholder(tf.float32,
                                   shape=[None] + list((256, 256)) + [1],
                                   name='images')
        images_normalized, added_residual = model.normalize(images_pl, config,
                                                            training_pl=tf.constant(False, dtype=tf.bool))
        logits, softmax, preds, features = model.predict_i2l(images_normalized, config,
                                                             training_pl=tf.constant(False, dtype=tf.bool),
                                                             returned_feature=feature_type)

        i2l_vars = []
        norm_vars = []
        for v in tf.global_variables():
            var_name = v.name
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                norm_vars.append(v)

        init_ops = tf.global_variables_initializer()
        sess = tf.Session()

        saver_i2l = tf.train.Saver(var_list=i2l_vars)

        tf.get_default_graph().finalize()
        sess.run(init_ops)

        path_to_model = base_logdir + model_dir
        ckp_path = utils.get_latest_model_checkpoint_path(path_to_model, 'model.ckpt')
        saver_i2l.restore(sess, ckp_path)

        features_ = sess.run(features, feed_dict={images_pl: images})
        features_ = np.array(features_).astype(float)
        sess.close()
        return features_


def prepare_prior(feature_type, prior_type='egp_sw'):
    if 'brain' in base_logdir:
        train_images = data_hcp.load_and_maybe_process_data(input_folder=config.orig_data_root_hcp,
                                                            preprocessing_folder=config.preproc_folder_hcp,
                                                            idx_start=0,
                                                            idx_end=20,
                                                            protocol='T1',
                                                            size=config.image_size,
                                                            depth=config.image_depth_hcp,
                                                            target_resolution=config.target_resolution_brain)
        images = train_images['images']
        nz = np.array(train_images['nz'])
    elif 'prostate' in base_logdir:
        filename = 'data/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5'
        with h5py.File(filename, 'r') as f:
            images = np.array(f['images_train'])
            nz = list(f['nz_train'])

    if prior_type == 'egp_sw':
        pat_end_slice = []
        pat_end_slice.append(0)
        for i in range(len(nz)):
            pat_end_slice.append(np.sum(nz[:(i + 1)]))

        means_all = []
        vars_all = []
        for i in range(len(nz)):
            features = []
            for img in iterate_minibatches_images(images[pat_end_slice[i]:pat_end_slice[i + 1]],
                                                  batch_size=nz[i],
                                                  state='prior'):
                features.append(pred_features(img, feature_type))
            features = np.concatenate(features, axis=0)
            print(features.shape)
            means = []
            vars = []
            for c in range(features.shape[-1]):
                means.append(np.mean(features[..., c]))
                vars.append(np.std(features[..., c]) ** 2)
            means = np.array(means).reshape(1, -1)
            vars = np.array(vars).reshape(1, -1)
            means_all.append(means)
            vars_all.append(vars)
            del features
        means_all = np.concatenate(means_all, axis=0)
        vars_all = np.concatenate(vars_all, axis=0)

        mean_mean = np.mean(means_all, axis=0)
        var_mean = np.std(means_all, axis=0) ** 2
        mean_var = np.mean(vars_all, axis=0)
        var_var = np.std(vars_all, axis=0) ** 2

        return mean_mean, var_mean, mean_var, var_var

    elif prior_type == 'egp_ndc':
        features = []
        i = 0
        # batch size is adjusted according to image depth
        for imgs in iterate_minibatches_images(images, batch_size=16, state='prior'):
            print(i)
            i += 1
            features.append(pred_features(imgs, feature_type))

        features = np.concatenate(features, axis=0)
        print(features.shape)
        means = []
        covar = np.zeros((features.shape[-1], features.shape[-1]))
        for c in range(features.shape[-1]):
            means.append(np.mean(features[..., c]))
        means = np.array(means).reshape((-1, 1))
        for n in range(features.shape[0]):
            mu_n = []
            for c in range(features.shape[-1]):
                mu_n.append(np.mean(features[n, ..., c]))
            mu_n = np.array(mu_n).reshape((-1, 1))
            covar += np.matmul(mu_n - means, (mu_n - means).T)
        covar = covar / features.shape[0]
        covar_inv = np.linalg.inv(covar)
        covar_inv = covar_inv.astype('float32')
        del features
        return means, covar_inv

    elif prior_type == 'egp_vp':
        i = 0
        features = []
        for img in iterate_minibatches_images(images, batch_size=16, state='prior'):
            print(i)
            features.append(pred_features(img, feature_type))
            i += 1
        features = np.concatenate(features, axis=0)
        print(features.shape)
        means = []
        vars = []
        for c in range(features.shape[-1]):
            means.append(np.mean(features[...,c]))
            vars.append(np.std(features[...,c])**2)
        del features
        return means, vars

    else:
        raise Exception('Prior type {} is not supported.'.format(prior_type))


if __name__ == '__main__':
    base_logdir = 'logdir_brain/'
    model_dir = 'models/'
    save_dir = 'priors_brain'
    prior_type = 'egp_sw'
    feature_types = ['NI', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2', 'conv6_2', 'conv7_2']
    for feature_type in feature_types:
        if prior_type == 'egp_sw':
            mean_mean, var_mean, mean_var, var_var = prepare_prior(feature_type, prior_type)
            np.save('{}/mean_mean_{}.npy'.format(save_dir, feature_type), mean_mean)
            np.save('{}/var_mean_{}.npy'.format(save_dir, feature_type), var_mean)
            np.save('{}/mean_var_{}.npy'.format(save_dir, feature_type), mean_var)
            np.save('{}/var_var_{}.npy'.format(save_dir, feature_type), var_var)
        elif prior_type == 'egp_ndc':
            means, covar_inv = prepare_prior(feature_type, prior_type)
            np.save('{}/means_{}.npy'.format(save_dir, feature_type), means)
            np.save('{}/covar_inv_{}.npy'.format(save_dir, feature_type), covar_inv)
        elif prior_type == 'egp_vp':
            means, vars = prepare_prior(feature_type, prior_type)
            np.save('{}/means_{}.npy'.format(save_dir, feature_type), means)
            np.save('{}/vars_{}.npy'.format(save_dir, feature_type), vars)


