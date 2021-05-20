import tensorflow as tf
import numpy as np
import utils
import model as model
import train_config as config
import sklearn.metrics as met


def predict_features(images, model_dir, returned_feature):
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[None] + list(config.image_size) + [1], name='images')

        images_normalized, added_residual = model.normalize(images_pl, config,
                                                            training_pl=tf.constant(False, dtype=tf.bool))
        _, _, _, feature = model.predict_i2l(images_normalized, config,
                                             training_pl=tf.constant(False, dtype=tf.bool),
                                             returned_feature=returned_feature)

        i2l_vars = []
        for v in tf.global_variables():
            i2l_vars.append(v)
        init_ops = tf.global_variables_initializer()

        sess = tf.Session()
        saver_i2l = tf.train.Saver(var_list=i2l_vars)
        tf.get_default_graph().finalize()

        sess.run(init_ops)

        path_to_i2l_model = model_dir
        ckp_path_i2l = utils.get_latest_model_checkpoint_path(path_to_i2l_model, 'model.ckpt')
        saver_i2l.restore(sess, ckp_path_i2l)

        features = []
        for i in range(images.shape[0]):
            x = np.expand_dims(images[i:i+1, ...], axis=-1)
            features.append(sess.run(feature, feed_dict={images_pl: x}))
        features = np.array(features)
        sess.close()
        return features


def predict_seg(image, model_dir_SD, model_dir_TD=None, patname=None):
    '''
    predict normalized images and masks (before or after TTA).

    '''
    with tf.Graph().as_default():
        images_pl = tf.placeholder(tf.float32, shape=[None] + list((256, 256)) + [1], name='images')

        images_normalized, added_residual = model.normalize(images_pl, config,
                                                            training_pl=tf.constant(False, dtype=tf.bool))
        # images_adapted = model.adaptor(images_normalized)
        logits, softmax, preds, feature = model.predict_i2l(images_normalized, config,
                                                            training_pl=tf.constant(False, dtype=tf.bool),
                                                            returned_feature=None)

        i2l_vars = []
        norm_vars = []

        for v in tf.global_variables():
            var_name = v.name
            if 'image_normalizer' in var_name:
                i2l_vars.append(v)
                norm_vars.append(v)
            if 'i2l_mapper' in var_name:
                i2l_vars.append(v)
            # if 'adaptor' in var_name:
            #     norm_vars.append(v)

        init_ops = tf.global_variables_initializer()
        sess = tf.Session()

        saver_i2l = tf.train.Saver(var_list=i2l_vars)
        saver_norm = tf.train.Saver(var_list=norm_vars)

        tf.get_default_graph().finalize()
        sess.run(init_ops)

        path_to_i2l_model = model_dir_SD
        ckp_path_i2l = utils.get_latest_model_checkpoint_path(path_to_i2l_model, 'model.ckpt')
        saver_i2l.restore(sess, ckp_path_i2l)

        if model_dir_TD is not None and patname is not None:
            path_to_norm_model = model_dir_TD + patname
            ckp_path_norm = utils.get_latest_model_checkpoint_path(path_to_norm_model, 'best_score.ckpt')
            saver_norm.restore(sess, ckp_path_norm)

        mask_predicted = []
        img_normalized = []
        for i in range(image.shape[0]):
            x = np.expand_dims(image[i:i + 1, ...], axis=-1)
            img_normalized.append(sess.run(images_normalized, feed_dict={images_pl: x}))
            mask_predicted.append(sess.run(preds, feed_dict={images_pl: x}))
        img_normalized = np.squeeze(np.array(img_normalized))
        mask_predicted = np.squeeze(np.array(mask_predicted))
        sess.close()
        return img_normalized, mask_predicted


def fg_dice_score(gt_labels, pred_labels):
    score = np.mean(met.f1_score(np.array(gt_labels).flatten(), pred_labels.flatten(), average=None)[1:])
    return score

