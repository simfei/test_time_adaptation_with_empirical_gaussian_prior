import tensorflow as tf
import numpy as np
import model as model
import tta_config as config
import utils
import data_hcp as data_hcp
import data_prostate_pirad_erc as data_usz
import logging


def run_training(log_dir, images,
                 log_dir_first_TD_subject='', patname=None):

    init_epoch = 0

    tf.reset_default_graph()
    with tf.Graph().as_default():
        tf.random.set_random_seed(config.run_number)
        np.random.seed(config.run_number)

        # set placeholders
        images_pl = tf.placeholder(tf.float32, shape=[config.batch_size]+list(config.image_size)+[1], name='images')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name='training_or_testing')

        # create model functions
        images_normalized, add_residual = model.normalize(images_pl, config, training_pl)
        # images_adapted = model.one_layer_adaptor(images_normalized)
        logits, softmax, preds, features = model.predict_i2l(images_normalized, config,
                                                             training_pl=tf.constant(False, dtype=tf.bool),
                                                             returned_feature=feature_type)

        pred_features0 = model.autoencoder(features[0], [32, 16, 8], 1,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_NI')
        pred_features1 = model.autoencoder(features[1], [64, 32, 16], 16,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv1_2')
        pred_features2 = model.autoencoder(features[2], [64, 32, 16], 32,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv2_2')
        pred_features3 = model.autoencoder(features[3], [64, 32, 16], 64,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv3_2')
        pred_features4 = model.autoencoder(features[4], [64, 32, 16], 128,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv4_2')
        pred_features5 = model.autoencoder(features[5], [64, 32, 16], 64,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv5_2')
        pred_features6 = model.autoencoder(features[6], [64, 32, 16], 32,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv6_2')
        pred_features7 = model.autoencoder(features[7], [64, 32, 16], 16,
                                           training_pl=tf.constant(False, dtype=tf.bool), name='autoencoder_conv7_2')
        pred_features_list = [pred_features0, pred_features1, pred_features2, pred_features3,
                              pred_features4, pred_features5, pred_features6, pred_features7,
                              ]

        # divide variables
        norm_vars = []
        i2l_vars = []
        ae_vars0 = []
        ae_vars1 = []
        ae_vars2 = []
        ae_vars3 = []
        ae_vars4 = []
        ae_vars5 = []
        ae_vars6 = []
        ae_vars7 = []
        for v in tf.global_variables():
            var_name = v.name
            if 'image_normalizer' in var_name:
                i2l_vars.append(v)
                norm_vars.append(v)
            if 'i2l_mapper' in var_name:
                i2l_vars.append(v)
            # if 'adaptor' in var_name:
            #     norm_vars.append(v)
            if 'autoencoder_NI' in var_name:
                ae_vars0.append(v)
            if 'autoencoder_conv1_2' in var_name:
                ae_vars1.append(v)
            if 'autoencoder_conv2_2' in var_name:
                ae_vars2.append(v)
            if 'autoencoder_conv3_2' in var_name:
                ae_vars3.append(v)
            if 'autoencoder_conv4_2' in var_name:
                ae_vars4.append(v)
            if 'autoencoder_conv5_2' in var_name:
                ae_vars5.append(v)
            if 'autoencoder_conv6_2' in var_name:
                ae_vars6.append(v)
            if 'autoencoder_conv7_2' in var_name:
                ae_vars7.append(v)

        # add loss op
        loss_op = model.tta_loss_AE(features, pred_features_list)
        loss_pl = tf.placeholder(tf.float32, shape=[], name='tta_loss')
        l_summary = tf.summary.scalar('tr_losses/tta_loss', loss_pl)
        loss_summary = tf.summary.merge([l_summary])

        # add train op
        train_op = model.training_step(loss_op, norm_vars, config.optimizer_handle, learning_rate_pl, update_bn_nontrainable_vars=True)

        # add init ops
        init_ops = tf.global_variables_initializer()
        uninit_vars = tf.report_uninitialized_variables()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(logdir_TD, sess.graph)

        # create savers
        saver_i2l = tf.train.Saver(var_list=i2l_vars)
        saver_AEs = []
        for vars in [ae_vars0, ae_vars1, ae_vars2, ae_vars3, ae_vars4, ae_vars5, ae_vars6, ae_vars7]:
            saver_AEs.append(tf.train.Saver(var_list=vars))
        saver_test_data = tf.train.Saver(var_list=norm_vars, max_to_keep=1)
        saver_best_loss = tf.train.Saver(var_list=norm_vars, max_to_keep=1)

        # freeze the graph before execution
        tf.get_default_graph().finalize()

        # initialize the variables
        sess.run(init_ops)

        # print names of uninitialized variables
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        # Restore the segmentation network parameters and the pre-trained i2i mapper parameters
        path_to_model = logdir_SD + model_dir
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'model.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)

        # restore AE network
        for i, logdir_AE in enumerate(logdir_AEs):
            checkpoint_path_ae = utils.get_latest_model_checkpoint_path(logdir_AE, 'model.ckpt')
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path_ae)
            saver_AEs[i].restore(sess, checkpoint_path_ae)

        if log_dir_first_TD_subject is not '':
            logging.info('============================================================')
            path_to_model = log_dir + feature_type + '/' + log_dir_first_TD_subject + '/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_score.ckpt')
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
            init_step = int(checkpoint_path.split('/')[-1].split('-')[-1])
            logging.info('Latest step was: %d' % init_step)
            saver_test_data.restore(sess, checkpoint_path)
            max_epochs_tta = config.max_epochs
        else:
            max_epochs_tta = 3 * config.max_epochs

        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)
        if not tf.gfile.Exists(log_dir+patname):
            tf.gfile.MakeDirs(log_dir+patname)

        # run training steps
        epoch = init_epoch
        best_loss = float('inf')
        while (epoch < (max_epochs_tta+init_epoch)):
            running_loss = 0
            steps = 0
            curr_lr = config.learning_rate
            for image in iterate_minibatches_images(images, batch_size=config.batch_size):
                feed_dict = {images_pl: image,
                             learning_rate_pl: curr_lr,
                             training_pl: True}
                _, loss = sess.run([train_op,loss_op], feed_dict=feed_dict)
                running_loss += loss
                steps += 1
            epoch_loss = running_loss / steps
            print('Epoch %d: loss = %.3f' % (epoch + 1, epoch_loss))
            loss_summary_msg = sess.run(loss_summary, feed_dict={loss_pl: epoch_loss})
            summary_writer.add_summary(loss_summary_msg, epoch)

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_file = log_dir + patname + '/best_score.ckpt'
                saver_best_loss.save(sess, best_file, global_step=epoch)
            epoch += 1

        sess.close()


def iterate_minibatches_images(images, batch_size):
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    for b_i in range(n_images // batch_size):
        if b_i + batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i * batch_size:(b_i + 1) * batch_size])
        x = images[batch_indices, ...]
        x = np.expand_dims(x, axis=-1)
        yield x


if __name__ == '__main__':
    # -----------------------
    logdir_SD = 'logdir_brain/'         # logdir of task model
    model_dir = 'models/'               # task model dir
    logdir_AEs = ['barin_AEs/AE_NI/', 'barin_AEs/AE_conv1_2/', 'barin_AEs/AE_conv2_2/', 'barin_AEs/AE_conv3_2/',
                  'barin_AEs/AE_conv4_2/', 'barin_AEs/AE_conv5_2/', 'barin_AEs/AE_conv6_2/', 'barin_AEs/AE_conv7_2/',
                  ]                     # list of AE model dirs
    logdir_TD = 'HCPT2/'                # dir of adapted models for all subjects
    feature_type = 'all'                # use AEs trained on all intermediate feature layers for TTA, including the normalized image.
    if 'HCPT2' in logdir_TD:
        idx_start = 40
        idx_end = 45
        protocol = 'T2'
    elif 'ABIDE' in logdir_TD:
        idx_start = 16
        idx_end = 36
        protocol = 'T1'
    elif 'USZ' in logdir_TD:
        idx_start = 0
        idx_end = 20
    # -----------------------

    if not tf.gfile.Exists(logdir_TD):
        tf.gfile.MakeDirs(logdir_TD)

    if 'HCPT2' in logdir_TD or 'ABIDE' in logdir_TD:
        data_test = data_hcp.load_and_maybe_process_data(input_folder='',
                                                         preprocessing_folder=config.preproc_folder_test,
                                                         idx_start=idx_start,
                                                         idx_end=idx_end,
                                                         protocol=protocol,
                                                         size=config.image_size,
                                                         depth=config.image_depth_hcp,
                                                         target_resolution=config.target_resolution_brain)
    elif 'USZ' in logdir_TD:
        data_test =data_usz.load_data(input_folder='',
                                      preproc_folder=config.preproc_folder_test,
                                      idx_start=idx_start,
                                      idx_end=idx_end,
                                      size=config.image_size,
                                      target_resolution=config.target_resolution_prostate,
                                      labeller='ek')

    images = data_test['images']
    labels = data_test['labels']
    patnames = np.array(data_test['patnames']).astype(str)
    nz = list(data_test['nz'])
    pat_end_slice = []
    pat_end_slice.append(0)
    for i in range(len(nz)):
        pat_end_slice.append(np.sum(nz[:(i+1)]))
    first_patname = patnames[0]

    for i in range(5):
        if i==0:
            run_training(logdir_TD,
                         images[pat_end_slice[i]:pat_end_slice[i+1], ...],
                         log_dir_first_TD_subject='',
                         patname=patnames[i],
                         )
        else:
            run_training(logdir_TD,
                         images[pat_end_slice[i]:pat_end_slice[i+1],...],
                         log_dir_first_TD_subject=first_patname,
                         patname=patnames[i],
                         )
