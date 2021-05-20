import tensorflow as tf
import numpy as np
import model as model
import tta_config as config
import utils
import data_hcp as data_hcp
import data_prostate_pirad_erc as data_usz
import logging


def run_training(log_dir,
                 images,
                 means, vars,
                 log_dir_first_TD_subject='',
                 patname=None,
                 ):

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
        # images_adapted = model.adaptor(images_normalized)
        logits, softmax, preds, features = model.predict_i2l(images_normalized, config,
                                                             training_pl=tf.constant(False, dtype=tf.bool),
                                                             returned_feature=feature_type)

        # divide variables
        norm_vars = []
        i2l_vars = []
        for v in tf.global_variables():
            var_name = v.name
            if 'image_normalizer' in var_name:
                norm_vars.append(v)
                i2l_vars.append(v)
            if 'i2l_mapper' in var_name:
                i2l_vars.append(v)

        # add loss op
        loss_op = model.tta_loss_EGP(features, means, vars)
        loss_pl = tf.placeholder(tf.float32, shape=[], name='tta_loss')
        l_summary = tf.summary.scalar('tr_losses/tta_loss', loss_pl)
        loss_summary = tf.summary.merge([l_summary])

        # accumulated gradient
        # --------------------------------------------------------------------------------------------------------------
        optimizer = config.optimizer_handle(learning_rate=learning_rate_pl)
        accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in norm_vars]
        accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accumulated_gradients]
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss_op, var_list=norm_vars)
        accumulated_gradients_op = [ac.assign(gg[0]) for ac, gg in zip(accumulated_gradients, gradients)]
        num_accumulation_steps_pl = tf.placeholder(dtype=tf.float32, name='num_accumulation_steps')
        accumulated_gradients_mean_op = [ag.assign(tf.divide(ag, num_accumulation_steps_pl)) for ag in accumulated_gradients]
        final_gradients = [(ag, gg[1]) for ag, gg in zip(accumulated_gradients, gradients)]
        train_op = optimizer.apply_gradients(final_gradients)
        # --------------------------------------------------------------------------------------------------------------

        # add train op
        # train_op = model.training_step(loss_op, norm_vars, config.optimizer_handle, learning_rate_pl, update_bn_nontrainable_vars=True)

        # add init ops
        init_ops = tf.global_variables_initializer()
        uninit_vars = tf.report_uninitialized_variables()

        sess = tf.Session()
        summary_writer = tf.summary.FileWriter(logdir_TD, sess.graph)

        # create savers
        saver_i2l = tf.train.Saver(var_list=i2l_vars)
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
        # After the adaptation for the 1st TD subject is done, start the adaptation for the subsequent subjects with those parameters
        path_to_model = logdir_SD + model_dir
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'model.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)

        if log_dir_first_TD_subject is not '':
            logging.info('============================================================')
            path_to_model = log_dir + feature_type + '/' + log_dir_first_TD_subject + '/'
            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_score.ckpt')
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
            init_epoch = int(checkpoint_path.split('/')[-1].split('-')[-1])
            logging.info('Latest epoch was: %d' % init_epoch)
            saver_test_data.restore(sess, checkpoint_path)
            max_epochs_tta = config.max_epochs
        else:
            max_epochs_tta = 5*config.max_epochs

        if not tf.gfile.Exists(log_dir):
            tf.gfile.MakeDirs(log_dir)
        if not tf.gfile.Exists(log_dir+'/'+patname):
            tf.gfile.MakeDirs(log_dir+'/'+patname)

        # run training steps
        epoch = init_epoch
        best_loss = float('inf')
        while (epoch < (max_epochs_tta+init_epoch)):
            # accumulated gradient optimization
            # ----------------------------------------------------------------------------------------------------------
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
            curr_lr = config.learning_rate
            for image in iterate_minibatches_images(images, batch_size=config.batch_size):
                feed_dict = {images_pl: image,
                             learning_rate_pl: curr_lr,
                             training_pl: True}
                sess.run(accumulated_gradients_op, feed_dict=feed_dict)
                num_accumulation_steps += 1
            sess.run(accumulated_gradients_mean_op, feed_dict={num_accumulation_steps_pl: num_accumulation_steps})
            sess.run(train_op, feed_dict=feed_dict)

            # compute epoch loss after optimization
            # ----------------------------------------------------------------------------------------------------------
            running_loss = 0
            steps = 0
            for image in iterate_minibatches_images(images, batch_size=config.batch_size):
                feed_dict = {images_pl: image,
                             training_pl: False}
                loss = sess.run(loss_op, feed_dict=feed_dict)
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
    logdir_SD = 'logdir_brain/'
    model_dir = 'models/'
    logdir_TD = 'HCPT2/'
    feature_type = 'all'
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

    feature_types = ['NI', 'conv1_2', 'conv2_2', 'conv3_2', 'conv4_2', 'conv5_2', 'conv6_2', 'conv7_2']
    means = []
    vars = []
    # mean_mean = []
    # var_mean = []
    # mean_var = []
    # var_var = []
    for fea_type in feature_types:
        print(fea_type)
        means.append(np.load('priors_hcpt1_da/means_{}_da.npy'.format(fea_type)))
        vars.append(np.load('priors_hcpt1_da/covar_inv_{}_da.npy'.format(fea_type)))
        # mean_mean.append(np.load('priors_hcpt1_da/mean_mean_{}_da.npy'.format(fea_type)))
        # var_mean.append(np.load('priors_hcpt1_da/var_mean_{}_da.npy'.format(fea_type)))
        # mean_var.append(np.load('priors_hcpt1_da/mean_var_{}_da.npy'.format(fea_type)))
        # var_var.append(np.load('priors_hcpt1_da/var_var_{}_da.npy'.format(fea_type)))

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
    # labels = data_test['labels']
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
                         means, vars,
                         log_dir_first_TD_subject='',
                         patname=patnames[i])
        else:
            run_training(logdir_TD,
                         images[pat_end_slice[i]:pat_end_slice[i+1], ...],
                         means, vars,
                         log_dir_first_TD_subject=first_patname,
                         patname=patnames[i])

