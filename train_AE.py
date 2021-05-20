import logging
import os.path
import time
import tensorflow as tf
import numpy as np
import utils
import model as model
import h5py

import data_hcp as data_hcp
import train_config as config
from predict import predict_features
from data_aug import augment_images


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

def run_training(log_dir, model_dir, name):
    init_epoch = 0

    # load data
    logging.info('Loading data...')
    if config.train_dataset is 'HCPT1':
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder=config.orig_data_root_hcp,
                                                                preprocessing_folder=config.preproc_folder_hcp,
                                                                idx_start=0,
                                                                idx_end=20,
                                                                protocol='T1',
                                                                size=config.image_size,
                                                                depth=config.image_depth_hcp,
                                                                target_resolution=config.target_resolution_brain)
        imtr, _ = [data_brain_train['images'], data_brain_train['labels']]
        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder=config.orig_data_root_hcp,
                                                              preprocessing_folder=config.preproc_folder_hcp,
                                                              idx_start=40,
                                                              idx_end=45,
                                                              protocol='T1',
                                                              size=config.image_size,
                                                              depth=config.image_depth_hcp,
                                                              target_resolution=config.target_resolution_brain)
        imvl, _ = [data_brain_val['images'], data_brain_val['labels']]
    elif config.train_dataset is 'NCI':
        filename = 'data/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5'
        with h5py.File(filename, 'r') as f:
            imtr = np.array(f['images_train'])
            imvl = np.array(f['images_validation'])

    print(imtr.shape, imvl.shape)
    # build TF graph
    with tf.Graph().as_default():
        tf.random.set_random_seed(config.run_number)
        np.random.seed(config.run_number)

        # create placeholder
        logging.info('Creating placeholder...')
        midplane = [64, 32, 16]
        if name == 'NI':
            image_size = [256, 256, 1]
            midplane = [32, 16, 8]
        elif name == 'conv1_2' or name == 'conv7_2':
            image_size = [256, 256, 16]
        elif name == 'conv2_2' or name == 'conv6_2':
            image_size = [128, 128, 32]
        elif name == 'conv3_2' or name == 'conv5_2':
            image_size = [64, 64, 64]
        elif name == 'conv4_2':
            image_size = [32, 32, 128]
        elif name == 'y':
            image_size = [256, 256, 15]
            midplane = [32, 16, 8]
        image_tensor_shape = [config.batch_size] + image_size
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='labels')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name='training_or_testing')

        # functions
        features = model.autoencoder(images_pl, midplane, image_size[-1], training_pl,
                                     name='autoencoder_{}'.format(name))

        print('shape of features: ', features.shape)

        # create a list of all vars that must be optimized wrt
        ae_vars = []
        for v in tf.trainable_variables():
            ae_vars.append(v)

        # create ops
        loss_op = model.ae_loss(labels_pl, features)
        tf.summary.scalar('loss', loss_op)
        train_op = model.training_step(loss_op, ae_vars, config.optimizer_handle, learning_rate_pl,
                                       update_bn_nontrainable_vars=True)
        eval_loss = model.ae_loss(labels_pl, features)
        summary = tf.summary.merge_all()

        # add init ops
        init_ops = tf.global_variables_initializer()

        logging.info('Adding the op to get a list of initialized variables...')
        uninit_vars = tf.report_uninitialized_variables()

        # create saver
        saver = tf.train.Saver(max_to_keep=2)
        saver_best_dice = tf.train.Saver(max_to_keep=2)

        # create session
        sess = tf.Session()

        # create a summary writer
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        # freeze the graph before execution
        logging.info('Freezing the graph now!')
        tf.get_default_graph().finalize()

        logging.info('============================================================')
        logging.info('initializing all variables...')
        sess.run(init_ops)

        logging.info('============================================================')
        logging.info('This is the list of all variables:')
        for v in tf.trainable_variables(): print(v.name)

        logging.info('============================================================')
        logging.info('This is the list of uninitialized variables:')
        uninit_variables = sess.run(uninit_vars)
        for v in uninit_variables: print(v)

        epoch = init_epoch
        curr_lr = config.learning_rate
        best_score = 0

        # run training epochs
        while (epoch < config.max_epochs):
            # batches
            start_time = time.time()
            running_loss = 0
            steps = 0
            for batch in iterate_minibatches(imtr, batch_size=config.batch_size):
                if batch.shape[0] < config.batch_size:
                    continue

                x = predict_features(batch, base_logdir + base_model, name)

                feed_dict = {images_pl: x,
                             labels_pl: x,
                             learning_rate_pl: curr_lr,
                             training_pl: True}

                # opt step
                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
                running_loss += loss
                steps += 1

            duration = time.time() - start_time
            epoch_loss = running_loss / steps
            print('Epoch %d: loss = %.3f (%.3f sec for the last step)' % (epoch + 1, epoch_loss, duration))

            # save summary
            if (epoch + 1) % config.summary_writing_frequency == 0:
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()


            # save checkpoint
            if epoch % config.save_frequency == 0:
                checkpoint_file = os.path.join(log_dir + model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)

            # evaluate the model
            if epoch % config.val_eval_frequency == 0:
                val_loss = do_eval(sess, eval_loss, images_pl, labels_pl, training_pl,
                                   imvl, config.batch_size)
                val_score = np.exp(-val_loss)
                print()

            # save model if val dice is the best yet
            if val_score > best_score:
                best_score = val_score
                best_file = os.path.join(log_dir + model_dir, 'best_score.ckpt')
                saver_best_dice.save(sess, best_file, global_step=epoch)
            epoch += 1

        sess.close()


def do_eval(sess, eval_loss, images_placeholder, labels_placeholder, training_time_placeholder,
            images, batch_size):
    loss_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images, batch_size):
        x = batch
        if x.shape[0]<batch_size:
            continue

        feed_dict = {images_placeholder: x,
                     labels_placeholder: x,
                     training_time_placeholder: False}

        loss = sess.run(eval_loss, feed_dict=feed_dict)

        loss_ii += loss
        num_batches += 1

    avg_loss = loss_ii / num_batches

    print('val loss: %.4f' % (avg_loss))
    return avg_loss


def iterate_minibatches(images, batch_size):
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    for b_i in range(n_images // batch_size):
        if b_i+batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        x = images[batch_indices,...]
        # data augmentation
        if config.da_ratio > 0:
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
        yield x


if __name__ == '__main__':
    base_logdir = 'logdir_brain/'     # logdir of the task model
    base_model = 'models/'            # task model dir

    log_dir = 'brain_AEs/'            # logdir of the AEs
    feature_type = 'conv7_2'          # feature type
    model_dir = 'AE_{}/'.format(feature_type)       # AE model dir

    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    if not tf.gfile.Exists(log_dir+model_dir):
        tf.gfile.MakeDirs(log_dir+model_dir)

    run_training(log_dir, model_dir, feature_type)
