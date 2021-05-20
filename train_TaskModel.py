import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import model as model
import h5py

import data_hcp as data_hcp
import train_config as config
from data_aug import augment_images_and_labels


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
log_dir = 'logdir_brain/'
model_dir = 'models/'

def run_training(continue_run):
    init_step = 0

    if continue_run:
        logging.info('============================================================')
        logging.info('Continuing previous run')
        try:
            init_checkpoint_path = utils.get_latest_model_checkpoint_path(log_dir+model_dir, 'model.ckpt')
            logging.info('Checkpoint path: %s' % init_checkpoint_path)
            init_epoch = int(
                init_checkpoint_path.split('/')[-1].split('-')[-1]) + 1  # plus 1 as otherwise starts with eval
            logging.info('Latest step was: %d' % init_step)
        except:
            logging.warning('Did not find init checkpoint. Maybe first run failed. Disabling continue mode...')
            continue_run = False
            init_epoch = 0
        logging.info('============================================================')

    # load data
    logging.info('Loading data...')
    if config.train_dataset is 'HCPT1':
        logging.info('Reading HCPT1 images...')
        logging.info('Data root directory: ' + config.orig_data_root_hcp)
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder=config.orig_data_root_hcp,
                                                                preprocessing_folder=config.preproc_folder_hcp,
                                                                idx_start=0,
                                                                idx_end=20,
                                                                protocol='T1',
                                                                size=config.image_size,
                                                                depth=config.image_depth_hcp,
                                                                target_resolution=config.target_resolution_brain)
        imtr, gttr = [data_brain_train['images'], data_brain_train['labels']]

        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder=config.orig_data_root_hcp,
                                                              preprocessing_folder=config.preproc_folder_hcp,
                                                              idx_start=20,
                                                              idx_end=25,
                                                              protocol='T1',
                                                              size=config.image_size,
                                                              depth=config.image_depth_hcp,
                                                              target_resolution=config.target_resolution_brain)
        imvl, gtvl = [data_brain_val['images'], data_brain_val['labels']]
    if config.train_dataset is 'NCI':
        filename = 'data/data_2d_size_256_256_res_0.625_0.625_cv_fold_1.hdf5'
        with h5py.File(filename, 'r') as f:
            imtr, gttr = [np.array(f['images_train']),
                          np.array(f['masks_train'])]
            imvl, gtvl = [np.array(f['images_validation']),
                          np.array(f['masks_validation'])]

    print(imtr.shape, gttr.shape, imvl.shape, gtvl.shape)

    # build TF graph
    with tf.Graph().as_default():
        tf.random.set_random_seed(config.run_number)
        np.random.seed(config.run_number)

        # create placeholder
        logging.info('Creating placeholder...')
        image_tensor_shape = [config.batch_size] + list(config.image_size) + [1]
        mask_tensor_shape = [config.batch_size] + list(config.image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name='images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name='labels')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name='learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name='training_or_testing')

        # I2NI
        images_normalized, _ = model.normalize(images_pl, config, training_pl)

        # NI2S
        logits, _, _, _ = model.predict_i2l(images_normalized, config, training_pl)

        print('shape of inputs: ', images_pl.shape)
        print('shape of logits: ', logits.shape)

        # create a list of all vars that must be optimized wrt
        i2l_vars = []
        for v in tf.trainable_variables():
            i2l_vars.append(v)

        # create ops
        loss_op = model.loss(logits, labels_pl, nlabels=config.nlabels, loss_type=config.loss_type_i2l)
        tf.summary.scalar('loss', loss_op)
        train_op = model.training_step(loss_op, i2l_vars, config.optimizer_handle, learning_rate_pl,
                                       update_bn_nontrainable_vars=True)
        eval_loss = model.evaluation_i2l(logits, labels_pl, images_pl, nlabels=config.nlabels, loss_type=config.loss_type_i2l)
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

        # summaries of the validation errors
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # summaries of the training errors
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])

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

        if continue_run:
            # Restore session
            logging.info('============================================================')
            logging.info('Restroring session from: %s' % init_checkpoint_path)
            saver.restore(sess, init_checkpoint_path)

        epoch = init_epoch
        curr_lr = config.learning_rate
        best_dice = 0

        # run training epochs
        while (epoch < config.max_epochs):
            # batches
            start_time = time.time()
            running_loss = 0
            steps = 0
            for batch in iterate_minibatches(imtr, gttr, batch_size=config.batch_size, train_or_eval='train'):
                x, y = batch
                if y.shape[0] < config.batch_size:
                    continue

                feed_dict = {images_pl: x,
                             labels_pl: y,
                             learning_rate_pl: curr_lr,
                             training_pl: True}

                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)
                running_loss += loss
                steps += 1

            duration = time.time() - start_time
            epoch_loss = running_loss / steps
            # save summary
            if (epoch + 1) % config.summary_writing_frequency == 0:
                print('Epoch %d: loss = %.3f (%.3f sec for the last epoch)' % (epoch + 1, epoch_loss, duration))
                summary_str = sess.run(summary, feed_dict=feed_dict)
                summary_writer.add_summary(summary_str, epoch)
                summary_writer.flush()

            if epoch % config.train_eval_frequency == 0:
                print('Training')
                print('-----------------------')
                train_loss, train_dice = do_eval(sess, eval_loss, images_pl, labels_pl,
                                                 training_pl, imtr, gttr, config.batch_size)
                tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss,
                                                                 tr_dice: train_dice})
                summary_writer.add_summary(tr_summary_msg, epoch)

            # save checkpoint
            if epoch % config.save_frequency == 0:
                checkpoint_file = os.path.join(log_dir+model_dir, 'model.ckpt')
                saver.save(sess, checkpoint_file, global_step=epoch)

            # evaluate the model
            if epoch % config.val_eval_frequency == 0:
                print('Validation')
                print('-----------------------')
                val_loss, val_dice = do_eval(sess, eval_loss, images_pl, labels_pl, training_pl,
                                             imvl, gtvl, config.batch_size)

                vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss,
                                                                 vl_dice: val_dice})
                summary_writer.add_summary(vl_summary_msg, epoch)
                print()

            # save model if val dice is the best yet
            if val_dice > best_dice:
                best_dice = val_dice
                best_file = os.path.join(log_dir+model_dir, 'best_dice.ckpt')
                saver_best_dice.save(sess, best_file, global_step=epoch)
            epoch += 1

        sess.close()


def do_eval(sess, eval_loss, images_placeholder, labels_placeholder, training_time_placeholder,
            images, labels, batch_size):
    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images, labels, batch_size, train_or_eval='eval'):
        x, y = batch
        if y.shape[0]<batch_size:
            continue
        feed_dict = {images_placeholder: x,
                     labels_placeholder: y,
                     training_time_placeholder: False}

        loss, fg_dice = sess.run(eval_loss, feed_dict=feed_dict)

        loss_ii += loss
        dice_ii += fg_dice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    print('loss: %.4f, dice: %.4f' % (avg_loss, avg_dice))
    return avg_loss, avg_dice


def iterate_minibatches(images, labels, batch_size, train_or_eval='train'):
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    for b_i in range(n_images // batch_size):
        if b_i+batch_size > n_images:
            continue

        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        x = images[batch_indices,...]
        y = labels[batch_indices,...]

        # data augmentation
        if config.da_ratio > 0:
            if train_or_eval is 'train' or train_or_eval is 'eval':
                x, y = augment_images_and_labels(images=x, labels=y,
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
        yield x, y


# ==================================================================
def main():
    continue_run = config.continue_run
    if not tf.gfile.Exists(log_dir):
        tf.gfile.MakeDirs(log_dir)
    if not tf.gfile.Exists(log_dir+model_dir):
        tf.gfile.MakeDirs(log_dir + model_dir)
        continue_run = False

    shutil.copy(config.__file__, log_dir)

    run_training(continue_run)


if __name__ == '__main__':
    main()
