import tensorflow as tf
import numpy as np
from scipy.stats import entropy

def compute_dice(logits, labels, epsilon=1e-10):
    with tf.name_scope('dice'):
        prediction = tf.nn.softmax(logits)
        intersection = tf.multiply(prediction, labels)

        reduction_axes = [1,2]

        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes)
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes)
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)

        # compute dice (per image per label)
        dice = 2*tp / (tp_plus_fp+tp_plus_fn+epsilon)

        # mean over all images in the batch and over all labels
        mean_dice = tf.reduce_mean(dice)

        # mean over all images in the batch and over all foreground labels
        mean_fg_dice = tf.reduce_mean(dice[:,1:])

    return dice, mean_dice, mean_fg_dice


def compute_dice_3d_without_batch_axis(prediction,
                                       labels,
                                       epsilon=1e-10):
    with tf.name_scope('dice_3d_without_batch_axis'):
        intersection = tf.multiply(prediction, labels)

        reduction_axes = [0, 1, 2]

        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes)
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes)
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)

        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)

        # mean over all images in the batch and over all labels.
        mean_fg_dice = tf.reduce_mean(dice[1:])

    return mean_fg_dice


def dice_loss(logits, labels):
    with tf.name_scope('dice_loss'):
        _, mean_dice, mean_fg_dice = compute_dice(logits, labels)

        loss = 1 - mean_dice
    return loss


def dice_loss_within_mask(logits, labels, mask):
    with tf.name_scope('dice_loss_within_mask'):
        _, mean_dice, mean_fg_dice = compute_dice(tf.math.multiply(logits, mask),
                                                  tf.math.multiply(labels, mask))

        # loss = 1 - mean_fg_dice
        loss = 1 - mean_dice

    return loss


def pixel_wise_cross_entropy_loss(logits, labels):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    return loss


def pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels):
    labels_copy = np.copy(labels)
    labels_copy = labels_copy + 1e-20
    labels_copy = labels_copy / tf.expand_dims(tf.reduce_sum(labels_copy, axis=-1), axis=-1)

    loss = - tf.reduce_mean(tf.reduce_sum(predicted_probabilities*tf.math.log(labels_copy), axis=-1))

    return loss


def ae_train_loss(in_features, out_features):
    # MSE loss used for training AEs
    with tf.name_scope('ae_loss'):
        loss = tf.reduce_mean((in_features-out_features)**2)
    return loss


def egp_loss(features, means, vars, lambda_=0):
    # loss for one-dimensional Gaussian prior (with penalty on variance)
    with tf.name_scope('tta_loss'):
        losses = 0
        for i, fea in enumerate(features):
            test_means, test_vars = tf.nn.moments(fea, axes=[0, 1, 2])
            loss1 = [(test_means[j]-means[i][j])**2 / (vars[i][j]+1e-10) for j in range(len(means[i]))]
            loss1 = tf.reduce_mean(loss1)
            loss2 = [(test_vars[j]-vars[i][j])**2 for j in range(len(vars[i]))]
            loss2 = tf.reduce_mean(loss2)
            losses += loss1 + lambda_*loss2
        return losses


def egp_ndc_loss(features, means, covar_inv):
    # loss for multi-dimensional Gaussian prior with non-diagonal covariance
    with tf.name_scope('tta_loss'):
        losses = 0
        for i, feature in enumerate(features):
            if means[i].shape != (feature.shape[-1], 1):
                prior_means = tf.cast(tf.reshape(means[i], [means[i].shape[0], 1]), tf.float32)
            else:
                prior_means = means[i]
            test_means = tf.reduce_mean(feature, axis=[0,1,2])
            test_means = tf.reshape(test_means, [feature.shape[-1], 1])

            temp = tf.linalg.matmul(tf.subtract(test_means, prior_means), covar_inv[i], transpose_a=True)
            loss = tf.linalg.matmul(temp, (test_means-prior_means))
            loss = tf.reshape(loss, [])
            losses += loss

        return losses


def egp_sw_loss(features, mean_mean, var_mean, mean_var, var_var):
    # loss for empirical Gaussian prior over SD subjects
    with tf.name_scope('tta_loss'):
        losses = 0
        for i, fea in enumerate(features):
            test_means, test_vars = tf.nn.moments(fea, axes=[0,1,2])
            loss1 = [((test_means[j]-mean_mean[i][j])**2 / (var_mean[i][j]+1e-10)) for j in range(len(mean_mean[i]))]
            loss1 = tf.reduce_mean(loss1)
            loss2 = [((test_vars[j]-mean_var[i][j])**2 / (var_var[i][j]+1e-10)) for j in range(len(mean_var[i]))]
            loss2 = tf.reduce_mean(loss2)
            losses += (loss1+0.1*loss2)
        return losses


def ae_tta_loss(in_features_list, out_features_list):
    # loss used for TTA with AEs
    with tf.name_scope('tta_loss'):
        losses = 0
        for i in range(len(out_features_list)):
            mean_in, var_in = tf.nn.moments(in_features_list[i], axes=[0,1,2])
            mean_out, var_out = tf.nn.moments(out_features_list[i], axes=[0,1,2])
            losses += tf.reduce_mean((mean_in-mean_out)**2/var_in)

    return losses






