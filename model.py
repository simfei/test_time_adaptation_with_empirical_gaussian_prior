import tensorflow as tf
from tfwrapper import losses
import matplotlib
import matplotlib.cm
import networks


# model
def normalize(images, exp_config, training_pl):
    images_normalized, added_residual = exp_config.model_handle_normalizer(images, exp_config, training_pl)
    return images_normalized, added_residual


def one_layer_adaptor(images):
    images_adapted = networks.one_layer_adaptor(images)
    return images_adapted


def predict_i2l(images, exp_config, training_pl, returned_feature='NI'):
    logits, features = exp_config.model_handle_i2l(images,
                                                   nlabels=exp_config.nlabels,
                                                   training_pl=training_pl,
                                                   returned_feature=returned_feature)

    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)
    return logits, softmax, mask, features


def autoencoder(images, midplane, outplane, training_pl, name):
    out = networks.unet2D_AE(images, midplane, outplane, training_pl, name)
    return out


# training loss
# ---------------------------------------------------------------------------------------------------------
def loss(logits, labels, nlabels, loss_type, mask_for_loss_with_mask=None, are_labels_1hot=False):
    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)

    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
    elif loss_type is 'crossentropy_reverse':
        predicted_probabilities = tf.nn.softmax(logits)
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels)
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
    elif loss_type == 'dice_with_mask':
        if mask_for_loss_with_mask is not None:
            segmentation_loss = losses.dice_loss_within_mask(logits, labels, mask_for_loss_with_mask)
    else:
        raise ValueError("Unknown loss: %s" % loss_type)

    return segmentation_loss


def ae_loss(in_features, out_features):
    loss = losses.ae_train_loss(in_features, out_features)
    return loss
# ----------------------------------------------------------------------------------------------------------

# tta loss
# ----------------------------------------------------------------------------------------------------------
# def tta_loss_EGP(features, mean_mean, var_mean, mean_var, var_var):
#     return losses.egp_sw_loss(features, mean_mean, var_mean, mean_var, var_var)

# def tta_loss_EGP(features, means, vars):
#     return losses.egp_loss(features, means, vars)


def tta_loss_EGP(features, means, covar_inv):
    return losses.egp_ndc_loss(features, means, covar_inv)


def tta_loss_AE(in_feature_list, out_feature_list):
    return losses.ae_tta_loss(in_feature_list, out_feature_list)
# -----------------------------------------------------------------------------------------------------------


def training_step(loss, var_list, optimizer_handle, learning_rate, update_bn_nontrainable_vars, return_optimizer=False):
    optimizer = optimizer_handle(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, var_list=var_list)

    if update_bn_nontrainable_vars is True:
        opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, opt_memory_update_ops])

    if return_optimizer is True:
        return train_op, optimizer
    else:
        return train_op


def evaluate_losses(logits, labels, nlabels, loss_type, are_labels1hot=False):
    '''loss between prediction and ground truth'''

    supervised_loss = loss(logits=logits, labels=labels, nlabels=nlabels,
                           loss_type=loss_type, are_labels_1hot=are_labels1hot)
    if are_labels1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)
    dice_all_imgs_all_labels, mean_dice, mean_dice_fg = losses.compute_dice(logits, labels)
    return supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg


def evaluation_i2l(logits, labels, images, nlabels, loss_type):
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits, labels, nlabels, loss_type)

    mask = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
    mask_gt = labels

    gt1 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=1, nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=2, nlabels=nlabels)

    pred1 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=1, nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=2, nlabels=nlabels)

    img1 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=0, nlabels=nlabels)
    img2 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=1, nlabels=nlabels)
    img3 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=2, nlabels=nlabels)

    tf.summary.image('example_labels_true', tf.concat([gt1, gt2, gt3], axis=0))
    tf.summary.image('example_labels_pred', tf.concat([pred1, pred2, pred3], axis=0))
    tf.summary.image('example_images', tf.concat([img1, img2, img3], axis=0))

    return supervised_loss, mean_dice


def prepare_tensor_for_summary(img, mode, n_idx_batch=0, n_idx_z=60, nlabels=None):
    if mode == 'mask':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1,-1,-1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, n_idx_z, 0, 0), (1,1,-1,-1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1,-1,-1,1,1))
        else:
            raise ValueError("Don't know how to deal with input dimension %d" % img.get_shape().ndims)

    elif mode == 'image':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else:
            raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else:
        raise ValueError("Unknown mode: %s. Must be image or mask" % mode)

    if mode == 'image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)

    V *= 255
    V = tf.cast(V, dtype=tf.uint8)  # (1,224,224)
    V = tf.squeeze(V)
    V = tf.expand_dims(V, axis=0)

    # gather
    if mode == 'mask':
        cmap = 'viridis'
        cm = matplotlib.cm.get_cmap(cmap)
        colors = tf.constant(cm.colors, dtype=tf.float32)
        V = tf.gather(colors, tf.cast(V, dtype=tf.int32))  # (1,224,224,3)

    elif mode == 'image':
        V = tf.reshape(V, tf.stack((-1, tf.shape(img)[1], tf.shape(img)[2], 1)))  # (1,224,224,1)

    return V
