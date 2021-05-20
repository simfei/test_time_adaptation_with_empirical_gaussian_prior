import tensorflow as tf
from tfwrapper import layers


# normalization network
def net2D_i2i(images, exp_config, training):
    with tf.variable_scope('image_normalizer'):   # a context manager for defining ops that create variables(layers)
        num_layers = exp_config.norm_num_hidden_layers
        n1 = exp_config.norm_num_filters_per_layer
        k = exp_config.norm_kernel_size

        out = images

        for l in range(num_layers):
            out = tf.layers.conv2d(inputs=out, filters=n1, kernel_size=k,
                                   padding='SAME', name='norm_conv1_'+str(l+1),
                                   use_bias=True, activation=None)
            if exp_config.norm_batch_norm is True:
                out = tf.layers.batch_normalization(inputs=out, name='norm_conv1_'+str(l+1)+'_bn', training=training)
            if exp_config.norm_activation is 'elu':
                out = tf.nn.elu(out)
            if exp_config.norm_activation is 'relu':
                out = tf.nn.relu(out)
            if exp_config.norm_activation is 'rbf':
                init_value = tf.random_normal([1,1,1,n1], mean=0.2, stddev=0.05)
                scale = tf.Variable(initial_value=init_value, name='scale_'+str(l+1))
                out = tf.exp(-(out**2)/(scale**2))

        delta = tf.layers.conv2d(inputs=out, filters=1, kernel_size=k,
                                 padding='SAME', name='norm_conv1_'+str(num_layers+1),
                                 use_bias=True, activation=tf.identity)

        out = images + delta
    return out, delta


def one_layer_adaptor(images):
    with tf.variable_scope('adaptor'):
        out = images
        out = tf.layers.conv2d(inputs=out, filters=1, kernel_size=1,
                               padding='SAME', name='conv1_1',
                               use_bias=True, activation=None,
                               kernel_initializer=tf.keras.initializers.Ones())
    return out


# segmentation network
def unet2D_i2l(images, nlabels, training_pl, returned_feature='NI'):
    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0

    with tf.variable_scope('i2l_mapper'):
        # down
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)

        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)

        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer2d(conv3_2)

        # middle
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)

        # up
        deconv3 = layers.bilinear_upsampling2D(conv4_2, size=(tf.shape(conv3_2)[1], tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2], axis=-1)
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training=training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training=training_pl)

        deconv2 = layers.bilinear_upsampling2D(conv5_2, size=(tf.shape(conv2_2)[1], tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2], axis=-1)
        conv6_1 = layers.conv2D_layer_bn(concat2, name='conv6_1', num_filters=n2, training=training_pl)
        conv6_2 = layers.conv2D_layer_bn(conv6_1, name='conv6_2', num_filters=n2, training=training_pl)

        deconv1 = layers.bilinear_upsampling2D(conv6_2, size=(tf.shape(conv1_2)[1], tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)
        conv7_1 = layers.conv2D_layer_bn(concat1, name='conv7_1', num_filters=n1, training=training_pl)
        conv7_2 = layers.conv2D_layer_bn(conv7_1, name='conv7_2', num_filters=n1, training=training_pl)

        pred = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

        if returned_feature == 'NI':
            feature = images
        elif returned_feature == 'conv1_2':
            feature = conv1_2
        elif returned_feature == 'conv2_2':
            feature = conv2_2
        elif returned_feature == 'conv3_2':
            feature = conv3_2
        elif returned_feature == 'conv4_2':
            feature = conv4_2
        elif returned_feature == 'conv5_2':
            feature = conv5_2
        elif returned_feature == 'conv6_2':
            feature = conv6_2
        elif returned_feature == 'conv7_2':
            feature = conv7_2
        elif returned_feature == 'all':
            feature = [images, conv1_2, conv2_2, conv3_2, conv4_2, conv5_2, conv6_2, conv7_2]
        elif returned_feature is None:
            feature = None
        else:
            raise ValueError('Wrong feature type!')
    return pred, feature


def unet2D_AE(images, midplane=[64, 32, 16], outplane=128, training_pl=True, name='autoencoder_x'):
    n1, n2, n3 = midplane[0], midplane[1], midplane[2]

    with tf.variable_scope(name):
        # down
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)

        pool1 = layers.max_pool_layer2d(conv1_2)
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)

        # middle
        pool2 = layers.max_pool_layer2d(conv2_2)
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)

        # up
        deconv2 = layers.bilinear_upsampling2D(conv3_2, size=(tf.shape(conv2_2)[1], tf.shape(conv2_2)[2]), name='upconv2')
        conv4_1 = layers.conv2D_layer_bn(deconv2, name='conv4_1', num_filters=n2, training=training_pl)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, name='conv4_2', num_filters=n2, training=training_pl)

        deconv1 = layers.bilinear_upsampling2D(conv4_2, size=(tf.shape(conv1_2)[1], tf.shape(conv1_2)[2]), name='upconv1')
        conv5_1 = layers.conv2D_layer_bn(deconv1, name='conv5_1', num_filters=n1, training=training_pl)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, name='conv5_2', num_filters=n1, training=training_pl)

        out = layers.conv2D_layer(x=conv5_2, name='out', num_filters=outplane, kernel_size=1)

        return out
