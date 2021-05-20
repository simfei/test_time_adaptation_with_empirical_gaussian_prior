import tensorflow as tf

def max_pool_layer2d(x, kernel_size=2, strides=2, padding="SAME"):
    op = tf.layers.max_pooling2d(inputs=x, pool_size=kernel_size,
                                 strides=strides, padding=padding)
    return op


def conv2D_layer(x, name, kernel_size=3, num_filters=32,
                 strides=1, padding='SAME'):
    conv = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=kernel_size,
                            padding=padding, name=name, use_bias=False)
    return conv


def conv2D_layer_bn(x, name, training, kernel_size=3, num_filters=32,
                    strides=1, activation=tf.nn.relu, padding='SAME'):
    conv = tf.layers.conv2d(inputs=x, filters=num_filters, kernel_size=kernel_size,
                            padding=padding, name=name, use_bias=False)
    conv_bn = tf.layers.batch_normalization(inputs=conv, name=name+'_bn', training=training)
    act = activation(conv_bn)
    return act


def deconv2D_layer_bn(x, name, training, kernel_size=3, num_filters=32,
                      strides=2, activation=tf.nn.relu, padding='SAME'):
    conv = tf.layers.conv2d_transpose(inputs=x, filters=num_filters, kernel_size=kernel_size,
                                      padding=padding, name=name, strides=strides, use_bias=False)
    conv_bn = tf.layers.batch_normalization(inputs=conv, name=name+'_bn', training=training)
    act = activation(conv_bn)
    return act


def bilinear_upsampling2D(x, size, name):
    x_reshaped = tf.image.resize_bilinear(x, size, name=name)
    return x_reshaped
