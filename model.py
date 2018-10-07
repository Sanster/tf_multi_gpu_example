import tensorflow as tf


def core_model(input_img, num_classes=10):
    """
        A simple model to perform classification on 28x28 grayscale images in MNIST style.

        Args:
        input_img:  A floating point tensor with a shape that is reshapable to batchsizex28x28. It
            represents the inputs to the model
        num_classes:  The number of classes
    """
    net = tf.reshape(input_img, [-1, 28, 28, 1])
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_1")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)

    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=[5, 5],
                           padding="same", activation=tf.nn.relu,
                           name="conv2d_2")
    net = tf.layers.max_pooling2d(inputs=net, pool_size=[2, 2], strides=2)
    net = tf.reshape(net, [-1, 7 * 7 * 64])
    net = tf.layers.dense(inputs=net, units=1024, name="dense_1", activation=tf.nn.relu)
    logits = tf.layers.dense(inputs=net, units=num_classes, name="dense_2")
    return logits
