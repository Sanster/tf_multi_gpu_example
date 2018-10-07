import argparse
import tensorflow as tf

from model import core_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--gpu_ids', type=str, default='')
    args = parser.parse_args()
    return args


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """

    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device

    return _assign


def training_dataset(epochs=5, batch_size=128):
    from tensorflow.examples.tutorials.mnist import input_data
    mnist_data = input_data.read_data_sets("data")

    # tuple (image, label)
    all_data_points = mnist_data.train.next_batch(60000)

    dataset = tf.data.Dataset.from_tensor_slices(all_data_points)
    dataset = dataset.repeat(epochs).shuffle(10000).batch(batch_size)
    return dataset


def training_model(input_fn):
    # 每块 GPU 训练时，都需要获得自己的数据，所以这里传入一个返回训练数据的函数
    inputs = input_fn()
    image = inputs[0]
    label = tf.cast(inputs[1], tf.int32)
    logits = core_model(image)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label, logits=logits)
    return tf.reduce_mean(loss)


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def do_training(update_op, loss):
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        try:
            step = 0
            while True:
                _, loss_value = sess.run((update_op, loss))
                if step % 100 == 0:
                    print('Step {} with loss {}'.format(step, loss_value))
                step += 1
        except tf.errors.OutOfRangeError:
            # we're through the dataset
            pass
    print('Final loss: {}'.format(loss_value))
