import tensorflow as tf
from helper import parse_args, training_dataset, training_model, do_training


def create_optimization(model_fn, input_fn, optimizer):
    loss = model_fn(input_fn)
    global_step = tf.train.get_or_create_global_step()
    update_op = optimizer.minimize(loss, global_step=global_step)

    return update_op, loss


def main(args):
    dataset = training_dataset(epochs=2)
    iterator = dataset.make_one_shot_iterator()

    optimizer = tf.train.AdamOptimizer(learning_rate=1E-3)

    update_op, loss = create_optimization(training_model, iterator.get_next, optimizer)

    do_training(update_op, loss)


if __name__ == '__main__':
    args = parse_args()
    main(args)
    training_dataset()
