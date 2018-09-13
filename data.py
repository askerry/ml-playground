"""Load specified data as a tf.data.Dataset."""

import tensorflow.data
from tensorflow.keras.datasets import cifar10, cifar100


def get_data_iterator(
        dataset,
        mode="train",
        batch_size=32,
        num_epochs=200,
        prep_fn=None):
    """
    Construct a tf.data.Iterator for the specified dataset.

    Args:
        dataset: string representing the dataset to load
        mode: string ("train" or "test") representing mode in which to run
        batch_size: integer representing size of batch
        prep_fn: optional preprocessing function that takes a tf.data.Dataset
            and returns a preprocessed Dataset.
    Returns:
        A tf.data.Iterator to be consumed by training or eval loops
    """
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    if dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    else:
        ValueError("Unknown dataset: %s" % dataset)

    if mode == "train":
        x, y = x_train, y_train
    elif mode == "test":
        x, y = x_test, y_test
    else:
        ValueError("Invalid mode: %s" % mode)

    dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
    if prep_fn:
        dataset = prep_fn(dataset)
    drop_remainder = mode == "train"
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    return dataset.make_one_shot_iterator()
