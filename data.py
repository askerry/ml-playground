"""Load specified data as a tf.data.Dataset."""

import numpy as np
import tensorflow.data
from tensorflow.keras.datasets import cifar10, cifar100


def get_data(
        dataset,
        mode="train",
        batch_size=256,
        num_epochs=10,
        prep_fn=None,
        preprocess_batch=None):
    """
    Construct a tf.data.Dataset for the specified dataset.

    Args:
        dataset: string representing the dataset to load
        mode: string ("train" or "test") representing mode in which to run
        batch_size: integer representing size of batch
        prep_fn: optional preprocessing function that takes a tf.data.Dataset
            and returns a preprocessed Dataset.
    Returns:
        A tf.data.Dataset to be consumed by training or eval loops
    """
    if dataset == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.squeeze(y_train, axis=1).astype(np.int32)
        y_test = np.squeeze(y_test, axis=1).astype(np.int32)
    elif dataset == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.squeeze(y_train, axis=1).astype(np.int32)
        y_test = np.squeeze(y_test, axis=1).astype(np.int32)
    else:
        raise ValueError("Unknown dataset: %s" % dataset)

    if prep_fn:
        x_train, y_train, x_test, y_test, metadata = prep_fn(
            x_train, y_train, x_test, y_test)

    if mode == "train":
        x, y = x_train, y_train
    elif mode == "test":
        x, y = x_test, y_test
    else:
        ValueError("Invalid mode: %s" % mode)

    dataset = tensorflow.data.Dataset.from_tensor_slices((x, y))
    drop_remainder = mode == "train"
    dataset = dataset.repeat(num_epochs).shuffle(buffer_size=500)
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if preprocess_batch:
        dataset = preprocess_batch(dataset, metadata)
    return dataset
