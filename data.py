"""Load specified data as a tf.data.Dataset."""
import os

import numpy as np
import tensorflow.data
from tensorflow.keras.datasets import cifar10, cifar100
from tensor2tensor import problems


def get_data(
        dataset_name,
        mode="train",
        batch_size=256,
        num_epochs=20,
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
    dataset = None
    metadata = {}
    if dataset_name == "cifar10":
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.squeeze(y_train, axis=1).astype(np.int32)
        y_test = np.squeeze(y_test, axis=1).astype(np.int32)
    elif dataset_name == "cifar100":
        (x_train, y_train), (x_test, y_test) = cifar100.load_data()
        x_train = x_train.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = np.squeeze(y_train, axis=1).astype(np.int32)
        y_test = np.squeeze(y_test, axis=1).astype(np.int32)
    elif dataset_name == "envi_iwslt32k":
        data_dir = os.path.join("data", dataset_name)
        tmp_dir = os.path.join("data", dataset_name + "_tmp")
        problem = problems.problem("translate_envi_iwslt32k")
        problem.generate_data(data_dir, tmp_dir)
        dataset = problem.dataset(mode, data_dir)
        metadata["encoders"] = problem.feature_encoders(data_dir)
        metadata["max_length"] = 64
    else:
        raise ValueError("Unknown dataset: %s" % dataset_name)

    if prep_fn:
        if dataset:
            dataset, metadata = prep_fn(dataset, metadata)
        else:
            x_train, y_train, x_test, y_test, metadata = prep_fn(
                x_train, y_train, x_test, y_test, metadata)

    if dataset is None:
        if mode == "train":
            x, y = x_train, y_train
        elif mode == "test":
            x, y = x_test, y_test
        else:
            ValueError("Invalid mode: %s" % mode)
        dataset = tensorflow.data.Dataset.from_tensor_slices(
            {"inputs": x, "targets": y})
        dataset = dataset.repeat(num_epochs).shuffle(buffer_size=500)

    drop_remainder = mode == "train"
    dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
    if preprocess_batch:
        dataset = preprocess_batch(dataset, metadata)
    return dataset, metadata
