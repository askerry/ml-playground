"""Load specified data as a tf.data.Dataset."""
import os
import re

import numpy as np
import tensorflow as tf
import tensorflow.data
from tensorflow.keras.datasets import cifar10, cifar100
from tensor2tensor import problems


def get_data(
        dataset_name,
        mode="train",
        batch_size=256,
        num_epochs=20,
        prep_fn=None,
        preprocess_batch=None,
        metadata=None):
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
    if metadata is None:
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
    elif dataset_name in ["envi_iwslt32k", "enfr_wmt_small8k"]:
        data_dir = os.path.join("data", dataset_name)
        tmp_dir = os.path.join("data", dataset_name + "_tmp")
        t2t_name = "translate_%s" % dataset_name
        problem = problems.problem(t2t_name)
        problem.generate_data(data_dir, tmp_dir)
        dataset = problem.dataset(mode, data_dir)
        metadata["problem"] = problem
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


def translate(model, problem, sentences, length):
    """Provide translation for list of sentences using provided model."""
    input_encoder = problem.feature_info["inputs"].encoder
    target_encoder = problem.feature_info["targets"].encoder
    end_token_id = 1
    sentences = [preprocess_sentence(s) for s in sentences]
    input_ids = [input_encoder.encode(s) + [end_token_id] for s in sentences]
    input_ids = tf.keras.preprocessing.sequence.pad_sequences(
        input_ids, maxlen=length, padding='post')
    predicted_ids, loss_value = model(
        tf.convert_to_tensor(input_ids), training=False, batch_size=len(input_ids))
    return [target_encoder.decode(p) for p in predicted_ids]


def _preprocess_sentence(sentence):
    sentence = sentence.lower().strip()
    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    return sentence.rstrip().strip()
