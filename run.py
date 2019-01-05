import argparse
import os
import time

import tensorflow as tf
import logging

import data
import evaluation
import training

import vgg.model
import nmt.model

logging.basicConfig(level=logging.DEBUG)


def run(model_name, dataset=None, config=None):
    """Run training and evaluation for specified model."""

    tf.enable_eager_execution()

    if model_name == "vgg":
        dataset = dataset or "cifar10"
        model_spec = get_model_spec(model_name, dataset, config=config)
    elif model_name == "nmt":
        dataset = dataset or "envi_iwslt32k"
        model_spec = get_model_spec(model_name, dataset, config=config)
    else:
        raise ValueError("%s is not a valid model" % model_name)
    train_dataset, metadata = data.get_data(
        dataset, prep_fn=model_spec.prep,
        preprocess_batch=model_spec.preprocess_batch)
    model = model_spec.construct_model()
    optimizer = model_spec.optimizer()
    training.train(
        train_dataset, model, model_spec.loss,
        optimizer, model_name, model_spec.problem_type)
    test_dataset, _ = data.get_data(dataset, mode="test")
    evaluation.evaluate(
        test_dataset, model, model_spec.loss, model_spec.problem_type)
    return model, metadata


def get_model_spec(model_name, dataset, config=None):
    """Get configuration for the specified model and dataset."""
    if model_name == "vgg":
        config = config or vgg.model.CONFIG
        num_classes = 10 if dataset == "cifar10" else 100
        config["num_classes"] = num_classes
        model_spec = vgg.model.ModelSpec(config, "classification")
    elif model_name == "nmt":
        config = config or nmt.model.CONFIG
        model_spec = nmt.model.ModelSpec(config, "translation")
    return model_spec


def get_tensorboard_url(model, port=6006):
    """Return localtunnel url for viewing tensorboard.

    Used when running model training on colab."""
    get_ipython().system_raw('npm install -g localtunnel')
    log_dir = os.path.join(model, training.LOGS_DIR)
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port {} &'
        .format(log_dir, port)
    )
    # Tunnel port 6006 (TensorBoard assumed running)
    output_file = "url.txt"
    get_ipython().system_raw(
        'lt --port {} >> {} 2>&1 &'.format(port, output_file))
    time.sleep(3)
    with open(output_file, "r") as f:
        for line in f.readlines():
            print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to run')
    parser.add_argument(
        '--dataset', type=str, help='Which dataset to use')

    args = parser.parse_args()
    run(args.model, args.dataset)

