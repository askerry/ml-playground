import argparse

import tensorflow as tf

import data
import evaluation
import training

import vgg.model


def run(model_name, dataset=None):
    """Run training and evaluation for specified model."""

    tf.enable_eager_execution()

    if model_name == "vgg":
        if dataset is None:
            dataset = "cifar10"
        model_spec = get_model_spec(model_name, dataset)
    else:
        raise ValueError("%s is not a valid model" % model_name)
    train_dataset = data.get_data(
        dataset, prep_fn=model_spec.prep,
        preprocess_batch=model_spec.preprocess_batch)
    model = model_spec.construct_model()
    optimizer = model_spec.optimizer()
    training.train(
        train_dataset, model, model_spec.loss, optimizer, model_name)
    test_dataset = data.get_data(dataset, mode="test")
    evaluation.evaluate(test_dataset, model, model_spec.loss)


def get_model_spec(model_name, dataset):
    """Get configuration for the specified model and dataset."""
    if model_name == "vgg":
        config = vgg.model.CONFIG
        num_classes = 10 if dataset == "cifar10" else 100
        config["num_classes"] = num_classes
        model_spec = vgg.model.ModelSpec(config)
    return model_spec


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to run')
    parser.add_argument(
        '--dataset', type=str, help='Which dataset to use')

    args = parser.parse_args()
    run(args.model, args.dataset)

