import data
import evaluation
import training

import vgg.model

import argparse


def run(model):
    """Run training and evaluation for specified model."""
    if model == "vgg":
        train_dataset = data.get_data("cifar10", prep_fn=vgg.model.prep)
        model = vgg.model.construct_model(vgg.model.CONFIG)
        optimizer = vgg.model.optimizer(vgg.model.CONFIG)
        training.train(train_dataset, model, vgg.model.loss, optimizer)
        test_dataset = data.get_data("cifar10", mode="test")
        evaluation.evaluate(test_dataset, model, vgg.model.loss)
    else:
        raise ValueError("%s is not a valid model" % model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to run')

    args = parser.parse_args()
    run(args.model)

