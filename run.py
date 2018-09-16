import data
import vgg.training
import vgg.evaluation
import vgg.preprocess

import argparse


def run(model):
    """Run training and evaluation for specified model."""
    if model == "vgg":
        train_dataset = data.get_data("cifar10", prep_fn=vgg.preprocess.prep)
        model = vgg.training.train(train_dataset)
        test_dataset = data.get_data("cifar10", mode="test")
        vgg.evaluation.evaluate(model, test_dataset)
    else:
        raise ValueError("%s is not a valid model" % model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to run')

    args = parser.parse_args()
    run(args.model)

