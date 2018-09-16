import data
import evaluation
import training

import vgg.model

import argparse


def run(model, dataset=None):
    """Run training and evaluation for specified model."""
    if model == "vgg":
        if dataset is None:
            dataset = "cifar10"
        # TODO: calculate num classes from the dataset directly
        num_classes = 10 if dataset == "cifar10" else 100
        train_dataset = data.get_data(dataset, prep_fn=vgg.model.prep)
        model = vgg.model.construct_model(
            vgg.model.CONFIG, num_classes=num_classes)
        optimizer = vgg.model.optimizer(vgg.model.CONFIG)
        training.train(train_dataset, model, vgg.model.loss, optimizer)
        test_dataset = data.get_data(dataset, mode="test")
        evaluation.evaluate(test_dataset, model, vgg.model.loss)
    else:
        raise ValueError("%s is not a valid model" % model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to run')
    parser.add_argument(
        '--dataset', type=str, help='Which dataset to use')

    args = parser.parse_args()
    run(args.model, args.dataset)

