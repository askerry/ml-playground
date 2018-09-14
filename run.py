import data
import vgg.training
import vgg.evaluation


def run(model="vgg"):
    """Run training and evaluation for specified model."""
    if model == "vgg":
        train_dataset = data.get_data("cifar10")
        model = vgg.training.train(train_dataset)
        test_dataset = data.get_data("cifar10", mode="test")
        vgg.evaluation.evaluate(model, test_dataset)
    else:
        raise ValueError("%s is not a valid model" % model)


if __name__ == "__main__":
    run()
