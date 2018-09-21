import argparse

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import data
import run
import training


def evaluate(dataset, model, loss_fn):
    """Compute eval metrics on the provided test dataset."""
    # TODO: allow for passing custom metrics
    mean_loss = tfe.metrics.Mean()
    accuracy = tfe.metrics.Accuracy()
    for x, y in dataset:
        forward_pass = model(x, training=False)
        predictions = tf.argmax(forward_pass, axis=1, output_type=tf.int32)
        loss = loss_fn(forward_pass, y)
        mean_loss(loss)
        accuracy(
            labels=y,
            predictions=predictions)

    print("Metrics on eval set: Loss=%.4f, Accuracy=%.4f" % (
        mean_loss.result().numpy(), accuracy.result().numpy()))


def eval_latest_checkpoint(model_name, dataset_name, checkpoint_dirname):
    """Run evaluation on the test set using the latest model checkpoint."""
    tf.enable_eager_execution()
    config, model_module = run.get_config(model_name, dataset_name)
    test_dataset = data.get_data(dataset_name, mode="test")
    x_shape = tuple(test_dataset.output_shapes[0].as_list())
    model = model_module.construct_model(config)
    model = training.load_latest_checkpoint(
        model, model_name, x_shape, checkpoint_dirname=checkpoint_dirname)
    accuracy = tfe.metrics.Accuracy()
    for batch, (x, y) in enumerate(test_dataset):
        forward_pass = model(x, training=False)
        predictions = tf.argmax(forward_pass, axis=1, output_type=tf.int32)
        accuracy(
            labels=y,
            predictions=predictions)
        if batch % 100 == 0:
            print("Batch %s: accuracy on eval set=%.4f" % (
                batch, accuracy.result().numpy()))
    print("Final accuracy on eval set=%.4f" % accuracy.result().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', type=str, help='The model you want to evaluate')
    parser.add_argument(
        '--dataset', type=str, help='Which dataset to use')
    parser.add_argument(
        '--checkpoint_dirname', type=str,
        help='Name of directory to read checkpoint from', default="checkpoints")

    args = parser.parse_args()
    eval_latest_checkpoint(args.model, args.dataset, args.checkpoint_dirname)
