import tensorflow as tf
import tensorflow.contrib.eager as tfe


def evaluate(dataset, model, loss_fn):
    """Compute eval metrics on the provided test dataset."""
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
