import tensorflow as tf
import tensorflow.contrib.eager as tfe

import vgg.training as training


def evaluate(model, dataset):
    """Compute eval metrics on the provded test dataset."""
    mean_loss = tfe.metrics.Mean()
    accuracy = tfe.metrics.Accuracy()
    for x, y in dataset:
        logits = model(x, training=False)
        predictions = tf.argmax(logits, axis=1, output_type=tf.int32)
        loss = training.loss(logits, y)
        mean_loss(loss)
        accuracy(
            labels=y,
            predictions=predictions)

    print("Metrics on eval set: Loss=%.4f, Accuracy=%.4f" % (
        mean_loss.result().numpy(), accuracy.result().numpy()))
