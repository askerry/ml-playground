import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


def train(dataset, model, loss_fn, optimizer):
    """Trains the specified network.

    Given a tf.data.Dataset, a model configuration, a loss function,
    and a optimizer, run a training loop and log metrics.
    """

    writer = tf.contrib.summary.create_file_writer("vgg/logs")
    step_counter = tf.train.get_or_create_global_step()
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        for (batch, (images, labels)) in enumerate(dataset):
            with tf.GradientTape() as tape:

                # Compute forward_pass and loss
                forward_pass = model(images, training=True)
                loss_value = loss_fn(forward_pass, labels)

                # Write output logs and summary values
                tf.contrib.summary.scalar('loss', loss_value)
                accuracy = tfe.metrics.Accuracy()
                accuracy(
                    labels=labels,
                    predictions=tf.argmax(forward_pass, axis=1, output_type=tf.int32))
                tf.contrib.summary.scalar('accuracy', accuracy.result())
                if batch % 100 == 0:
                    print('Step #%d\tLoss: %.4f, Accuracy: %.4f' % (
                        batch, loss_value, accuracy.result()))

                # Apply gradients to update weights
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
    return model
