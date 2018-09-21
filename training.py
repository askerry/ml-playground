import os

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def train(dataset,
          model,
          loss_fn,
          optimizer,
          model_dir,
          log_frequency=100,
          checkpoint_frequency=100):
    """Trains the specified network.

    Given a tf.data.Dataset, a model configuration, a loss function,
    and a optimizer, run a training loop and log metrics.
    """

    log_file = os.path.join(model_dir, "logs")
    writer = tf.contrib.summary.create_file_writer(log_file)
    step_counter = tf.train.get_or_create_global_step()
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        for (x, y) in dataset:
            batch_num = step_counter.numpy()
            with tf.GradientTape() as tape:

                # Compute forward_pass and loss
                forward_pass = model(x, training=True)
                loss_value = loss_fn(forward_pass, y)

                # Write output logs and summary values
                tf.contrib.summary.scalar('loss', loss_value)
                accuracy = tfe.metrics.Accuracy()
                class_predictions = tf.argmax(
                    forward_pass, axis=1, output_type=tf.int32)
                accuracy(labels=y, predictions=class_predictions)
                tf.contrib.summary.scalar('Accuracy', accuracy.result())
                if batch_num % log_frequency == 0:
                    print('Step #%d\tLoss: %.4f, Accuracy: %.4f' % (
                        batch_num, loss_value, accuracy.result()))

                if batch_num % checkpoint_frequency == 0:
                    write_checkpoint(model, step_counter, model_dir)

                # Apply gradients to update weights
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)

    write_checkpoint(model, step_counter, model_dir)
    return model


def write_checkpoint(model, global_step, model_dir):
    """Write a snapshot of the current model to disk."""
    print("Writing model checkpoint on step %s" % global_step.numpy())
    checkpoint_dir = os.path.join(model_dir, "checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    tfe.Saver(model.variables).save(
        checkpoint_prefix, global_step=global_step)


def load_latest_checkpoint(
        model, model_dir, x_shape, checkpoint_dirname="checkpoints"):
    """Read the latest snapshot of a model from disk."""
    checkpoint_dir = os.path.join(model_dir, checkpoint_dirname)
    # HACK: variables must be initialized for them to be properly
    # restored from the checkpoint, so we do a dummy forward pass
    # to initialize the model variables
    dummy_x = np.zeros((1,) + x_shape[1:], dtype=np.float32)
    model(tfe.Variable(dummy_x, dtype=np.float32), training=True)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    print("Loading model from latest checkpoint: %s" % latest_checkpoint)
    tfe.Saver(model.variables).restore(latest_checkpoint)
    return model
