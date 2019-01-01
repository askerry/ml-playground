import datetime
import os
import time

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

LOGS_DIR = "logs"
CHECKPOINT_DIR = "checkpoints"


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

    start = datetime.datetime.now()
    training_id = "%s-%s" % (model_dir, start.strftime("%Y%m%d.%H%M"))
    log_file = os.path.join(model_dir, LOGS_DIR, training_id)
    writer = tf.contrib.summary.create_file_writer(log_file)
    step_counter = tf.train.get_or_create_global_step()
    start_time = time.time()
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
                    minutes_passed = (time.time() - start_time) / 60
                    print('(%d mins) Step #%d\tLoss: %.4f, Accuracy: %.4f' % (
                        minutes_passed, batch_num, loss_value, accuracy.result()))

                if batch_num % checkpoint_frequency == 0:
                    write_checkpoint(model, step_counter, model_dir, training_id)

                # Apply gradients to update weights
                grads = tape.gradient(loss_value, model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)

    write_checkpoint(model, step_counter, model_dir)
    return model


def write_checkpoint(model, global_step, model_dir, training_id):
    """Write a snapshot of the current model to disk."""
    print("Writing model checkpoint on step %s" % global_step.numpy())
    checkpoint_dir = os.path.join(model_dir, CHECKPOINT_DIR, training_id)
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tfe.Saver(model.variables).save(
        checkpoint_prefix, global_step=global_step)


def load_checkpoint(
        model, x_shape,
        checkpoint_dir=None, checkpoint_file=None):
    """Read a snapshot of the specified model from disk.

    If a checkpoint file is not specified, defaults to latest in the provided
    directory. If a specific checkpoint_file is not provided, a checkpoint_dir
    must be specified.
    """
    if checkpoint_file is None:
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)

    # HACK: variables must be initialized for them to be properly
    # restored from the checkpoint, so we do a dummy forward pass
    # to initialize the model variables
    dummy_x = np.zeros((1,) + x_shape[1:], dtype=np.float32)
    model(tfe.Variable(dummy_x, dtype=np.float32), training=True)
    print("Loading model from checkpoint: %s" % checkpoint_file)
    tfe.Saver(model.variables).restore(checkpoint_file)
    return model
