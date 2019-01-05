from collections import OrderedDict
import datetime
import logging
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
          problem_type,
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
        for element in dataset:
            x = element["inputs"]
            y = element["targets"]
            batch_num = step_counter.numpy()
            with tf.GradientTape() as tape:

                def classification_batch():
                    """Train step for standard classification problems"""
                    logits = model(x, training=True)
                    loss_value = loss_fn(logits, y)
                    accuracy = tfe.metrics.Accuracy()
                    class_predictions = tf.argmax(
                        logits, axis=1, output_type=tf.int32)
                    accuracy(labels=y, predictions=class_predictions)
                    return class_predictions, OrderedDict(
                        (("loss", loss_value), ("accuracy", accuracy.result())))
                def translation_batch():
                    """Train step for common translation problems"""
                    predictions, loss_value = model(x, y, loss_fn, training=True)
                    approx_bleu = bleu_score(predictions, y)
                    return predictions, OrderedDict(
                        (("loss", loss_value), ("approx_bleu_score", approx_bleu)))
                if problem_type == "translation":
                    predictions, metrics = translation_batch()
                elif problem_type == "classification":
                    predictions, metrics = classification_batch()
                else:
                    raise ValueError("No problem type %s" % problem_type)

                # Write output logs and summary values
                for metric_name, metric in metrics.items():
                    tf.contrib.summary.scalar(metric_name, metric)

                if batch_num % log_frequency == 0:
                    logging.debug("example predictions: ", predictions[0])
                    logging.debug("SUM: ", tf.reduce_sum(predictions))
                    minutes_passed = (time.time() - start_time) / 60
                    metric_str = ", ".join(
                        ["%s: %.4f" % item for item in metrics.items()])
                    logging.info('(%d mins) Step #%d\t %s' % (
                        minutes_passed, batch_num, metric_str))
                    logging.info("......................................\n\n")

                if batch_num % checkpoint_frequency == 0:
                    write_checkpoint(model, step_counter, model_dir, training_id)

                # Apply gradients to update weights
                grads = tape.gradient(metrics["loss"], model.variables)
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)

    write_checkpoint(model, step_counter, model_dir, training_id)
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
