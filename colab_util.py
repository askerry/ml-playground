"""Small utilities for running models in a colab notebook."""
from google.colab import drive
import os
from shutil import copyfile
import time

import tensorflow as tf

import training

def save_checkpoint_to_gdrive(model_name, run=None, checkpoint=None, drive_name="My Drive"):
    """Saves model checkpoint to a mounted google drive."""
    drive.mount('/content/gdrive')
    model_dir = "/content/gdrive/%s/Colab Checkpoints/%s/" % (drive_name, model_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
    if run is None:
        runs = sorted(os.listdir(os.path.join(model_name, "checkpoints")))
        run = runs[-1]
    if not os.path.exists(model_dir + run):
        os.mkdir(model_dir + run)
    if checkpoint is None:
        checkpoint_dir = os.path.join(model_name, "checkpoints", run)
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    else:
        checkpoint_file = os.path.join(
            model_name, "checkpoints", run, checkpoint)
    files = [f for f in os.listdir(checkpoint_dir) if f.startswith(
        checkpoint_file.split("/")[-1])]
    for file in files:
        src = os.path.join(model_name, "checkpoints", run, file)
        dest = os.path.join(model_dir, run, file)
        print("writing checkpoint to %s" % dest)
        copyfile(src, dest)


def load_model_from_gdrive_checkpoint(
        model_name, model, x_shape, run=None,
        checkpoint=None, drive_name="My Drive"):
    """Loads model checkpoint from a mounted google drive."""
    drive.mount('/content/gdrive')
    drive_dir = "/content/gdrive/%s/Colab Checkpoints/%s" % (
        drive_name, model_name)
    if checkpoint:
        checkpoint_file = os.path.join(drive_dir, run, checkpoint)
    else:
        if run is None:
            runs = os.listdir(drive_dir)
            run = runs[-1]
        checkpoint_dir = os.path.join(drive_dir, run)
        checkpoint_file = tf.train.latest_checkpoint(checkpoint_dir)
    model = training.load_checkpoint(model, x_shape, checkpoint_file=checkpoint_file)
    return model

def get_tensorboard_url(model, port=6006):
    """Return localtunnel url for viewing tensorboard.

    Used when running model training on colab."""
    get_ipython().system_raw('npm install -g localtunnel')
    log_dir = os.path.join(model, training.LOGS_DIR)
    get_ipython().system_raw(
        'tensorboard --logdir {} --host 0.0.0.0 --port {} &'
        .format(log_dir, port)
    )
    # Tunnel port 6006 (TensorBoard assumed running)
    output_file = "url.txt"
    get_ipython().system_raw(
        'lt --port {} >> {} 2>&1 &'.format(port, output_file))
    time.sleep(3)
    with open(output_file, "r") as f:
        for line in f.readlines():
            print(line)
