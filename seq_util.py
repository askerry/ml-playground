"""Shared utilities for common sequence manipulations."""

import tensorflow as tf

def pad_to_length(tensor, target_length):
    length = tf.shape(tensor)[0]
    num_zeros = tf.maximum(target_length - length, 0)
    tensor = tf.pad(tensor, [[0, num_zeros]])[:target_length]
    tensor.set_shape([target_length])
    return tensor
