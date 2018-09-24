"""Shared utilities for common image manipulations."""

import numpy as np
import tensorflow as tf


def flip_image_batch(images, horizontal_axis=2):
    """Randomly flips images horizontally."""
    return tf.image.random_flip_left_right(images)


def max_min_scale(images):
    """Apply max-min scaling to transform to 0-1 range."""
    return (images - tf.reduce_min(images)) / (
        tf.reduce_max(images) - tf.reduce_min(images))


def image_pca(images):
    """Compute PCA over all images in dataset."""
    # Collapse across batch and spatial dimensions to get a tensor
    # of all pixel RGB values
    scaled = max_min_scale(images)
    reshaped = tf.reshape(scaled, (-1, 3))
    cov_mat = np.cov(reshaped, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    return eig_vals, eig_vecs


def color_shift_batch(images, eig_vals, eig_vecs):
    """Applies a color shift to a batch of RGB images.

    Implements the PCA-based color shifting augmentation described in
    Krizhevsky et al., 2012."""
    # To each training image, add a multiple of the principal components,
    # proportional to corresponding eigenvalues, scaled by a random variable
    # drawn from a normal distribution with mean 0, standard deviation 0.1.
    batch_size = tf.shape(images)[0]
    num_channels = 3
    alphas = tf.random_normal((batch_size, num_channels), mean=0, stddev=.1)
    shift = tf.tensordot(tf.to_float(eig_vecs),
                         tf.transpose(eig_vals * alphas),
                         axes=[[1], [0]])
    return images + tf.reshape(
        tf.transpose(shift), (batch_size, 1, 1, num_channels))
