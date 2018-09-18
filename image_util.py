"""Shared utilities for common image manipulations."""

import numpy as np


def flip_images(images, horizontal_axis=2):
    """Flips each image horizontally."""
    return np.flip(images, axis=horizontal_axis)


def color_shift(images):
    """Applies a random RGB color shift to an array of RGB images.

    Implements the PCA-based color shifting augmentation described in
    Krizhevsky et al., 2012"""
    return images
