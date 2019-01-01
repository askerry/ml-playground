"""Specifies architecture and training configuration for VGG architecture."""

import tensorflow as tf
import tensorflow.contrib.layers

import image_util
import interfaces

layers = tf.keras.layers

# Training and architectural parameters from the best performing
# model in the paper.

CONFIG = {
    "batch_size": 256,
    "num_epochs": 10,

    # Optimized using mini-batch gradient descent with momentum of .9
    "momentum": .9,

    # The learning rate was initially set to .01 and then decreased by
    # a factor of 10 when the validation accuracy stopped improving
    "initial_learning_rate": .01,
    "learning_rate_factor": 10,

    # Regularized by weight decay (L2) and dropout
    "l2_multiplier": .0005,
    "dropout_ratio": .5,

    # Convolutional filters use a 3 × 3 receptive field
    "filter_size": 3,
    "convolutional_stride": 1,

    # Max-pooling is performed over a 2 × 2 pixel window
    "max_pooling_window": 2,

    # Number of channels in initial convolutional layers
    "num_initial_channels": 64,

    # Fully-connected layers have 4096 channels (except final
    # soft-max layer with has the same number of channels as classes)
    "num_fc_channels": 4096,
}


class ModelSpec(interfaces.ModelBase):

    def construct_model(self):
        """Returns VGG model with the specified parameters."""
        data_format = "channels_last"

        pooling_window = self.config["max_pooling_window"]
        max_pool_layer = layers.MaxPooling2D(
            (pooling_window, pooling_window),
            (pooling_window, pooling_window),
            padding='same',
            data_format=data_format)

        def conv_layer(channel_factor):
            # We generally want to keep the dimensionality fixed, so as we
            # apply max pooling, we increase the number of channels by the
            # same factor.
            num_channels = self.config["num_initial_channels"] * (
                self.config["max_pooling_window"]**channel_factor)
            l2 = tensorflow.contrib.layers.l2_regularizer
            return layers.Conv2D(
                num_channels,
                self.config["filter_size"],
                strides=self.config["convolutional_stride"],
                padding='same',
                data_format=data_format,
                activation=tf.nn.relu,
                kernel_regularizer=l2(self.config["l2_multiplier"]))

        model_layers = [
            conv_layer(0),
            conv_layer(0),
            max_pool_layer,
            conv_layer(1),
            conv_layer(1),
            max_pool_layer,
            conv_layer(2),
            conv_layer(2),
            conv_layer(2),
            conv_layer(2),
            max_pool_layer,
            conv_layer(3),
            conv_layer(3),
            conv_layer(3),
            conv_layer(3),
            max_pool_layer,
            conv_layer(3),
            conv_layer(3),
            conv_layer(3),
            conv_layer(3),
            max_pool_layer,
            layers.Flatten(),
            layers.Dense(self.config["num_fc_channels"], activation=tf.nn.relu),
            layers.Dropout(self.config["dropout_ratio"]),
            layers.Dense(self.config["num_fc_channels"], activation=tf.nn.relu),
            layers.Dropout(self.config["dropout_ratio"]),
            layers.Dense(self.config["num_classes"])
        ]
        return tf.keras.Sequential(model_layers)

    def optimizer(self):
        """Return configured optimizer."""
        optimizer = tf.train.MomentumOptimizer(
            _learning_rate, self.config["momentum"])
        return optimizer

    @staticmethod
    def loss(logits, labels):
        """Compute loss function for batch."""
        # TODO: loss should be multinomial logistic regression objective
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits, labels=labels))

    @staticmethod
    def preprocess_batch(dataset, metadata):
        """Preprocess a single batch of data."""
        dataset = dataset.map(lambda batch: {
            "inputs": image_util.flip_image_batch(batch["inputs"]),
            "targets": batch["targets"],
        })
        dataset = dataset.map(lambda batch: {
            "inputs": image_util.color_shift_batch(
                batch["inputs"], metadata["eig_vals"], metadata["eig_vecs"]),
            "targets": batch["targets"],
        })
        return dataset

    @staticmethod
    def prep(train_x, train_y, test_x, test_y, metadata):
        """Preprocessing applied to image features.

        - Normalize the RGB values to 0-1 float scale
        - From each pixel, subtract the mean RGB value of the training set.
        - Augment dataset by flipping images horizontally
        - Compute dataset statistics needed for PCA color augmentation
        """
        mean_rgb = tf.reduce_mean(train_x, axis=(0, 1, 2))
        train_x = train_x - mean_rgb
        test_x = test_x - mean_rgb
        eig_vals, eig_vecs = image_util.image_pca(train_x)
        metadata.update({
            "eig_vals": eig_vals,
            "eig_vecs": eig_vecs,
        })
        return train_x, train_y, test_x, test_y, metadata


def _learning_rate():
    # TODO: implement the correct learning rate logic as per the paper:
    # The learning rate was initially set to .01 and then decreased by
    # a factor of 10 when the validation accuracy stopped improving
    return CONFIG["initial_learning_rate"]
