"""Specifies architecture and configuration for Neural Machine Translation."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers

import seq_util
import interfaces

layers = tf.keras.layers

CONFIG = {
    "batch_size": 8,
    "num_epochs": 2,
    "dropout_ratio": 0.1,
    "learning_rate": 0.0001,
    "EOS": 1,
    "SOS": 0,
}

class ModelSpec(interfaces.ModelBase):

    def construct_model(self):
        """Returns NMT model with the specified parameters."""

        # TODO: model architecture
        model_layers = []
        return tf.keras.Sequential(model_layers)


    def optimizer(self):
        """Return configured optimizer."""
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.config["learning_rate"])
        return optimizer

    @staticmethod
    def loss(logits, labels):
        """Compute loss function for batch."""
        # Mask out padding values in the labeled set
        mask = 1 - np.equal(labels, 0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits) * mask
        return tf.reduce_mean(loss)

    @staticmethod
    def prep(dataset, metadata, inference=False):
        """Preprocess a single batch of data."""
        def pad_element(element, target_size):
            element["inputs"] = seq_util.pad_to_length(element["inputs"], target_size)
            # element["inputs"] = tf.keras.preprocessing.sequence.pad_sequences(
            #     element["inputs"], maxlen=target_size, padding='post')
            if not inference:
                element["targets"] = seq_util.pad_to_length(element["targets"], target_size)
                # element["targets"] = tf.keras.preprocessing.sequence.pad_sequences(
                #     element["targets"], maxlen=target_size, padding='post')
            return element
        dataset = dataset.map(lambda d: pad_element(d, metadata["max_length"]))
        return dataset, metadata
