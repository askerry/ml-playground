"""Specifies architecture and training configuration for AIAYN model."""

import tensorflow as tf
import tensorflow.contrib.layers

import seq_util
import interfaces

layers = tf.keras.layers

CONFIG = {
    "batch_size": 256,
    "num_epochs": 74,
    "dropout_ratio": 0.1,
    "learning_rate": 0.0001,
}

EOS_ID = 1

def encode(input_str, encoders):
  """Input str to features dict, ready for inference"""
  inputs = encoders["inputs"].encode(input_str) + [EOS_ID]  # add EOS id
  batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
  return {"inputs": batch_inputs}

def decode(integers, encoders):
  """List of ints to str"""
  integers = list(np.squeeze(integers))
  if EOS_ID in integers:
    integers = integers[:integers.index(EOS_ID)]
  return encoders["inputs"].decode(np.squeeze(integers))

def get_max_length(mixed_length_tensors, max_len=999):
    return min(len(max(mixed_length_tensors, key=lambda t: t.shape[1].value)), max_len)


class ModelSpec(interfaces.ModelBase):

    def construct_model(self):
        """Returns AIAYN model with the specified parameters."""

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
        return tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                labels, logits))


    @staticmethod
    def prep(dataset, metadata, inference=False):
        """Preprocess a single batch of data."""
        def pad_element(element, target_size):
            element["inputs"] = seq_util.pad_to_length(element["inputs"], target_size)
            if not inference:
                element["targets"] = seq_util.pad_to_length(element["targets"], target_size)
            return element
        dataset = dataset.map(lambda d: pad_element(d, metadata["max_length"]))
        return dataset, metadata

class Translator(object):
    def __init__(self, model, metadata):
        self.model = model
        self.metadata = metadata

    def translate(input_string):
        inputs = encode(input_string, self.metadata["encoders"])
        inputs, _ = ModelSpec.prep(inputs, self.metadata)
        outputs = model(inputs)
        predictions = tf.argmax(outputs, axis=1, output_type=tf.int32)
        return decode(predictions, self.metadata["decoders"])


