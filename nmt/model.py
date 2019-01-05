"""Specifies architecture and configuration for Neural Machine Translation."""
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers

import seq_util
import interfaces

layers = tf.keras.layers


CONFIG = {
    "batch_size": 64,
    "embedding_size": 256,
    "encoder_units": 128,
    "decoder_units": 128,
    "num_encoders": 3,
    "num_decoders": 2,
    "num_epochs": 3,
    "dropout_ratio": 0.1,
    "learning_rate": 0.0001,
    "EOS": 1,
}


class Encoder(tf.keras.Model):
    """Encoder for processing source sequence."""
    def __init__(self, num_units, embedding_dim, vocab_size):
        super(Encoder, self).__init__()
        self.num_units = num_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.encoder_cell = gru(self.num_units)
        self.fc = layers.Dense(vocab_size)

    def __call__(self, x, hidden, training=False):
        """Runs encoding on batch of input sequences.
        Args:
            x: tensor of token ids with shape [
                batch_size, sequence_length, (embedding dim or hidden units)]
            hidden: previous encoder hidden state, shape=[batch_size, hidden units]
            training (bool): whether in training or eval/inference mode
        Returns:
            output (tensor of [batch_size, sequence_length, hidden units]):
                encoder output vector, used as the context vector by the decoder
            hidden (tensor of [batch_size, hidden units]):
                the updated hidden state of the encoder
        """
        # output shape is [batch_size, sequence_length, encoder hidden units]
        output, updated_hidden = self.encoder_cell(x, initial_state=hidden)
        return output, updated_hidden

    def initialize_hidden_state(self, batch_size):
        """Initialize hidden state on first encoding pass."""
        return tf.zeros((batch_size, self.num_units))


class Decoder(tf.keras.Model):
    """Decoder for producing target sequence."""
    def __init__(self, num_units, embedding_dim, vocab_size):
        super(Decoder, self).__init__()
        self.num_units = num_units
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.decoder_cell = gru(self.num_units)
        self.fc = layers.Dense(vocab_size)

        # used for attention
        self.fc1 = layers.Dense(self.num_units)
        self.fc2 = layers.Dense(self.num_units)
        self.V = layers.Dense(1)

    def _get_attention_weights(self, encoder_output, hidden):
        """Calculates Bahdanau attention weights.

        Returns an attention vector with a weight per timestep of input."""
        # Expand dimensionality of hidden in order to add with the
        # encoder output to calculate the score. New shape will be [batch_size, 1, hidden size]
        # to align with [batch_size, sequence_length, hidden size] for encoder output
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        # score will have shape [batch_size, sequence length, 1]
        score = self.V(tf.nn.tanh(
            self.FC1(encoder_output) + self.FC2(hidden_with_time_axis)))
        attention_weights = tf.nn.softmax(score, axis=1)
        # Returns a [batch_size, sequence length, 1] shaped attention vector
        return attention_weights

    def __call__(self, decoder_input, context_vector, hidden):
        """"Predicts the next word/token in the sequence.

        The decoder cell is called for each time step in the sequence.
        Args:
            decoder_input (tensor of [batch_size, 1]):
                the preceding token in the sequence
            context_vector (tensor of [batch_size, sequence_length, hidden units]): 
                the output of the encoder, used as the "context"
                or "thought" vector from which to decode into the target sequence
            hidden (tensor of [batch_size, hidden units]):
                the previous hidden state, used for attention (comes from the
                encoder hidden state initially)
        Returns:
            output_logits (tensor of [batch_size, vocab_size]):
                logits representing the probabilities of each token in vocab
            state (tensor of [batch_size, hidden units]): updated hidden state
        """
        # Apply embedding to yield a [batch_size, 1, num embedding dimensions]
        # shaped input tensor
        embedded_input = self.embedding(decoder_input)
        attention_weights = self._get_attention_weights(context_vector, hidden)
        weighted_context_vector = attention_weights * context_vector
        # Sum over the sequence length dimension to yield a [batch_size, hidden units]
        # context vector
        weighted_context_vector = tf.reduce_sum(weighted_context_vector, axis=1)
        # Combine the previous word and the context vector into a single input
        # with shape [batch_size, 1, embedding dim + hidden units]
        concatenated_input = tf.concat(
            [tf.expand_dims(context_vector, 1), embedded_input], axis=-1)
        output, state = self.decoder_cell(
            concatenated_input, initial_state = hidden)
        # Resulting output is [batch_size, hidden units]
        output = tf.squeeze(output, 1)
        # Apply dense layer to convert to logits [batch_size, vocab_size]
        output_logits = self.fc(output)
        return output_logits, state


class NMT(tf.keras.Model):
    """Neural Machine Translation model made up of encoding and decoding steps."""
    def __init__(self, vocab_size, batch_size, embedding_dim,
                 num_encoder_units=16, num_decoder_units=16,
                 num_encoders=1, num_decoders=1):
        super(NMT, self).__init__()
        self.batch_size = batch_size
        self.num_encoders = num_encoders
        self.num_decoders = num_decoders
        self.encoder = Encoder(num_encoder_units, embedding_dim, vocab_size)
        self.decoder = Decoder(num_decoder_units, embedding_dim, vocab_size)
        self.variables.extend(self.encoder.variables + self.decoder.variables)

    def _initialize_decoder_input(self, batch_size):
        SOS_INDEX = 0
        return tf.expand_dims([SOS_INDEX] * batch_size, 1)

    def _decode(self, encoder_hidden, context_vector, loss_fn, labels, training, batch_size):
        """Run decoding loop for all timesteps."""
        loss = 0
        predicted_logits = []
        num_words = context_vector.shape[1]
        hidden = encoder_hidden
        decoder_input = self._initialize_decoder_input(batch_size)
        for t in range(0, num_words):
            logits, hidden, _ = self.decoder(decoder_input, context_vector, hidden)
            predicted_logits.append(logits)
            predictions = tf.argmax(logits, axis=1)
            if training:
                loss += loss_fn(logits, labels[:, t])
                # if in training, use teacher forcing where we feed
                # the target in as the next input
                decoder_input = tf.expand_dims(labels[:, t], 1)
            else:
                # otherwise, the predicted word is fed back as input
                decoder_input = tf.expand_dims(predictions, 1)
        result = tf.convert_to_tensor(predicted_logits)
        result = tf.transpose(result, perm=[1, 0, 2])
        return result, loss

    def _encode(self, input_ids, batch_size):
        """Run encoding on batch of source token ids."""
        encoder_input = self.encoder.embedding(input_ids)
        encoder_hidden = self.encoder.initialize_hidden_state(batch_size)
        for i in range(self.num_encoders):
            encoder_input, encoder_hidden = self.encoder(
                encoder_input, hidden=encoder_hidden)
        return encoder_input, encoder_hidden

    def __call__(self, x, y=None, loss_fn=None, training=False, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        encoder_output, encoder_hidden = self._encode(x, batch_size)
        decoder_output, loss = self._decode(
            encoder_hidden, encoder_output, loss_fn, y, training, batch_size)
        return decoder_output, loss


class ModelSpec(interfaces.ModelBase):

    def construct_model(self):
        """Returns NMT model with the specified parameters."""

        # TODO: get vocab dynamically
        vocab_size = 21222
        model = NMT(
            vocab_size, self.config["batch_size"],
            self.config["embedding_size"],
            num_encoder_units=self.config["encoder_units"],
            num_decoder_units=self.config["decoder_units"],
            num_encoders=self.config["num_encoders"],
            num_decoders=self.config["num_decoders"])
        return model


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
        return tf.reduce_sum(loss)

    @staticmethod
    def prep(dataset, metadata, inference=False):
        """Preprocess a single batch of data."""
        def pad_element(element, target_size):
            element["inputs"] = seq_util.pad_to_length(
                element["inputs"], target_size)
            # element["inputs"] = tf.keras.preprocessing.sequence.pad_sequences(
            #     element["inputs"], maxlen=target_size, padding='post')
            if not inference:
                element["targets"] = seq_util.pad_to_length(
                    element["targets"], target_size)
                # element["targets"] = tf.keras.preprocessing.sequence.pad_sequences(
                #     element["targets"], maxlen=target_size, padding='post')
            return element
        dataset = dataset.map(lambda d: pad_element(d, metadata["max_length"]))
        return dataset, metadata
