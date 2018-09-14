import tensorflow as tf
import tensorflow.contrib.eager as tfe

layers = tf.keras.layers

tf.enable_eager_execution()

# Training and architectural parameters from the best performing
# model in the paper.

VGG_CONFIG = {
    "batch_size": 256,
    "num_epochs": 74,

    # Optimized using mini-batch gradient descent with momentum of .9
    "momentum": .9,

    # The learning rate was initially set to .01 and then decreased by
    # a factor of 10 when the validation accuracy stopped improving
    "initial_learning_rate": .01,
    "learning_rate_factor": 10,

    # Regularised by weight decay (L2) and dropout
    "l2_multiplier": .0005,
    "dropout_ratio": .5,

    # Convolutional filters use a 3 × 3 receptive field
    "filter_size": 3,
    "convolutional_stride": 1,

    # Max-pooling is performed over a 2 × 2 pixel window
    "max_pooling_window": 2,

    # Fully-connected layers have 4096 channels (except final
    # soft-max layer with has the same number of channels as classes)
    "num_fc_channels": 4096,
}


def train(dataset, config=None):
    if config is None:
        config = VGG_CONFIG
    model = construct_model(config)
    optimizer = tf.train.MomentumOptimizer(
        learning_rate, config["momentum"])

    writer = tf.contrib.summary.create_file_writer("vgg")
    step_counter = tf.train.get_or_create_global_step()
    with writer.as_default(), tf.contrib.summary.always_record_summaries():
        for (batch, (images, labels)) in enumerate(dataset):
            with tf.GradientTape() as tape:

                # Compute logits and loss
                logits = model(images, training=True)
                loss_value = loss(logits, labels)

                # Write output logs and summary values
                tf.contrib.summary.scalar('loss', loss_value)
                accuracy = tfe.metrics.Accuracy()
                accuracy(
                    labels=labels,
                    predictions=tf.argmax(logits, axis=1, output_type=tf.int32))
                tf.contrib.summary.scalar('accuracy', accuracy.result())
                grads = tape.gradient(loss_value, model.variables)

                if batch % 100 == 0:
                    print('Step #%d\tLoss: %.4f, Accuracy: %.4f' % (
                        batch, loss_value, accuracy.result()))

                # Apply gradients to update weights
                optimizer.apply_gradients(
                    zip(grads, model.variables), global_step=step_counter)
    return model


def learning_rate():
    # TODO: implement the correct learning rate logic as per the paper:
    # The learning rate was initially set to .01 and then decreased by
    # a factor of 10 when the validation accuracy stopped improving
    return VGG_CONFIG["initial_learning_rate"]


def loss(logits, labels):
    """Compute loss function for batch."""
    return tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels))


def construct_model(config):
    """Returns VGG model with the specified parameters."""
    data_format = "channels_last"
    pooling_window = config["max_pooling_window"]

    max_pool_layer = layers.MaxPooling2D(
        (pooling_window, pooling_window),
        (pooling_window, pooling_window),
        padding='same',
        data_format=data_format)

    def conv_layer(size):
        return layers.Conv2D(
            size,
            config["convolutional_stride"],
            padding='same',
            data_format=data_format,
            activation=tf.nn.relu)

    model_layers = [
        conv_layer(64),
        conv_layer(64),
        max_pool_layer,
        conv_layer(128),
        conv_layer(128),
        max_pool_layer,
        conv_layer(256),
        conv_layer(256),
        conv_layer(256),
        conv_layer(256),
        max_pool_layer,
        conv_layer(512),
        conv_layer(512),
        conv_layer(512),
        conv_layer(512),
        max_pool_layer,
        conv_layer(512),
        conv_layer(512),
        conv_layer(512),
        conv_layer(512),
        max_pool_layer,
        layers.Flatten(),
        layers.Dense(config["num_fc_channels"], activation=tf.nn.relu),
        layers.Dropout(config["dropout_ratio"]),
        layers.Dense(config["num_fc_channels"], activation=tf.nn.relu),
        layers.Dropout(config["dropout_ratio"]),
        layers.Dense(10)
    ]
    return tf.keras.Sequential(model_layers)
