# Taken from https://arxiv.org/pdf/1805.07648.pdf

class Attention(tf.keras.layers.Layer):

    def __init__(self):

        self._input_shape = None

        super(Attention, self).__init__()

    def build(self, input_shape):

        batch_size, sequence_length, num_features = input_shape
        self._input_shape = input_shape

        self.context_weights = self.add_weight(
            name="context_weights",
            shape=(num_features, num_features),
            initializer="glorot_uniform",
            trainable=True
        )

        self.context_bias = self.add_weight(
            name="context_bias",
            shape=(1, num_features),
            initializer="glorot_uniform",
            trainable=True
        )

        self.attention_weights = self.add_weight(
            name="attention_weights",
            shape=(num_features, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        self.attention_bias = self.add_weight(
            name="attention_bias",
            shape=(1, 1),
            initializer="glorot_uniform",
            trainable=True
        )

        super(Attention, self).build(input_shape)

    def call(self, input):

        batch_size, sequence_length, num_features = self._input_shape 
        reshaped_input = tf.keras.backend.reshape(
            input,
            (-1, num_features)
        )

        weights = tf.keras.backend.dot(
            reshaped_input,
            self.context_weights
        ) + self.context_bias

        weights = tf.keras.activations.tanh(weights)

        weights = tf.keras.backend.dot(
            weights,
            self.attention_weights
        ) + self.attention_bias

        weights = tf.keras.activations.softmax(weights)
        weights = tf.keras.backend.reshape(
            weights,
            (-1, sequence_length, 1)
        )

        transposed_input = tf.transpose(
            input,
            (0, 2, 1)
        )

        output = tf.keras.backend.batch_dot(
            transposed_input,
            weights
        )

        output = tf.squeeze(output, [2])
        return output

