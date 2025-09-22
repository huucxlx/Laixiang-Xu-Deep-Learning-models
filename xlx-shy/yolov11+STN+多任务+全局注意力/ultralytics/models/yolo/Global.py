class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.weight = self.add_weight(name="watt", shape=(input_shape[-1], 1), initializer="normal")
        self.bias = self.add_weight(name="batt", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        att = K.expand_dims(K.softmax(K.squeeze(K.tanh(K.dot(x, self.weight) + self.bias), axis=-1)), axis=-1)
        return K.sum(x * att, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        return super(Attention, self).get_config()