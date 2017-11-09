import keras.backend as K
from keras.layers import Layer


class SaltAndPepper(Layer):
    """
    salt and pepper noise
    """

    def __init__(self, noise, **kwargs):
        super(SaltAndPepper, self).__init__(**kwargs)
        self.supports_masking = True
        self.noise = noise

    def call(self, inputs, training=None):
        def noised():
            # salt and pepper noise
            a = K.random_binomial(shape=K.shape(inputs), p=1. - self.noise, dtype='float32')
            b = K.random_binomial(shape=K.shape(inputs), p=0.5, dtype='float32')
            c = K.cast(K.equal(a, 0), 'float32') * b
            return inputs * a + c

        return K.in_train_phase(noised, inputs, training=training)

    def get_config(self):
        config = {'noise': self.noise}
        base_config = super(SaltAndPepper, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
