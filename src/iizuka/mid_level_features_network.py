import tensorflow as tf

class MidLevelFeatNet(tf.keras.layers.Layer):
    """
    This class represents the Mid-Level Features Network which extracts local mid-level features
    from the low-level features and passes them to the Fusion Layer.
    """

    def __init__(self, **kwargs): 
        super(MidLevelFeatNet, self).__init__(**kwargs)
        self.net_layers = []
        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, x, training=False):
        for layer in self.net_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(MidLevelFeatNet, self).get_config()
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)