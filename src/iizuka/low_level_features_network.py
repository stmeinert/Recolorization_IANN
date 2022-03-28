import tensorflow as tf

class LowLevelFeatNet(tf.keras.layers.Layer):
    """
    This class represents the Low-Level Features Network which extracts local low-level features
    directly from the input image and passes them to the Mid-Level Features Network and 
    the Globla Features Network.
    """

    def __init__(self, **kwargs): 
        super(LowLevelFeatNet, self).__init__(**kwargs)
        self.net_layers = []
        self.net_layers.append(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(2,2), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

        self.net_layers.append(tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        
        self.net_layers.append(tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, x, training=False):
        for layer in self.net_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(LowLevelFeatNet, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)