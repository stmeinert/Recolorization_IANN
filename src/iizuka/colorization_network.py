import tensorflow as tf
from src.iizuka.fusion_layer import FusionLayer

class ColorizationNet(tf.keras.layers.Layer):

    def __init__(self, batch_size, **kwargs): 
        super(ColorizationNet, self).__init__(**kwargs)
        self.net_layers = []
        self.net_layers.append(FusionLayer(batch_size))
        self.net_layers.append(tf.keras.layers.Conv2D(128, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

        self.net_layers.append(tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last', interpolation='nearest'))
        self.net_layers.append(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

        self.net_layers.append(tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last', interpolation='nearest'))
        self.net_layers.append(tf.keras.layers.Conv2D(32, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(2, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.sigmoid))

    @tf.function
    def call(self, x, training=False):
        for layer in self.net_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(ColorizationNet, self).get_config()
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)