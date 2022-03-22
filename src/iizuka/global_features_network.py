import tensorflow as tf

class GlobalFeatNet(tf.keras.layers.Layer):

    def __init__(self, **kwargs): 
        super(GlobalFeatNet, self).__init__(**kwargs)
        self.net_layers = []
        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(2,2), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), strides=(1,1), padding='same'))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

        # NOTE: Paper does not specify how to transition from Conv2D- to Dense-Layer (Flatten causes number of variables to explode)
        self.net_layers.append(tf.keras.layers.GlobalMaxPooling2D())
        self.net_layers.append(tf.keras.layers.Dense(units=1024))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Dense(units=512))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())
        self.net_layers.append(tf.keras.layers.Dense(units=256))
        self.net_layers.append(tf.keras.layers.Activation(tf.nn.relu))
        self.net_layers.append(tf.keras.layers.BatchNormalization())

    @tf.function
    def call(self, x, training=False):
        for layer in self.net_layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        config = super(GlobalFeatNet, self).get_config()
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)