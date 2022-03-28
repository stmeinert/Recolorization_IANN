import tensorflow as tf

class FusionLayer(tf.keras.layers.Layer):
    """
    This class represents the Fusion Layer which is the first part of the Colorization Network
    and append the global features at each spatial location of the local mid-level features.
    """

    def __init__(self, batch_size, **kwargs): 
        super(FusionLayer, self).__init__(**kwargs)
        self.batch_size = tf.constant(batch_size)

    @tf.function
    def call(self, x, training=False):
        """ Implementation of a similar approach can be found in https://github.com/baldassarreFe/deep-koalarization/blob/master/src/koalarization/fusion_layer.py """
        imgs, embs = x
        reshaped_shape = tf.stack([tf.constant(self.batch_size), tf.constant(imgs.shape[1]), tf.constant(imgs.shape[2]), tf.constant(embs.shape[1])])
        # reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
        embs = tf.repeat(embs, imgs.shape[1] * imgs.shape[2])
        embs = tf.reshape(embs, reshaped_shape)
        return tf.concat([imgs, embs], axis=3)

    def get_config(self):
        config = super(FusionLayer, self).get_config()
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)