import tensorflow as tf
from src.iizuka.low_level_features_network import LowLevelFeatNet
from src.iizuka.mid_level_features_network import MidLevelFeatNet
from src.iizuka.global_features_network import GlobalFeatNet
from src.iizuka.colorization_network import ColorizationNet

class IizukaRecolorizationModel(tf.keras.Model):

    def __init__(self, batch_size, **kwargs): 
        super(IizukaRecolorizationModel, self).__init__(**kwargs)

        self.rescale = tf.keras.layers.Resizing(224, 224, interpolation='nearest', crop_to_aspect_ratio=True)
        self.low = LowLevelFeatNet()
        self.mid = MidLevelFeatNet()
        self.glob = GlobalFeatNet()
        self.colorize = ColorizationNet(batch_size)
        self.upS = tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last', interpolation='nearest')

        self.optimizer = tf.keras.optimizers.Adadelta(learning_rate=1.0)
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        ]

    @tf.function
    def call(self, x, training=False):
        re = self.rescale(x, training=training)
        l1 = self.low(re, training=training)
        g = self.glob(l1, training=training)

        l2 = self.low(x, training=training)
        m = self.mid(l2, training=training)

        c = self.colorize((m,g), training=training)
        out = self.upS(c, training=training)

        # bring the a-b-values from range [0,1] to [-128, 127]
        out = out * 255.0
        out = out - 128.0
        return out

    @tf.function
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
            
    @tf.function
    def train_step(self, data):
        
        x, targets = data

        # throw away L-dimension in target
        targets = targets[:,:,:,-2:]
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions)# + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(targets,predictions)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):

        x, targets = data

        # throw away L-dimension in target
        targets = targets[:,:,:,-2:]
        
        predictions = self(x, training=False)
        
        loss = self.loss_function(targets, predictions)# + tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}

    def get_config(self):
        config = super(IizukaRecolorizationModel, self).get_config()
        return config
        
    @classmethod
    def from_config(cls, config):
        return cls(**config)