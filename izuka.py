import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm

class LowLevelFeatNet(tf.keras.layers.Layer):

    def __init__(self): 
        super(LowLevelFeatNet, self).__init__()
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
        self.net_layers.append(tf.keras.layers.Conv2D(filters=156, kernel_size=(3,3), strides=(2,2), padding='same'))
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


class MidLevelFeatNet(tf.keras.layers.Layer):

    def __init__(self): 
        super(MidLevelFeatNet, self).__init__()
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


class GlobalFeatNet(tf.keras.layers.Layer):

    def __init__(self): 
        super(GlobalFeatNet, self).__init__()
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



class FusionLayer(tf.keras.layers.Layer):

    def __init__(self): 
        super(FusionLayer, self).__init__()

    @tf.function
    def call(self, x, training=False):
        """ Implementation of a similar approach can be found in https://github.com/baldassarreFe/deep-koalarization/blob/master/src/koalarization/fusion_layer.py """
        imgs, embs = x
        reshaped_shape = imgs.shape[:3].concatenate(embs.shape[1])
        embs = tf.repeat(embs, imgs.shape[1] * imgs.shape[2])
        embs = tf.reshape(embs, reshaped_shape)
        return tf.concat([imgs, embs], axis=3)


class ColorizationNet(tf.keras.layers.Layer):

    def __init__(self): 
        super(ColorizationNet, self).__init__()
        self.net_layers = []
        self.net_layers.append(FusionLayer())
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


class IizukaRecolorizationModel(tf.keras.Model):

    def __init__(self): 
        super(IizukaRecolorizationModel, self).__init__()

        self.rescale = tf.keras.layers.Resizing(224, 224, interpolation='nearest', crop_to_aspect_ratio=True)
        self.low = LowLevelFeatNet()
        self.mid = MidLevelFeatNet()
        self.glob = GlobalFeatNet()
        self.colorize = ColorizationNet()
        self.upS = tf.keras.layers.UpSampling2D(size=(2,2), data_format='channels_last', interpolation='nearest')

        self.optimizer = tf.keras.optimizers.Adadelta
        self.loss_function = tf.keras.losses.MeanSquaredError
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        # tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        # tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                        ]

    @tf.function
    def call(self, x):
        re = self.rescale(x)
        l1 = self.low(re)
        g = self.glob(l1)

        l2 = self.low(x)
        m = self.mid(l2)

        c = self.colorize((m,g))
        x = self.upS(c)

        return (x*255)-128

    
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
            
    @tf.function
    def train_step(self, data):
        
        x, targets = data
        
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            
            loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
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
        
        predictions = self(x, training=False)
        
        loss = self.loss_function(targets, predictions) + tf.reduce_sum(self.losses)
        
        self.metrics[0].update_state(loss)
        
        for metric in self.metrics[1:]:
            metric.update_state(targets, predictions)

        return {m.name: m.result() for m in self.metrics}