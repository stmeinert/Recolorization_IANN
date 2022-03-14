import tensorflow as tf
#create the model, like in https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40
# "colorful image colorization"

import math
import numpy as np
from loss.prob_loss import ProbLoss
from data_pipeline.data_pipeline import BATCH_SIZE, SIZE


#################################################################
# Functions used for mapping between distribution and real values
#################################################################

Q_SIZE = 22*22
SIGMA = 5 # -> for soft encoded H^-1

def bin_to_ab(bin_nr):
    """
    Returns center of bin 'bin_nr'
    """
    if bin_nr == None:
        bin_nr = tf.constant([0], dtype=tf.float32)

    if bin_nr < 0 or bin_nr >= Q_SIZE:
        return tf.constant([0,0], dtype=tf.float32)

    #   a -->
    # b  0 |  1 |  2  | ...
    # | 22 | 23 | ...
    # |
    # V

    #print(bin_nr)
    b = tf.cast(tf.math.floor(bin_nr / 22), dtype=tf.float32)
    a = b*22 + tf.cast(bin_nr % 22, dtype=tf.float32)

    # range [0;22] -> [-110;110], take center of bin -> -105, not -110
    a = a*10 - 105
    b = b*10 - 105

    ab = tf.stack([a,b], axis=0)
    return ab
    return tf.constant([[[[a,b]]]], dtype=tf.float32)


def ab_to_bin(ab):
    # range [-110;110] -> [0;21]

    #a = math.floor((a+110) / 10)
    #b = math.floor((b+110) / 10)
    #return 22*b+a

    print(ab)

    ab = tf.reshap(ab, [2])
    ab = (ab+110)/10
    ab = tf.math.floor(ab)
    return tf.math.reduce_sum(tf.math.multiply(tf.constant([1,22])))

def ab_to_nearest(a, b):
    # TODO: do it better/faster!!!
    points = []

    for bin in range(Q_SIZE):
        point = bin_to_ab(a, b).numpy()
        points.append(bin, math.sqrt((a-point[0])**2 + (b-point[1])**2))

    def take_second(x):
        return x[1]

    points.sort(key=take_second)
    # return bins and dists
    return points[:5]


def H(Z):
    """
    Function that maps probability distribution Z over the Q bins to one ab value
    
    Assume that Z is a rank 1 Tensor over all bins

    Takes the mode -> TODO: annealed mean
    """
    #print(Z)
    #print(tf.math.argmax(Z))
    ab = bin_to_ab(tf.math.argmax(Z))
    #print(ab)
    return ab


def H_1_hard(target):
    """
    Returns Tensor of one-hot vectors (probability distributions) with matching bin = 1, 0 otherwise
    """
    ret = tf.zeros(shape=[BATCH_SIZE, SIZE[0], SIZE[1], Q_SIZE])
    for b in range(BATCH_SIZE):
        for h in range(SIZE[0]):
            for w in range(SIZE[1]):
                bins = ab_to_bin(target[:,h,w,:])
                prob_tensor = tf.one_hot(indices=bins, depth=Q_SIZE, on_value=1, off_value=0)
                ret[b,h,w,:] = prob_tensor

    return target
    #return tf.one_hot(indices=[ab_to_bin(a, b)], depth=Q_SIZE, on_value=1, off_value=0)


def H_1_soft(a, b):
    bins = ab_to_nearest(a, b)

    prob_dist = [0 for bin in range(Q_SIZE)]
    sum = 0
    
    for bin, dist in bins:
        # gaussian kernel
        prob_dist[bin] = 1/math.sqrt(2 * math.pi * SIGMA**2) * math.exp(-dist**2 / (2*SIGMA**2))
        sum += prob_dist[bin]

    # normalize distribution to sum up to 1
    for bin, dist in bins:
        prob_dist[bin] *= 1/sum

    return tf.constant(prob_dist, dtype=tf.float32)



################################################################
# Zhangs model using the probability distribution
################################################################


class CIC_Prob(tf.keras.Model):
    def __init__(self):
        super(CIC_Prob, self).__init__()
        # TODO change optimizer, question, what optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                       ]
        #TODO change the loss function
        #self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)   
        self.loss_function = ProbLoss()
    

        self.all_layers = [
            # inserting my own layers

            #conv1_1
            tf.keras.layers.Conv2D(filters=64, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            #conv1_2
            tf.keras.layers.Conv2D(filters=64, 
                                   kernel_size=3,  
                                   strides=2, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

            #conv2_1
            tf.keras.layers.Conv2D(filters=128, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=128, 
                                   kernel_size=3,  
                                   strides=2, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.BatchNormalization(),

            #conv3
            tf.keras.layers.Conv2D(filters=256, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=256, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=256, 
                                   kernel_size=3,  
                                   strides=2, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

            # conv_4
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

            # conv_5
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2, 
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

             # conv_6 same as conv_5
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2, 
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   dilation_rate = 2,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),


            #conv_7
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3, 
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512, 
                                   kernel_size=3,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

            #conv_8
            tf.keras.layers.Conv2DTranspose(filters=256,kernel_size= 4, 
                                            strides=2, padding = "same", 
                                            use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=256, 
                                   kernel_size=3,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=256, 
                                   kernel_size=3,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            
            #final conv2D-Layer

            tf.keras.layers.Conv2D(filters=Q_SIZE, 
                                   kernel_size=1,
                                   strides=1, 
                                   padding="valid",
                                   activation=None,
                                   use_bias= True)

            # the following order model_out(softmax(x))
            # model_out == Conv2D(filters=2, ) 
            #tf.keras.layers.Activation(tf.nn.softmax),
            #tf.keras.layers.Activation(tf.nn.sigmoid),
            #tf.keras.layers.Conv2D(filters=2, 
            #                       kernel_size=1,
            #                       strides=1, 
            #                       padding="valid",
            #                       dilation_rate = 1,
            #                       activation=None,
            #                       use_bias= False),
            # do we need upsampling ??? like here
            # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
            #inserting the upsampling

        ]

        self.upsample_layer = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')
    
    # can stay the same
    #@tf.function
    def call(self, x, training=False):
        for layer in self.all_layers:
            try:
                x = layer(x,training)
            except:
                x = layer(x)
    
        if training:
            return x

        #shape = tf.shape(x).numpy()
        #shape[-1] = 2 # set last dimension from Q_SIZE to 2
        #ret = np.zeros(shape)
        shape = tf.shape(x)

        #print(x)
        tb = tf.TensorArray(tf.float32, size=shape[0])#, element_shape=shape)

        # now fill ret by using H()
        ib = 0
        for b in range(shape[0]):
            th = tf.TensorArray(tf.float32, size=shape[1])
            ih = 0
            for h in range(shape[1]):
                tw = tf.TensorArray(tf.float32, size=shape[2])
                iw = 0
                for w in range(shape[2]):
                    Z = x[b,h,w,:]
                    #print(tf.shape(Z))
                    ab = H(Z)
                    #x[b,h,w,:] = ab
                    tw.write(iw, ab)
                    iw += 1
                    
                th.write(ih, tf.reshape(tw.stack(), shape=[shape[2], 2]))
                ih += 1
            tb.write(ib, tf.reshape(th.stack(), shape=[shape[1], shape[2], 2]))
            ib += 1

        x = tf.reshape(tb.stack(), shape=[shape[0], shape[1], shape[2], 2])

        #x = tf.constant(ret, dtype=tf.float32)
        x = self.upsample_layer(x)
        x = (x*255) - 128
        return x
        



    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
    
    
    
    # TODO check again        
    #@tf.function
    def train_step(self, data):
        
        x, target = data
        
        with tf.GradientTape() as tape:
            prediction = self(x, training=True)
            
            #shape = tf.shape(targets).numpy()
            #shape[-1] = Q_SIZE # set last dimension from Q_SIZE to 2
            #ret = np.zeros(shape)

            target = H_1_hard(target)


            # now fill ret by using H()
            #for b in range(shape[0]):
            #    for h in range(shape[1]):
            #        for w in range(shape[2]):
            #            ab = x[b,h,w,:]
            #            ab = ab.numpy()
            #            #print(tf.shape(Z))
            #            prob_dist = H_1_hard(ab[0], ab[1] )
            #            ret[b,h,w] = prob_dist.numpy()
            
            #targets = tf.convert_to_tensor(ret)
            print(target)
            loss = self.loss_function(target, prediction)# + tf.reduce_sum(self.losses)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        self.metrics[0].update_state(loss)
        
        # for all metrics except loss, update states (accuracy etc.)
        for metric in self.metrics[1:]:
            metric.update_state(target,prediction)

        # Return a dictionary mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test(self, data):

        for (x, target) in data:
            predictions = self(x, training=False)
            print("Prediciton")
            print(predictions)
            #print(target)
            target = target[:,:,:,1:]
            loss = self.loss_function(target, predictions)# + tf.reduce_sum(self.losses)
            
            #self.metrics[0].update_state(loss)
            
            #for metric in self.metrics[1:]:
                #metric.update_state(target, predictions)

        #return {m.name: m.result() for m in self.metrics}
        return (0,0)