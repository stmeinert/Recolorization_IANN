# Model
# zhang_prob

import tensorflow as tf
#create the model, like in https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40
# "colorful image colorization"

import math
from loss.prob_loss import ProbLoss


#################################################################
# Functions used for mapping between distribution and real values
#################################################################

LEARNING_RATE = 0.001

Q_SIZE = 22*22

# use hard or soft h^-1 encoding
H_1_HARD = False
SIGMA = 5 # -> for soft encoded H^-1, not used right now
N_NEIGHBOURS = 8

# tensor that contains all <N_NEIGHBOURS> nearest neighbours of each bin
lookup_tensor = None

# TODO: calculate values of gaussian distribution regarding bins distances, 
#       save tensor of shape Q_SIZE for each bin -> gaussian not uniform distribution over nearest n bins
def fill_lookup_tensor():
    """
    Fills the lookup tensor by calculating euklidean distances inbetween all bins
    """
    global lookup_tensor

    def euclidean_dist(x, y):
        return math.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)

    def take_second(x):
        return x[1]


    ta = tf.TensorArray(dtype=tf.uint32, size=Q_SIZE)
    bin_centers = [bin_to_ab(tf.constant([bin], dtype=tf.uint32)).numpy() for bin in range(Q_SIZE)]

    for i in range(Q_SIZE):
        dists = [(j, euclidean_dist(bin_centers[i], bin_centers[j])) for j in range(Q_SIZE)]
        dists.sort(key=take_second)
        ret = [bin for bin, dist in dists[:N_NEIGHBOURS]]


        ta = ta.write(i, tf.constant(ret, dtype=tf.uint32))

    lookup_tensor = ta.stack()



@tf.function(experimental_compile=True)
def bin_to_ab(bin_nr):
    """
    Returns center of bin 'bin_nr'
    """
    bin_nr = tf.clip_by_value(bin_nr, clip_value_min=0, clip_value_max=(Q_SIZE-1))

    #   a -->
    # b  0 |  1 |  2  | ...
    # | 22 | 23 | ...
    # |
    # V

    b = tf.cast(tf.math.floor(bin_nr / 22), dtype=tf.float32)
    a = tf.cast(bin_nr % 22, dtype=tf.float32)

    # range [0;22] -> [-110;110], take center of bin -> -105, not -110
    a = a*10 - 105
    b = b*10 - 105

    ab = tf.stack([a,b], axis=0)
    return ab


@tf.function(experimental_compile=True)
def ab_to_bin(ab):
    """
    Input is Tensor of shape (2,)
    """
    ab = tf.clip_by_value(ab, clip_value_min=-110., clip_value_max=110.)
    # range [-110;110] -> [0;21]
    ab = (ab+110)/10
    ab = tf.cast(tf.math.floor(ab), dtype=tf.int32)

    # a_value * 1 + b_value * 22
    return tf.math.reduce_sum(tf.math.multiply(ab, tf.constant([1,22], dtype=tf.int32)))


@tf.function(experimental_compile=True)
def ab_to_nearest(ab):
    global lookup_tensor
    
    bin = ab_to_bin(ab)
    return lookup_tensor[bin,:]


@tf.function(experimental_compile=True)
def H_1(target):
    if H_1_HARD:
        return H_1_hard(target)
    else:
        return H_1_soft(target)



@tf.function(experimental_compile=True)
def H(x):
    """
    Function that maps probability distribution Z over the Q bins to one ab value
    
    Assume that Z is a rank 1 Tensor over all bins
    Takes the mode -> TODO: annealed mean
    """
    shape = tf.shape(x)
    tb = tf.TensorArray(tf.float32, size=shape[0], name="tb")#, element_shape=shape)
    ib = 0

    for b in tf.range(shape[0]):
        th = tf.TensorArray(tf.float32, size=shape[1], name="th")
        ih = 0
        for h in tf.range(shape[1]):
            tw = tf.TensorArray(tf.float32, size=shape[2], name="tw")
            iw = 0
            for w in tf.range(shape[2]):
                Z = x[b,h,w,:]
                ab = bin_to_ab(tf.math.argmax(Z))
                tw = tw.write(iw, ab)
                iw += 1
                
            tw_resh = tf.reshape(tw.stack(), shape=[shape[2], 2])
            th = th.write(ih, tw_resh)
            ih += 1

        th_resh = tf.reshape(th.stack(), shape=[shape[1], shape[2], 2])
        tb = tb.write(ib, th_resh)
        ib += 1

    x = tf.reshape(tb.stack(), shape=[shape[0], shape[1], shape[2], 2])
    return x


@tf.function(experimental_compile=True)
def H_1_hard(target):
    """
    Returns Tensor of one-hot vectors (probability distributions) with matching bin = 1, 0 otherwise
    """
    shape = tf.shape(target)

    #ret = tf.zeros(shape=[shape[0], shape[1], shape[2], Q_SIZE])
    tb = tf.TensorArray(tf.float32, size=shape[0])
    ib = 0
    for b in tf.range(shape[0]):
        th = tf.TensorArray(tf.float32, size=shape[1])
        ih = 0

        for h in tf.range(shape[1]):
            tw = tf.TensorArray(tf.float32, size=shape[2])
            iw = 0

            for w in tf.range(shape[2]):
                bins = ab_to_bin(target[b,h,w,:])
                prob_tensor = tf.one_hot(indices=bins, depth=Q_SIZE, on_value=1, off_value=0, dtype=tf.float32)
                tw = tw.write(iw, prob_tensor)
                iw += 1
            th = th.write(ih, tf.reshape(tw.stack(), shape=[shape[2], Q_SIZE]))
            ih += 1
        
        tb = tb.write(ib, tf.reshape(th.stack(), shape=[shape[1], shape[2], Q_SIZE]))
        ib += 1
        
    
    target = tf.reshape(tb.stack(), shape=[shape[0], shape[1], shape[2], Q_SIZE])
    return target


@tf.function(experimental_compile=True)
def H_1_soft(target):
    shape = tf.shape(target)

    tb = tf.TensorArray(tf.float32, size=shape[0])
    ib = 0
    for b in tf.range(shape[0]):
        th = tf.TensorArray(tf.float32, size=shape[1])
        ih = 0

        for h in tf.range(shape[1]):
            tw = tf.TensorArray(tf.float32, size=shape[2])
            iw = 0

            for w in tf.range(shape[2]):
                bins = ab_to_nearest(target[b,h,w,:])
                on_value = 1/tf.shape(bins)[0]
                prob_dist = tf.one_hot(indices=bins, depth=Q_SIZE, on_value=on_value, off_value=0, dtype=tf.float32)

                tw = tw.write(iw, prob_dist)
                iw += 1
            th = th.write(ih, tf.reshape(tw.stack(), shape=[shape[2], Q_SIZE]))
            ih += 1
        
        tb = tb.write(ib, tf.reshape(th.stack(), shape=[shape[1], shape[2], Q_SIZE]))
        ib += 1
        
    
    target = tf.reshape(tb.stack(), shape=[shape[0], shape[1], shape[2], Q_SIZE])
    return target



################################################################
# Zhangs model using the probability distribution
################################################################

divide_factor = 2

class CIC_Prob(tf.keras.Model):
    def __init__(self):
        super(CIC_Prob, self).__init__()
        # TODO change optimizer, question, what optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
        self.loss_function = ProbLoss()
        #self.loss_function = tf.keras.losses.MeanSquaredError()
    

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
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

            # conv_5
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2, 
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),

             # conv_6 same as conv_5
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2,  
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2, 
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
                                   kernel_size=3,
                                   dilation_rate = 2,
                                   strides=1, 
                                   padding="same",
                                   activation=None,
                                   use_bias= True),
            tf.keras.layers.Activation(tf.nn.relu),
            tf.keras.layers.BatchNormalization(),


            #conv_7
            tf.keras.layers.Conv2D(filters=512/divide_factor, 
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
                                   use_bias= True),

            # we need this for the output to be in [0;1] --> loss function takes logarithm of this value!

            #inserting the upsampling
            tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')
        ]
        self.last_activation = tf.keras.layers.Activation(tf.nn.softmax)

        if not H_1_HARD:
            fill_lookup_tensor()

        #self.upsample_layer = tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')
    
    # can stay the same
    @tf.function
    def call(self, x, training=False):
        for layer in self.all_layers:
            try:
                x = layer(x,training)
            except:
                x = layer(x)

        x = self.last_activation(x)
    
        if training:
            return x

        x = H(x)
        #x = (x*255) - 128
        return x
        



    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
    
    
          
    @tf.function
    def train_step(self, data):
        x, target = data
        loss = -1

        with tf.GradientTape() as tape:
            prediction = self(x, training=True)

            target = H_1_hard(target[:,:,:,1:])
            loss = self.loss_function(target, prediction)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        return loss
        
    
    @tf.function
    def test(self, data):
        loss = 0.
        i = 0.
        for (x, target) in data:
            predictions = self(x, training=True)
            target = H_1_hard(target[:,:,:,1:])
            loss += self.loss_function(target, predictions)
            i += 1.

        return loss / i