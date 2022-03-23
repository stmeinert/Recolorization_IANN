import tensorflow as tf
#create the model, like in https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40
# "colorful image colorization"

import math
from src.zhang.l2_loss import L2_Loss as loss

divide_factor = 2

class CIC(tf.keras.Model):
    def __init__(self):
        super(CIC, self).__init__()
        # TODO change optimizer, question, what optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        self.loss_function = loss()

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

            tf.keras.layers.Conv2D(filters=2, 
                                   kernel_size=1,
                                   strides=1, 
                                   padding="valid",
                                   activation=None,
                                   use_bias= True),

            # we need this for the output to be in [0;1] --> loss function takes logarithm of this value!

            #inserting the upsampling
            tf.keras.layers.UpSampling2D(size=(4, 4), data_format='channels_last', interpolation='bilinear')
        ]
    

    # can stay the same
    def call(self, x, training=False):

        for layer in self.all_layers:
            try:
                x = layer(x,training)
            except:
                x = layer(x)
       
        x = (x*255) - 128
        return x


    @tf.function
    def train_step(self, data):
        x, target = data
        loss = -1

        with tf.GradientTape() as tape:
            prediction = self(x)

            loss = self.loss_function(target[:,:,:,1:], prediction)
        
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        
        # update loss metric
        return loss
        
    
    @tf.function
    def test(self, data):
        loss = 0.
        i = 0.
        for (x, target) in data:
            predictions = self(x)
            loss += self.loss_function(target[:,:,:,1:], predictions)
            i += 1.

        return loss / i