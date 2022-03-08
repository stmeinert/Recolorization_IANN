import tensorflow as tf
#create the model, like in https://link.springer.com/chapter/10.1007/978-3-319-46487-9_40
# "colorful image colorization"

class CIC(tf.keras.Model):
    def __init__(self):
        super(CIC, self).__init__()
        # TODO change optimizer, question, what optimizer
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.metrics_list = [
                        tf.keras.metrics.Mean(name="loss"),
                        tf.keras.metrics.CategoricalAccuracy(name="acc"),
                        tf.keras.metrics.TopKCategoricalAccuracy(3,name="top-3-acc") 
                       ]
        #TODO change the loss function
        self.loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)   
    

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

            tf.keras.layers.Conv2D(filters=313, 
                                   kernel_size=1,
                                   strides=1, 
                                   padding="valid",
                                   activation=None,
                                   use_bias= True),
            # the following order model_out(softmax(x))
            # model_out == Conv2D(filters=2, ) 
            tf.keras.layers.Activation(tf.nn.softmax),
            tf.keras.layers.Conv2D(filters=2, 
                                   kernel_size=1,
                                   strides=1, 
                                   padding="valid",
                                   dilation_rate = 1,
                                   activation=None,
                                   use_bias= False),
            # do we need upsampling ??? like here
            # self.upsample4 = nn.Upsample(scale_factor=4, mode='bilinear')
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
       
        x = (x*256) - 128
        return x
    
    def reset_metrics(self):
        
        for metric in self.metrics:
            metric.reset_states()
    # TODO check again        
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