import tensorflow as tf
import numpy as np

# makes sure we won't take log(0)
# otherwise we sometimes get loss NaN
EPSILON = 0.000001
USE_WEIGHT = True
#WEIGHTS_PATH = 'weights_tensor.npy'
WEIGHTS_PATH = 'drive/MyDrive/weights_tensor_30000_1000.npy'

class ProbLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ProbLoss, self).__init__()

        if USE_WEIGHT:
           weights_array = np.load(WEIGHTS_PATH)
           self.weights_tensor = tf.convert_to_tensor(weights_array,dtype=tf.float32)
           #print(self.weights_tensor)


    @tf.function#(experimental_compile=True)
    def call(self, y_true, y_pred):
        # inputs have shape (batch_size, H, W, Q)
        # loss like described in Let there be color, 2.1

        # make sure y_pred is not 0, log(0) is not defined
        y_pred = y_pred + EPSILON

        sum_tensor = tf.constant(0.,dtype=tf.float32)
        log_mult = tf.math.multiply(y_true, tf.math.log(y_pred))
        q_sum = tf.math.reduce_sum(log_mult, axis=-1)
        tensor_shape = tf.shape(y_true)
        for b in tf.range(tensor_shape[0]):
            for h in tf.range(tensor_shape[1]):
                for w in tf.range(tensor_shape[2]):
                    weight = self.weights_tensor[tf.argmax(y_true[b,h,w,:])]
                    sum_tensor -= (q_sum[b,h,w] * weight)
                        
        return sum_tensor