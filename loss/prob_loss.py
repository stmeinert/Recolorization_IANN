import tensorflow as tf
import numpy as np

# makes sure we won't take log(0)
# otherwise we sometimes get loss NaN
EPSILON = 0.000001

class ProbLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ProbLoss, self).__init__()
        self.lossFlag_weights = True
        if self.lossFlag_weights:
           self.weights_array = np.load('weights_tensor.npy')
           self.weights_tensor = tf.convert_to_tensor(self.weights_array,dtype=tf.float32)
           print(self.weights_tensor)


    @tf.function(experimental_compile=True)
    def call(self, y_true, y_pred):
        # inputs have shape (batch_size, H, W, Q)
        # loss like described in Let there be color, 2.1
        y_pred = y_pred + EPSILON
        sum_tensor = tf.constant(0,dtype=tf.float32)
        tensor_shape = tf.shape(y_true)
        if self.lossFlag_weights:
            for b in tf.range(tensor_shape[0]):
                for h in tf.range(tensor_shape[1]):
                    for w in tf.range(tensor_shape[2]):
                        sum_tensor = tf.add(tf.multiply(self.weights_tensor[tf.argmax(y_true[b,h,w,:],axis=-1)],tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred)),axis=-1)[b,h,w]),sum_tensor)
            return sum_tensor

        else:
            return tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred))), tf.constant([-1], dtype=tf.float32))