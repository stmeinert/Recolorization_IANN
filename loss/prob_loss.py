import tensorflow as tf


# makes sure we won't take log(0)
# otherwise we sometimes get loss NaN
EPSILON = 0.000001

class ProbLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ProbLoss, self).__init__()

    @tf.function
    def call(self, y_true, y_pred):
        #print(y_true)
        #print(y_pred)
        # inputs have shape (batch_size, H, W, Q)
        # loss like described in Let there be color, 2.1
        y_pred = y_pred + EPSILON
        return tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred))), tf.constant([-1], dtype=tf.float32))