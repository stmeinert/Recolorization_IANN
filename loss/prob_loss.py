import tensorflow as tf

class ProbLoss(tf.keras.losses.Loss):
    def __init__(self):
        super(ProbLoss, self).__init__()


    def call(self, y_true, y_pred):
        # inputs have shape (batch_size, H, W, Q)
        # loss like described in Let there be color, 2.1
        return tf.math.multiply(tf.math.reduce_sum(tf.math.multiply(y_true, tf.math.log(y_pred))), tf.constant([-1]))