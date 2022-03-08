import tensorflow as tf

class L2_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(L2_Loss, self).__init__()

    @tf.function
    def call(self, y_true, y_false):
        loss = tf.math.reduce_sum(tf.math.sqrt(tf.math.square(tf.math.subtract(y_true, y_false)))) / 2
        return loss
