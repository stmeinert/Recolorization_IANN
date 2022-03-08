import tensorflow as tf
import tensorflow_datasets as tfds

# !pip install tensorflow-io
import tensorflow_io as tfio

import numpy as np


SIZE = (128,128)
BATCH_SIZE = 32

#################################################
# Prepare data
#################################################

def resize(image):
    return tf.image.resize_with_pad(image, target_height=SIZE[0], target_width=SIZE[1], method=tf.image.ResizeMethod.BILINEAR)


def to_lab(image):
    # expects input to be normalized to [0;1]!!
    # output channels are [l,a,b]
    return tfio.experimental.color.rgb_to_lab(image)


def to_grayscale(image):
    # take l channel (size index starts at one^^)
    image = tf.slice(image, begin=[0, 0, 0], size=[-1, -1, 1])
    return image

def prepare_image_data(image_ds):
    # resize image to desired dimension, replace label with colored image
    image_ds = image_ds.map(lambda x: (resize(x['image']), resize(x['image'])))

    # normalize data to [0;1) for lab encoder
    image_ds = image_ds.map(lambda image, target: ((image/256), (target/256)))

    # convert image and target image to lab color space
    image_ds = image_ds.map(lambda image, target: (to_lab(image), to_lab(target)))

    # only take l channel of input tensor
    image_ds = image_ds.map(lambda image, target: (to_grayscale(image), target))

    # l in lab is in [0;100] -> normalize to [0;1]/[-1;1]?
    # ab are in range [-128;127]
    image_ds = image_ds.map(lambda image, target: ((image/50)-1, target))

    image_ds = image_ds.shuffle(1000).batch(BATCH_SIZE)#.prefetch(20)
    return image_ds


#################################################
# Train and Test steps
#################################################


def train_step(model, input, target, loss_function, optimizer):
    """
    This function executes a training step on the given network.
    Using gradient descend
    :param model: the network model
    :param input: the given input tensors
    :param target: the given target tensors
    :param loss_function: the given loss function
    :param optimizer: the given optimizer
    :return: the loss during this trainin step
    """
    # loss_object and optimizer_object are instances of respective tensorflow classes
    with tf.GradientTape() as tape:
        prediction = model(input, training=True)

        # get l channel, target should be in shape (batch, SIZE, SIZE, lab)
        l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
        prediction = tf.concat([l, prediction], axis=-1) # should be concatenated along last dimension

        loss = loss_function(target, prediction)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def test(model, test_data, loss_function):
    """
    # Tests over complete test data
    :param model: the network model
    :param test_data: the given input tensors
    :param loss_function: the given loss function
    :return: the test loss and test accuracy
    """
    test_accuracy_aggregator = []
    test_loss_aggregator = []
    for (input, target) in test_data:
        prediction = model(input)

        # get l channel, target should be in shape (batch, SIZE, SIZE, lab)
        l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
        prediction = tf.concat([l, prediction], axis=-1) # should be concatenated along last dimension

        sample_test_loss = loss_function(target, prediction)
        
        # TODO -> what is the accuracy here?
        #sample_test_accuracy =  np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = tf.reduce_all(tf.equal(prediction, target))
        #sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_aggregator.append(sample_test_loss.numpy())
        test_accuracy_aggregator.append(sample_test_accuracy)

    test_loss = tf.reduce_mean(test_loss_aggregator)
    test_accuracy = tf.reduce_sum(tf.cast(test_accuracy_aggregator, tf.float32)) / tf.size(test_accuracy_aggregator, out_type=tf.float32)
    return test_loss, test_accuracy