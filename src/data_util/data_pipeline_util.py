
import tensorflow as tf

# !pip install tensorflow-io
import tensorflow_io as tfio

import zipfile
import os

#################################################
# Prepare data
#################################################

def resize(image, size):
    return tf.image.resize_with_pad(image, target_height=size[0], target_width=size[1], method=tf.image.ResizeMethod.BILINEAR)


def to_lab(image):
    # expects input to be normalized to [0;1]!!
    # output channels are [l,a,b]
    return tfio.experimental.color.rgb_to_lab(image)


def to_grayscale(image):
    # take l channel (size index starts at one^^)
    image = tf.slice(image, begin=[0, 0, 0], size=[-1, -1, 1])
    return image

def prepare_image_data(image_ds, batch_size):
    # resize image to desired dimension, replace label with colored image
    image_ds = image_ds.map(lambda x: (resize(x['image']), resize(x['image'])))

    # normalize data to [0;1) for lab encoder
    image_ds = image_ds.map(lambda image, target: ((image/256), (target/256)))

    # convert image and target image to lab color space
    image_ds = image_ds.map(lambda image, target: (to_lab(image), to_lab(target)))

    # only take l channel of input tensor
    image_ds = image_ds.map(lambda image, target: (to_grayscale(image), target))

    # l in lab is in [0;100] -> normalize to [-1;1]
    # ab are in range [-128;127]
    image_ds = image_ds.map(lambda image, target: ((image/50)-1, target))

    image_ds = image_ds.shuffle(1000).batch(batch_size).prefetch(20)
    return image_ds


def prepare_validation_data(image_ds, batch_size):
    """
    Same as for train and test data, but don't shuffle so you can the progress over same image in tensorboard
    """
    # resize image to desired dimension, replace label with colored image
    image_ds = image_ds.map(lambda x: (resize(x['image']), resize(x['image'])))

    # normalize data to [0;1) for lab encoder
    image_ds = image_ds.map(lambda image, target: ((image/256), (target/256)))

    # convert image and target image to lab color space
    image_ds = image_ds.map(lambda image, target: (to_lab(image), to_lab(target)))

    # only take l channel of input tensor
    image_ds = image_ds.map(lambda image, target: (to_grayscale(image), target))

    # l in lab is in [0;100] -> normalize to [-1;1]
    # ab are in range [-128;127]
    image_ds = image_ds.map(lambda image, target: ((image/50)-1, target))

    image_ds = image_ds.batch(batch_size).prefetch(20)
    return image_ds


def unzip_and_load_ds(ds_name, extract_ds_path, zip_ds_path):
    path = os.path.join(os.getcwd(), extract_ds_path, 'content', ds_name)

    # only extract again if path does not exist!
    if not os.path.exists(path):
      with zipfile.ZipFile(zip_ds_path, 'r') as zip_ref:
          zip_ref.extractall(zip_ds_path)

    return tf.data.experimental.load(path,compression= 'GZIP')

