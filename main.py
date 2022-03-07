import tensorflow as tf
import tensorflow_datasets as tfds

from data_pipeline.data_pipeline import *



if __name__ == '__main__':
    train_ds, test_ds, val_ds = tfds.load(name='places365_small', 
                                      split=(f'train[:{TRAIN_IMAGES}]', f'test[:{TEST_IMAGES}]', f'validation[:{VAL_IMAGES}]'))
    
    print(train_ds.element_spec)
    
    train_ds = train_ds.apply(prepare_image_data)
    test_ds = test_ds.apply(prepare_image_data)
    val_ds = val_ds.apply(prepare_image_data)

    # Sets up a timestamped log directory.
    #logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    #file_writer = tf.summary.create_file_writer(logdir)

    #with file_writer.as_default():
    #    tf.summary.image("First dimension", train_ds[:25], max_outputs=25, step=0)

    # Load the TensorBoard notebook extension.
    #%load_ext tensorboard
    #%tensorboard --logdir logs/train_data