from types import NoneType
import tensorflow as tf
import tensorflow_datasets as tfds

from data_pipeline.data_pipeline import *

from datetime import datetime


if __name__ == '__main__':
    train_ds, test_ds, val_ds = tfds.load(name='places365_small', 
                                      split=(f'train[:{TRAIN_IMAGES}]', f'test[:{TEST_IMAGES}]', f'validation[:{VAL_IMAGES}]'))
    
    train_ds = train_ds.apply(prepare_image_data)
    test_ds = test_ds.apply(prepare_image_data)
    val_ds = val_ds.apply(prepare_image_data)

    # parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.1

    # model and loss
    model = None
    optimizer = None
    loss = None

    # lists for losses and accuracies
    train_losses = []
    test_losses = []
    test_accuracies = []

     # testing once before we begin
    test_loss, test_accuracy = test(model, test_ds, loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)
    
    # check how model performs on train data once before we begin
    train_loss, _ = test(model, train_ds, loss)
    train_losses.append(train_loss)
    
    #   We train for num_epochs epochs.
    for epoch in range(NUM_EPOCHS):
        start_time = datetime.datetime.now()
        print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_ds:
            train_loss = train_step(model, input, target, loss, optimizer)
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        test_loss, test_accuracy = test(model, test_ds, loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)
        diff_time = datetime.datetime.now() - start_time
        print(f"Epoch {epoch} took {diff_time} to complete.")


    # Sets up a timestamped log directory.
    #logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    #file_writer = tf.summary.create_file_writer(logdir)

    #with file_writer.as_default():
    #    tf.summary.image("First dimension", train_ds[:25], max_outputs=25, step=0)

    # Load the TensorBoard notebook extension.
    #%load_ext tensorboard
    #%tensorboard --logdir logs/train_data