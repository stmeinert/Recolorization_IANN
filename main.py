from multiprocessing.spawn import prepare
import tensorflow as tf
import tensorflow_datasets as tfds

from data_pipeline.data_pipeline import *

from datetime import datetime

from zhang import CIC
from izuka import IizukaRecolorizationModel
from zhang_prob import CIC_Prob

from loss.l2_loss import L2_Loss

TRAIN_IMAGES = 250
TEST_IMAGES = 10
VAL_IMAGES = 10


if __name__ == '__main__':
    train_ds, test_ds, val_ds = tfds.load(name='places365_small', 
                                      split=(f'train[:{TRAIN_IMAGES}]', f'test[:{TEST_IMAGES}]', f'validation[:{VAL_IMAGES}]'))
    
    train_ds = train_ds.apply(prepare_image_data)
    test_ds = test_ds.apply(prepare_image_data)
    val_ds = val_ds.apply(prepare_validation_data)

    # parameters
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.001

    # model and loss
    model = CIC_Prob()
    #model = IizukaRecolorizationModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    #loss = tf.keras.losses.MeanSquaredError()
    loss = L2_Loss()

    # lists for losses and accuracies
    train_losses = []
    test_losses = []
    test_accuracies = []

    # testing once before we begin
    #test_loss, test_accuracy = test(model, test_ds, loss)
    test_loss = model.test(test_ds)
    test_losses.append(test_loss)
    
    # check how model performs on train data once before we begin
    #train_loss, _ = test(model, train_ds, loss)
    print("Testing untrained model on training data")
    train_loss = model.test(train_ds)
    train_losses.append(train_loss)

    # Sets up a timestamped log directory.
    logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    # save first version validation images before training starts
    print("Getting first example images from untrained model")
    for input, target in val_ds.take(1):
        prediction = model(input, training=False)

        # get l channel, target should be in shape (SIZE, SIZE, lab)
        l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
        prediction = tf.concat([l, prediction], axis=-1) # should be concatenating along last dimension

        # convert prediction and target back to rgb, input to [0;1]
        input = (input+1)/2
        #print(prediction)
        #print(target)
        prediction = tfio.experimental.color.lab_to_rgb(prediction)
        target = tfio.experimental.color.lab_to_rgb(target)

        #images.append((input, prediction, target))

        with file_writer.as_default():
            tf.summary.image(f'Input', input, max_outputs=BATCH_SIZE, step=-1)
            tf.summary.image('Target', target, max_outputs=BATCH_SIZE, step=-1)
            tf.summary.image('Prediciton', prediction, max_outputs=BATCH_SIZE, step=-1)

    
    global_start_time = datetime.now()
    #   We train for num_epochs epochs.
    for epoch in range(NUM_EPOCHS):
        start_time = datetime.now()
        print(f'Epoch: {str(epoch)} starting with test loss {test_losses[-1]}')
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in train_ds:
            #train_loss = train_step(model, input, target, loss, optimizer)
            train_loss = model.train_step((input, target))
            epoch_loss_agg.append(train_loss)
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))
        # testing, so we can track accuracy and test loss
        #test_loss, test_accuracy = test(model, test_ds, loss)
        test_loss = model.test(test_ds)
        test_losses.append(test_loss)
        #test_accuracies.append(test_accuracy)
        diff_time = datetime.now() - start_time
        print(f"Epoch {epoch} took {diff_time} to complete.")

        with file_writer.as_default():
            tf.summary.scalar("Train Loss", train_loss, step=epoch)
            tf.summary.scalar("Test Loss", test_loss, step=epoch)

        for input, target in val_ds.take(1):
            prediction = model(input)

            # get l channel, target should be in shape (SIZE, SIZE, lab)
            l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
            prediction = tf.concat([l, prediction], axis=-1) # should be concatenating along last dimension

            # convert prediction and target back to rgb, input to [0;1]
            input = (input+1)/2
            prediction = tfio.experimental.color.lab_to_rgb(prediction)
            target = tfio.experimental.color.lab_to_rgb(target)

            # write images to log directory
            with file_writer.as_default():
                tf.summary.image('Input', input, max_outputs=BATCH_SIZE, step=epoch)
                tf.summary.image('Target', target, max_outputs=BATCH_SIZE, step=epoch)
                tf.summary.image('Prediciton', prediction, max_outputs=BATCH_SIZE, step=epoch)

    whole_training_time = datetime.now() - global_start_time
    print(f'Training took {whole_training_time} to complete')