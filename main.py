try:
    import google.colab
    IN_COLAB = True
except:
    IN_COLAB = False

import tensorflow as tf
import tensorflow_datasets as tfds

from data_pipeline.data_pipeline import *

from datetime import datetime

from tqdm import tqdm

if not IN_COLAB:
    from zhang import CIC
    from izuka import IizukaRecolorizationModel
    from zhang_prob import CIC_Prob

    from loss.l2_loss import L2_Loss



##############################################
# Parameters which the user can set
##############################################
# number of epochs to train
NUM_EPOCHS = 20

# which dataset to use
PLACES = 0
CELEB = 1
DS = CELEB

# size of training, test and validation sets
TRAIN_IMAGES = 32
TEST_IMAGES = 32
VAL_IMAGES = 32


# which model to use
ZHANG = 0
ZHANG_PROB = 1
IZUKA = 2

MODEL = ZHANG_PROB

# logdir root 
#   -> use tensorboard --logdir <LOGDIR_ROOT> to view logs
LOGDIR_ROOT = "logs/train_data/"

###############################################
# End of settable parameters
###############################################


# celeb dataset is already batched
if DS == CELEB:
    TRAIN_IMAGES = 1 if (TRAIN_IMAGES // BATCH_SIZE) == 0 else (TRAIN_IMAGES // BATCH_SIZE)
    TEST_IMAGES = 1 if (TEST_IMAGES // BATCH_SIZE) == 0 else (TEST_IMAGES // BATCH_SIZE)
    VAL_IMAGES = 1 if (VAL_IMAGES // BATCH_SIZE) == 0 else (VAL_IMAGES // BATCH_SIZE)



if __name__ == '__main__':
    train_ds, test_ds, val_ds = None, None, None

    if DS == PLACES:
        train_ds, test_ds, val_ds = tfds.load(name='places365_small', 
                                        split=(f'train[:{TRAIN_IMAGES}]', f'test[:{TEST_IMAGES}]', f'validation[:{VAL_IMAGES}]'))
        
        train_ds = train_ds.apply(prepare_image_data)
        test_ds = test_ds.apply(prepare_image_data)
        val_ds = val_ds.apply(prepare_validation_data)
    else:
        ds = unzip_and_load_ds()
        test_ds = ds.take(TEST_IMAGES)
        val_ds = ds.skip(TEST_IMAGES).take(VAL_IMAGES)
        train_ds = ds.skip(TEST_IMAGES+VAL_IMAGES).take(TRAIN_IMAGES)
    
    
    # model and loss
    model = None

    if MODEL == ZHANG:
        model = CIC()
    elif MODEL == ZHANG_PROB:
        model = CIC_Prob()
    elif MODEL == IZUKA:
        model = IizukaRecolorizationModel()
    else:
        print(f"This model class does not exist! ({MODEL})")
        exit(0)

    # lists for losses and accuracies
    train_losses = []
    test_losses = []
    test_accuracies = []

    # testing once before we begin
    print("Testing untrained model on test data")
    test_loss = model.test(test_ds)
    test_losses.append(test_loss)
    
    # check how model performs on train data once before we begin
    print("Testing untrained model on training data")
    train_loss = model.test(train_ds)
    train_losses.append(train_loss)

    # Sets up a timestamped log directory.
    logdir = LOGDIR_ROOT + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)

    # save first version validation images before training starts
    print("Getting first example images from untrained model")
    for input, target in val_ds.take(1):
        prediction = model(input, training=False)

        # get l channel, target should be in shape (BATCH, SIZE, SIZE, lab)
        l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
        prediction = tf.concat([l, prediction], axis=-1) # should be concatenating along last dimension

        # convert prediction and target back to rgb, input to [0;1]
        input = (input+1)/2
        prediction = tfio.experimental.color.lab_to_rgb(prediction)
        target = tfio.experimental.color.lab_to_rgb(target)

        # write input, prediction and target to log directory
        with file_writer.as_default():
            tf.summary.image(f'Input', input, max_outputs=BATCH_SIZE, step=-1)
            tf.summary.image('Target', target, max_outputs=BATCH_SIZE, step=-1)
            tf.summary.image('Prediciton', prediction, max_outputs=BATCH_SIZE, step=-1)

    
    # track time for whole training
    global_start_time = datetime.now()

    #   We train for num_epochs epochs.
    for epoch in range(NUM_EPOCHS):
        start_time = datetime.now()
        print(f'Epoch: {str(epoch)} starting with test loss {test_losses[-1]}')
        
        # training (and checking in with training)
        epoch_loss_agg = []
        for input, target in tqdm(train_ds):
            train_loss = model.train_step((input, target))
            epoch_loss_agg.append(train_loss)
        
        # track training loss
        train_losses.append(tf.reduce_mean(epoch_loss_agg))

        # testing on test dataset, so we can get test loss
        test_loss = model.test(test_ds)
        test_losses.append(test_loss)

        # track epoch duration
        diff_time = datetime.now() - start_time
        print(f"Epoch {epoch} took {diff_time} to complete.")

        # write training and test loss to logfile
        with file_writer.as_default():
            tf.summary.scalar("Train Loss", train_loss, step=epoch)
            tf.summary.scalar("Test Loss", test_loss, step=epoch)

        # save precited sample images
        for input, target in val_ds.take(1):
            prediction = model(input)

            # get l channel, target should be in shape (BATCH, SIZE, SIZE, lab)
            l = tf.slice(target, begin=[0,0,0,0], size=[-1,-1,-1,1])
            prediction = tf.concat([l, prediction], axis=-1) # should be concatenating along last dimension

            # convert prediction and target back to rgb, input to [0;1]
            input = (input+1)/2
            prediction = tfio.experimental.color.lab_to_rgb(prediction)
            target = tfio.experimental.color.lab_to_rgb(target)

            # write images to log directory
            with file_writer.as_default():
                #tf.summary.image('Input', input, max_outputs=BATCH_SIZE, step=epoch)
                #tf.summary.image('Target', target, max_outputs=BATCH_SIZE, step=epoch)
                tf.summary.image('Prediciton', prediction, max_outputs=BATCH_SIZE, step=epoch)

    whole_training_time = datetime.now() - global_start_time
    print(f'Training took {whole_training_time} to complete')