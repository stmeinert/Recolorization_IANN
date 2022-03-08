import tensorflow as tf
import tensorflow_datasets as tfds

from data_pipeline.data_pipeline import *

from datetime import datetime


TRAIN_IMAGES = 100
TEST_IMAGES = 100
VAL_IMAGES = 100


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
    #input = tf.keras.Input(shape=(SIZE[0],SIZE[1],1))
    #conv2d = tf.keras.layers.Conv2D(filters=2, kernel_size=3, strides=1, padding='same', activation=None, use_bias=True)(input)
    #model = tf.keras.Model(inputs=input, outputs=conv2d)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    loss = tf.nn.l2_loss

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

    # Sets up a timestamped log directory.
    logdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    # Creates a file writer for the log directory.
    file_writer = tf.summary.create_file_writer(logdir)
    
    #   We train for num_epochs epochs.
    for epoch in range(NUM_EPOCHS):
        start_time = datetime.now()
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
        diff_time = datetime.now() - start_time
        print(f"Epoch {epoch} took {diff_time} to complete.")

        images = []
        for input, target in test_ds.take(25):
            prediction = model(input)

            # get l channel, target should be in shape (SIZE, SIZE, lab)
            l = tf.slice(target, begin=[0,0,0], size=[-1,-1,1])
            prediction = tf.concat([l, prediction], axis=-1) # should be concatenating along last dimension

            # convert prediction and target back to rgb, input to [0;1]
            input = (input+1)/2
            prediction = tfio.experimental.color.lab_to_rgb(prediction)
            target = tfio.experimental.color.lab_to_rgb(target)

            images.append((input, prediction, target))

            with file_writer.as_default():
                tf.summary.image(f'Epoch{epoch}', images, max_outputs=25, step=0)



    # Load the TensorBoard notebook extension.
    #%load_ext tensorboard
    #%tensorboard --logdir logs/train_data