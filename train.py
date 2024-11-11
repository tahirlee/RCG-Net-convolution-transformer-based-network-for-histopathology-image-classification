import tensorflow as tf
import os
import numpy as np
import config
from config import EPOCHS, BATCH_SIZE
from prepare_data import get_datasets

from models.RCGNet import RCGNet

# Set the visible GPUs
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_visible_devices(physical_devices, 'GPU')

# Set GPU memory growth
for physical_device in physical_devices:
    tf.config.experimental.set_memory_growth(physical_device, True)

def set_seed(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)

def get_model():

    input_shape = (256, 256, 3)
    weight_decay = 1e-4  # Set the weight decay value

    model = RCGNet(input_shape, 5)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)

    # Add weight decay to the optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=optimizer.lr,  # Keep the original learning rate
        beta_1=optimizer.beta_1,
        beta_2=optimizer.beta_2,
        epsilon=optimizer.epsilon,
        decay=weight_decay
    )
    model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model

def scheduler(epoch, lr):
  if epoch < 10:
    return lr
  else:
    return lr * tf.math.exp(-0.1)


if __name__ == '__main__':

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    set_seed(123)  # Set the desired seed value
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # Check if GPU is available
    print(tf.config.list_physical_devices('GPU'))

    # Check if TensorFlow is using GPU acceleration
    print("Tensorflow is using GPU:", tf.test.is_built_with_cuda())


    # Check if eager mode on
    print(tf.executing_eagerly())
    print(tf.__version__)

    # Load data
    train_generator, valid_generator, test_generator, \
    train_num, valid_num, test_num = get_datasets()


    checkpoint_filepath = config.trained_model_name
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor='val_loss',
        mode='min',
        save_best_only=True)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, verbose=2, mode='min', restore_best_weights=True)


    reduce_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    callback_list = [model_checkpoint_callback,early_stop]


    model = get_model()
    model.summary()

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    # start training
    hist = model.fit(train_generator,
                        epochs=EPOCHS,
                        steps_per_epoch=train_num // BATCH_SIZE,
                        validation_data=valid_generator,
                        validation_steps=valid_num // BATCH_SIZE,
                        callbacks=callback_list)

    np.save('RCG_history.npy', hist.history)

    model.save(config.trained_model_name)

