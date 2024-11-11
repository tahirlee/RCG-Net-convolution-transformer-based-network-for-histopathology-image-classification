import tensorflow as tf
import config

def get_datasets():
    # Preprocess the dataset
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(config.train_dir,
                                                        subset='training',
                                                        target_size=(config.image_height, config.image_width),
                                                        color_mode="rgb",
                                                        batch_size=config.BATCH_SIZE,
                                                        seed=1234,
                                                        shuffle=True,
                                                        class_mode="categorical")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0)
    validation_data = val_datagen.flow_from_directory(config.val_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=1234,
                                                      shuffle=True,
                                                      class_mode="categorical")

    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0
    )
    test_generator = test_datagen.flow_from_directory(config.test_dir,
                                                      target_size=(config.image_height, config.image_width),
                                                      color_mode="rgb",
                                                      batch_size=config.BATCH_SIZE,
                                                      seed=1234,
                                                      shuffle=True,
                                                      class_mode="categorical")

    train_num = train_generator.samples
    valid_num = validation_data.samples
    test_num = test_generator.samples

    return train_generator, validation_data, test_generator, train_num, valid_num, test_num
