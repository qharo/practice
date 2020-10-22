import os
import numpy as np
import matplotlib as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

#SPLITTING THE DATA INTO TRAINING AND TEST SETS
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs', split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True, as_supervised=True
)

get_label_name = metadata.features['label'].int2str

#RESHAPING ALL IMAGES
IMG_SIZE = 160
def format_example(image, label):
    image = tf.cast(image, tf.float32)
    image = (image/127.5) -1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label

train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

#SHUFFLE AND BATCH IMAGES
BATCH_SIZE = 32
SHUFFLE_BATCH_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BATCH_SIZE).batch(BATCH_SIZE)
validation_batches = validation.shuffle(SHUFFLE_BATCH_SIZE).batch(BATCH_SIZE)
test_batches = test.shuffle(SHUFFLE_BATCH_SIZE).batch(BATCH_SIZE)

#SETTING THE PRETRAINED MODEL
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

#FREEZING BASE
base_model.trainable = False

#ADDING OUR OWN CLASSIFIER
global_average_layer = keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)
model = keras.Sequential([
    base_model, 
    global_average_layer, 
    prediction_layer
])

#TRAINING THE MODEL
base_learning_rate = 0.0001
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=base_learning_rate),
    loss = keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)
initial_epochs = 3
validation_steps = 20
loss0, accuracy0 = model.evaluate(validation_batches, steps=validation_steps)

history = model.fit(train_batches, epochs=initial_epochs, validation_data=validation_batches)
acc = history.history['accuracy']
print(acc)