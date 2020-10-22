import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

#FASHION MNIST IS A DATASET THAT COMES ALONGWITH KERAS
fashion_mnist = keras.datasets.fashion_mnist

#SPLITTING INTO TRAIN AND TEST SETS
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ["T-Shirt/Top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle Boot"]

#DATA PREPROCESSING
train_images = train_images/255.0
test_images = test_images/255.0

#CREATING THE MODEL
model = keras.Sequential([ #Sequential is for NOT recurrent or convolutional
    keras.layers.Flatten(input_shape=(28,28)), #input
    keras.layers.Dense(128, activation="relu"), #hidden
    keras.layers.Dense(10, activation="softmax"), #output
])

#COMPILING THE MODEL
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

#TRAINING THE MODEL
model.fit(train_images, train_labels, epochs=10)

#EVALUATING THE MODEL
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=1)
print("Test accuracy: ", test_acc)

#model.predict() to predict data
 
