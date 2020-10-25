import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import os 
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

#LOADING DATA
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

#PREPROCESSING TO ENSURE INPUT HAS THE SAME LENGTH
train_data = sequence.pad_sequences(train_data, MAXLEN)
test_data = sequence.pad_sequences(test_data, MAXLEN)

#CREATING THE MODEL
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

#COMPILE MODEL
model.compile(loss="binary_crossentropy", optimizer="rmsprop", metrics=['acc'])

#TRAIN MODEL
history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)

#RESULTS
results = model.evaluate(test_data, test_labels)
print(results)

