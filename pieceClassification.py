from os import listdir
import json

import math
import numpy as np
# import sklearn.cluster
# import matplotlib.pyplot as plt

import cv2 as cv

import tensorflow as tf
import keras
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
# from keras.applications import InceptionV3
# from keras.applications.InceptionV3 import InceptionV3
from keras import layers









# Load test, train, and valid dataset.
test_data_x, test_data_y = None, None
train_data_x, train_data_y = None, None
valid_data_x, valid_data_y = None, None

# Model parameters
batch_size = 128
learning_rate = 0.001  

# Make Base Model
base_model = InceptionV3(weights = 'imagenet', input_shape= (75, 100, 3), include_top = False, name = "pc-inception-v3") # are we doing greyscale or color?

# Freeze base_model
base_model.trainable = False

# Do we need to also add our data to the inputs?
inputs = keras.Input(shape=(75, 100, 3))

# Calculate Outputs
x = base_model(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D(name = "pc-pooling") * (base_model.output) 
output = keras.layers.Dense(12, activation = "softmax", name = "pc-predictions")(x)

#Make Model
model = keras.Model(inputs = inputs, outputs = output)

model.summary()


# Compile, Train, And Save Model


model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                    loss = keras.losses.CategoricalCrossentropy(),
                    metrics = ['accuracy'])


epochs = 2 # How to change amount of epochs for classification head vs whole network? Also the piece classifaction should be double the occupancy classification
model.fit(train_data_x, train_data_y,
                validation_set = (valid_data_x, valid_data_y), epochs= epochs)

model.save(filepath, overwrite = True)














    # train = True
    # filepath = "chess_classifier.keras"

    # # Load test dataset.
    # test_data_x, test_data_y = None, None

    # # Train the model if requested.
    # if (train):
    #     # Load training dataset
    #     train_data_x, train_data_y = None, None
    #     valid_data_x, valid_data_y = None, None
        
    #     # Model parameters
    #     batch_size = 128
    #     learning_rate = 0.001    # Learning rate for piece classifier.

    #     # Fine-tune InceptionV3 to act as a chess piece classifier
    #     base_model = InceptionV3(weights = 'imagenet', include_top = False, name = "pc-inception-v3")
    #     x = keras.GlobalAveragePooling2D(name = "pc-pooling")(base_model.output)
    #     x = keras.Dense(1024, name = "pc-dense")(x)
    #     # 2 x (King, Queen, Rook, Bishop, Knight, Pawn) for Black/White
    #     predictions = keras.Dense(12, activation = "softmax", name = "pc-predictions")(x)
    #     model = keras.Model(inputs = base_model.input, outputs = predictions)
    #     model.summary()

    #     # Compile, train, and save the dataset
    #     model.compile(optimizer = keras.optimizers.Adam(learning_rate),
    #                 loss = keras.losses.CategoricalCrossentropy(),
    #                 metrics = ['accuracy'])
    #     model.fit(train_data_x, train_data_y,
    #             validation_set = (valid_data_x, valid_data_y),
    #             batch_size = batch_size, validation_batch_size = batch_size)
    #     model.save(filepath, overwrite = True)

    # # Test the model on the test dataset.