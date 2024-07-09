from os import listdir
import json
import time

import math
import numpy as np
import cv2 as cv

import keras
from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras._tf_keras.keras.applications.resnet import ResNet101

# Given an image of a chessboard and its metadata, split it into a collection of 64 tiles
# labeled by the pieces (or lack thereof) which occupy that tile.
# Returns a 64 element array of tile images and their labels.
def board_localization(image, piece_data, corners, white_view, inner_grid):
    # Identify the corners of the image.
    height, width, _ = image.shape
    top_left, temp = None, -1
    for p in corners:
        dist = math.sqrt(p[0]*p[0] + p[1]*p[1])
        if (temp == -1 or dist < temp):
            temp = dist
            top_left = p
    bottom_left, temp = None, -1
    for p in corners:
        dist = math.sqrt(p[0]*p[0] + (p[1]-height)**2)
        if (temp == -1 or dist < temp):
            temp = dist
            bottom_left = p
    bottom_right, temp = None, -1
    for p in corners:
        dist = math.sqrt((p[0]-width)**2 + (p[1]-height)**2)
        if (temp == -1 or dist < temp):
            temp = dist
            bottom_right = p
    top_right, temp = None, -1
    for p in corners:
        dist = math.sqrt((p[0]-width)**2 + p[1]**2)
        if (temp == -1 or dist < temp):
            temp = dist
            top_right = p

    # Fix the top-left corner and find a mapping of the remaining corners to a rectangular grid
    x, y = top_left
    source_points = np.asarray([top_left, top_right, bottom_left, bottom_right], dtype = np.float32)
    dest_points = np.asarray([top_left, (top_right[0], y), (x, bottom_left[1]), (top_right[0], bottom_left[1])], dtype = np.float32)
    A = cv.getPerspectiveTransform(source_points, dest_points)
    warped = cv.warpPerspective(image, A, (width, height))
    
    # Break into tiles
    x_prime, y_prime, t = np.dot(A, np.asarray([top_left[0], top_left[1], 1]))
    warped_top_left = (round(x_prime/t), round(y_prime/t))
    x_prime, y_prime, t = np.dot(A, np.asarray([bottom_right[0], bottom_right[1], 1]))
    warped_bottom_right = (round(x_prime/t), round(y_prime/t))
    # Pad tiles to include one tile to the left and right and two tiles above the chessboard.
    tiles = np.zeros((10, 10, 4))  # Tiles are represented by the top-left and bottom-right points.
    dx, dy = abs(warped_bottom_right[0] - warped_top_left[0]), abs(warped_bottom_right[1] - warped_top_left[1])
    sx, sy = dx/8, dy/8
    # If the corners actually specify the inner 6x6 set of tiles, extend the top left and bottom right so that
    # they reach the corners.
    if (inner_grid):
        warped_top_left = (warped_top_left[0] - sx, warped_top_left[1] - sy)
        warped_bottom_right = (warped_bottom_right[0] - sx, warped_bottom_right[1] - sy)
    for i in range(-2, 8):
        y = warped_top_left[1] + sy * i
        next_y = warped_top_left[1] + sy * i + sy
        for j in range(-1, 9):
            x = warped_top_left[0] + sx * j
            next_x = warped_top_left[0] + sx * j + sx
            tiles[i][j] = (x, y, next_x, next_y)
    tiles = np.asarray(tiles).reshape(10, 10, 4)

    # Crop warped image
    crop_width, crop_height = 75, 125
    images, piece_images, piece_labels, empty_labels = [], [], [], []
    square_to_piece = {piece[1]: piece[0] for piece in piece_data}
    col_letters = 'abcdefgh'
    labels_to_int = {'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6, 'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12}
    for piece_i in range(8):
        for piece_j in range(8):
            # Find the label
            square = col_letters[piece_j] + str(8-piece_i) if white_view else col_letters[7-piece_j] + str(piece_i + 1)
            if (square in square_to_piece): label = labels_to_int[square_to_piece[square]]
            else: label = 0

            # Convert labels to one-hot encoding
            if (label != 0):
                one_hot = np.zeros(12)
                one_hot[label - 1] = 1
                piece_labels.append(one_hot)
            is_empty = np.zeros(2)
            is_empty[min(label, 1)] = 1
            empty_labels.append(is_empty)

            # p1w = (X0, Y0)
            # p2w = (X1, Y0)
            # p3w = (X0, Y1)
            # p4w = (X1, Y1)

            # Linear interpolation of X coordinates so that the crop stretches as we get closer to the edge.
            flip = False
            if (piece_j <= 3):
                alpha = abs(3 - piece_j)/3
                X0 = round((1-alpha) * tiles[piece_i - 2][piece_j][0] + alpha * tiles[piece_i - 2][piece_j - 1][0])
                X1 = round(tiles[piece_i - 2][piece_j][2])
                flip = True
            else:
                alpha = abs(4 - piece_j)/3
                X0 = round(tiles[piece_i - 2][piece_j][0])
                X1 = round((1-alpha) * tiles[piece_i-2][piece_j][2] + alpha * tiles[piece_i - 2][piece_j + 1][2])
            Y0 = round(tiles[piece_i - 2][piece_j][1])
            Y1 = round(tiles[piece_i][piece_j][3])
           
           # Crop the image
            crop = cv.resize(warped[Y0:Y1, min(X0,X1):max(X0,X1)], (crop_width, crop_height))
            if (flip): crop = cv.flip(crop, 1)
            images.append(crop)
            if (label == 0): piece_images.append(crop)
    return images, piece_images, piece_labels, empty_labels

if __name__ == "__main__":
    
    def gather_data(imname):
        im = cv.imread("Data/train/" + imname + ".png")
        metadata = json.load(open("Data/train/" + imname + ".json"))
        pieces = [(p['piece'], p['square'], p['box']) for p in metadata['pieces']]
        corners = metadata['corners']
        white_view = metadata['white_turn']
        # Board localization
        return board_localization(im, pieces, corners, white_view, False)
    
    train = True
    start_time = time.time()
    if train:
        # Gather and read the training and validation datasets.
        train_files = listdir("Data/train")
        train_images, train_piece_images, train_piece_labels, train_empty_labels = [], [], [], []
        
        print("Loading training dataset...")
        for imname in train_files:
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, piece_images, one_hot_labels, empty_labels = gather_data(x)
            train_images += images
            train_piece_images += piece_images
            train_piece_labels += one_hot_labels
            train_empty_labels += empty_labels
        
        valid_files = listdir("Data/train")
        valid_images, valid_piece_images, valid_piece_labels, valid_empty_labels = [], [], [], []
        
        print("Loading validation dataset...")
        for imname in train_files:
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, piece_images, one_hot_labels, empty_labels = gather_data(x)
            valid_images += images
            valid_piece_images += piece_images
            valid_piece_labels += one_hot_labels
            valid_empty_labels += empty_labels

        print("Datasets loaded in %.3f seconds." % (time.time() - start_time))

        #=======================================#
        # Occupancy Classification (Resnet 101) #
        #=======================================#

        # Model parameters
        batch_size = 128
        learning_rate = 0.001  

        # Make Base Model
        base_model = ResNet101(weights = 'imagenet', input_shape= (75, 75, 3), include_top = False, name = "oc-resnet101")
        # Freeze base_model
        base_model.trainable = True
        inputs = keras.Input(shape = (75, 125, 3))

        # Calculate Outputs
        x = base_model(inputs, training = True)
        x = keras.layers.GlobalAveragePooling2D(name = "oc-pooling") * (base_model.output) 
        x = keras.Dense(1024, name = "oc-dense")(x)
        x = keras.Dense(256, name = "oc-dense2")(x)
        output = keras.layers.Dense(2, activation = "softmax", name = "oc-predictions")(x)

        # Make Model
        model = keras.Model(inputs = inputs, outputs = output)
        model.summary()

        # Compile, Train, And Save Model
        model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                      loss = keras.losses.CategoricalCrossentropy(),
                      metrics = ['accuracy'])
        
        num_epochs = 3
        model.fit(train_images, train_empty_labels,
                  validation_set = (valid_images, valid_empty_labels), epochs = num_epochs)
        model.save("occupancy_classifier.keras", overwrite = True)
        
        #===================================#
        # Piece Classification (Resnet 101) #
        #===================================#

        # Model parameters
        batch_size = 128
        learning_rate = 0.001  

        # Make Base Model
        base_model = InceptionV3(weights = 'imagenet', input_shape= (75, 75, 3), include_top = False, name = "pc-inception")
        # Freeze base_model
        base_model.trainable = True
        inputs = keras.Input(shape = (75, 125, 3))

        # Calculate Outputs
        x = base_model(inputs, training = True)
        x = keras.layers.GlobalAveragePooling2D(name = "pc-pooling") * (base_model.output) 
        x = keras.Dense(1024, name = "pc-dense")(x)
        x = keras.Dense(256, name = "pc-dense2")(x)
        output = keras.layers.Dense(12, activation = "softmax", name = "pc-predictions")(x)

        # Make Model
        model = keras.Model(inputs = inputs, outputs = output)
        model.summary()

        # Compile, Train, And Save Model
        model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                      loss = keras.losses.CategoricalCrossentropy(),
                      metrics = ['accuracy'])
        
        num_epochs = 6
        model.fit(train_piece_images, train_piece_labels,
                  validation_set = (valid_piece_images, valid_piece_labels), epochs = num_epochs)
        model.save("piece_classifier.keras", overwrite = True)