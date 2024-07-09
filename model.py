from os import listdir
import json
import time

import math
import numpy as np

import cv2 as cv

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
    images, labels = [], []
    square_to_piece = {piece[1]: piece[0] for piece in piece_data}
    col_letters = 'abcdefgh'
    labels_to_int = {'E': 0, 'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6, 'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12}
    for piece_i in range(8):
        for piece_j in range(8):
            # Find the label
            square = col_letters[piece_j] + str(8-piece_i) if white_view else col_letters[7-piece_j] + str(piece_i + 1)
            if (square in square_to_piece): label = square_to_piece[square]
            else: label = 'E'
            labels.append(labels_to_int[label])

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
    return images, labels

if __name__ == "__main__":
    
    def gather_data(imname):
        im = cv.imread("Data/train/" + imname + ".png")
        metadata = json.load(open("Data/train/" + imname + ".json"))
        pieces = [(p['piece'], p['square'], p['box']) for p in metadata['pieces']]
        corners = metadata['corners']
        white_view = metadata['white_turn']

        # Board localization
        images, labels = board_localization(im, pieces, corners, white_view, False)
        one_hot_labels, empty_labels = [], []
        for l in labels:
            one_hot = np.zeros(13)
            one_hot[l] = 1
            one_hot_labels.append(one_hot)
            is_empty = np.zeros(2)
            is_empty[min(l, 1)] = 1
            empty_labels.append(is_empty)
        del labels
        return images, one_hot_labels, empty_labels
    
    train = True
    start_time = time.time()
    if train:
        # Gather and read the training and validation datasets.
        train_files = listdir("Data/train")
        train_images, train_labels, train_empty_labels = [], [], []
        
        print("Loading training dataset...")
        for imname in train_files:
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, one_hot_labels, empty_labels = gather_data(x)
            train_images.append(images)
            train_labels.append(one_hot_labels)
            train_empty_labels.append(empty_labels)
        
        valid_files = listdir("Data/train")
        valid_images, valid_labels, valid_empty_labels = [], [], []
        
        print("Loading validation dataset...")
        for imname in train_files:
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, one_hot_labels, empty_labels = gather_data(x)
            valid_images.append(images)
            valid_labels.append(one_hot_labels)
            valid_empty_labels.append(empty_labels)
        
        print("Datasets loaded in %.3f seconds." % (time.time() - start_time))

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

    #     # Compile, train, and save the dataset
    #     model.compile(optimizer = keras.optimizers.Adam(learning_rate),
    #                 loss = keras.losses.CategoricalCrossentropy(),
    #                 metrics = ['accuracy'])
    #     model.fit(train_data_x, train_data_y,
    #             validation_set = (valid_data_x, valid_data_y),
    #             batch_size = batch_size, validation_batch_size = batch_size)
    #     model.save(filepath, overwrite = True)

    # # Test the model on the test dataset.