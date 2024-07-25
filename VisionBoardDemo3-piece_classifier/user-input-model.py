from os import listdir
import json
import time

import math
import numpy as np
import cv2 as cv

import keras
# from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
# from keras._tf_keras.keras.applications.resnet import ResNet101


#====================#
# BOARD LOCALIZATION #
#====================#

# Given an image of a chessboard and its metadata, split it into a collection of 64 tiles
# labeled by the pieces (or lack thereof) which occupy that tile.
# Returns a 64 element array of tile images and their labels.

#==================#
# INPUT PARAMETERS #
#==================#

# image = The input chessboard image you want to break up.

# piece_data = An array of tuples of the form (piece name, square location) where piece name and square location
#              are specified in algebraic notation. For instance, a black knight at d3 is represented as the tuple ("n", "d3").
#              Pieces may be specified in any order. Set this as an empty array if you don't want labels (such as using the AI in practice).

# corners = An array of the corners of the chessboard as (x, y) points on the image. Corners may be specified in any order.

# white_view = True if the image of the board is from white's perspective. False if from black's perspective.

# inner_grid = True if "corners" is the set of corners of the 6x6 inner chessboard, False if "corners" is the set of corners
#              of the entire 8x8 chessboard.

# cw, ch = Width and height respectively for each output cropped image.

# gather_piece_data = True if you want ONLY the piece images & labels, False if you want everything with labels occupied/not occupied.
#                     True to gather data for the piece classifer, False otherwise.

#=========#
# OUTPUTS #
#=========#

# images = The set of ALL 64 cropped tiles (in the order top-left to bottom-right)

# piece_images = The set of ONLY the tiles with pieces in them (in the order top-left to bottom-right)

# piece_labels = The set of labels (one-hot encoding) for the piece images (in the order top-left to bottom-right)

# empty_labels = The set of empty/occupied labels (one-hot encoding) for all images (in the order top-left to bottom-right)

def board_localization(image, piece_data, corners, white_view, inner_grid, cw, ch, gather_piece_data):
    # Identify the corners of the image (since corners are not necessarily specified in order of TL -> BR).
    # Do this by choosing the point with the minimum distance to each corner of the image.
    # For instance, TL corner is the one closest to (0, 0)
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
    # Points of this rectangle are the top left corner (x, y), averaging the x-coordinates of the top-right and bottom right corners,
    # averaging the y-coordinates of the bottom-left and bottom-right corners.
    # So the rectangle we wish to map to is TL --> (x, y), TR --> (avgX, y), BL --> (x, avgY), BR --> (avgX, avgY)
    x, y = top_left
    source_points = np.asarray([top_left, top_right, bottom_left, bottom_right], dtype = np.float32)
    avgX, avgY = round((top_right[0] + bottom_right[0])/2), round((bottom_right[1] + bottom_left[1])/2)
    dest_points = np.asarray([top_left, (avgX, y), (x, avgY), (avgX, avgY)], dtype = np.float32)
    
    # Use OpenCV to do the hard part for me and change perspective so that source points get mapped to destination points.
    A = cv.getPerspectiveTransform(source_points, dest_points)
    warped = cv.warpPerspective(image, A, (width, height))
    
    # Break into tiles
    # The warped perspective transform A does the following:
    # Given point (x, y), it gets mapped to (x', y') by
    # t * [x', y', 1].T = A * [x, y, 1].T
    # v.T indicates transpose of vector v.
    x_prime, y_prime, t = np.dot(A, np.asarray([top_left[0], top_left[1], 1]))
    warped_top_left = (round(x_prime/t), round(y_prime/t))
    x_prime, y_prime, t = np.dot(A, np.asarray([bottom_right[0], bottom_right[1], 1]))
    warped_bottom_right = (round(x_prime/t), round(y_prime/t))

    # Pad tiles to include one tile to the left and right and two tiles above the chessboard.
    tiles = np.zeros((10, 10, 4))  # Tiles are represented by the top-left and bottom-right points.
    dx, dy = abs(warped_bottom_right[0] - warped_top_left[0]), abs(warped_bottom_right[1] - warped_top_left[1])
    # Tile side lengths in pixels (not perfect squares)
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
            # (x, y) is the top-left corner, (next_x, next_y) is the bottom-right
            tiles[i][j] = (x, y, next_x, next_y)
    tiles = np.asarray(tiles).reshape(10, 10, 4)

    # Crop warped image
    images, piece_images, piece_labels, empty_labels = [], [], [], []
    
    # Mappings to help convert labels to one-hot vectors
    square_to_piece = {piece[1]: piece[0] for piece in piece_data}
    col_letters = 'abcdefgh'
    labels_to_int = {'P': 1, 'R': 2, 'N': 3, 'B': 4, 'Q': 5, 'K': 6, 'p': 7, 'r': 8, 'n': 9, 'b': 10, 'q': 11, 'k': 12}

    # Go through each tile (in the warped image) and crop it.
    for piece_i in range(8):
        for piece_j in range(8):
            # Find the label
            # Input is in algebraic notation and we need to adjust it depending on if the view is from black or white's perspective.
            square = col_letters[piece_j] + str(8-piece_i) if white_view else col_letters[7-piece_j] + str(piece_i + 1)
            if (square in square_to_piece): label = labels_to_int[square_to_piece[square]]
            else: label = 0 # Empty tile

            # Convert labels to one-hot encodings
            if (label != 0 and gather_piece_data):
                one_hot = np.zeros(12)
                one_hot[label - 1] = 1
                piece_labels.append(one_hot)
            is_empty = np.zeros(2)
            is_empty[min(label, 1)] = 1
            empty_labels.append(is_empty)

            # p1w, p2w, p3w, and p4w are the top-left, top-right, bottom-left, bottom-right points in the warped image
            # of the region we wish to crop.
            # p1w = (X0, Y0) = top-left point two units up (and possibly one unit to the left)
            # p2w = (X1, Y0) = top-right point two units up (and possibly one unit to the right)
            # p3w = (X0, Y1) = bottom-left point
            # p4w = (X1, Y1) = bottom-right point

            # Linear interpolation of X coordinates so that the crop stretches as we get closer to the edge.
            # We can linearly move from point P to point Q using a parametric equation.
            # f(alpha) = (1-alpha) * P + alpha * Q means that f(0) = P and f(1) = Q. Any 0 < alpha < 1 will lie between P and Q
    
            # So, for smoothly varying X coordinates, alpha = 0 when near the center and alpha = 1 when near the edge.
            # P = the x-coordinate of the top-left point two tiles up
            # Q = the x-coordinate of the top-left point two tiles up and one unit to either the left/right (depends which side of the board we are on).
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
           
           # Crop the image to a width and height specified by cw and ch
            crop = cv.resize(warped[Y0:Y1, min(X0,X1):max(X0,X1)], (cw, ch))
            if (flip): crop = cv.flip(crop, 1)
            if (not gather_piece_data): images.append(crop)
            if (label != 0 and gather_piece_data): piece_images.append(crop)
    
    # Free memory
    del tiles, warped

    return images, piece_images, piece_labels, empty_labels

def train_and_view(train: bool = True, use_oc: bool = True, num_epochs_piece: int = 50, num_epochs_occupancy: int = 3, occupancy_layer_amount: int = 3, piece_layer_amount: int = 3, learning_rate = 1e-4, filters = [16, 32, 64], batch_size=128):


    ###################
    # HYPERPARAMETERS #
    ###################
    crop_width, crop_height = 100, 100
    
    # Helper function for collecting the data we need for cropping the images
    def gather_data(imname, filename, cw, ch, gather_piece_data):
        im = cv.imread(filename + imname + ".png")
        metadata = json.load(open(filename + imname + ".json"))
        pieces = [(p['piece'], p['square'], p['box']) for p in metadata['pieces']]
        corners = metadata['corners']
        white_view = metadata['white_turn']
        # Board localization
        return board_localization(im, pieces, corners, white_view, False, cw, ch, gather_piece_data)
    

    '''
    #=================================================#
    # SET THIS IF YOU WANT TO TRAIN OR TEST THE MODEL #
    #=================================================#
    train = False # Choose whether to train or test.
    use_oc = False # Training/testing occupancy classifier = true, otherwise train/test piece classifier
    '''
    
    if (train):
        #=================#
        # DATA COLLECTION #
        #=================#
        start_time = time.time()

        # Gather and read the training and validation datasets.
        
        # For small tests of the AI, let's not load the whole dataset.
        use_up_to = 1000  # Set as None if you want to use the whole thing.

        train_files = listdir("Data/train")
        train_images, train_piece_images, train_piece_labels, train_empty_labels = [], [], [], []
        
        # Training dataset
        print("Loading training dataset...")
        for imname in (train_files[:use_up_to] if use_up_to != None else train_files):
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, piece_images, one_hot_labels, empty_labels = gather_data(x, "Data/train/", crop_width, crop_height, not use_oc)
            if (use_oc):
                train_images += images
                train_empty_labels += empty_labels
            else:
                train_piece_images += piece_images
                train_piece_labels += one_hot_labels
        train_images = np.asarray(train_images, dtype = np.float32)
        train_piece_images = np.asarray(train_piece_images, dtype = np.float32)
        train_piece_labels = np.asarray(train_piece_labels, dtype = np.float32)
        train_empty_labels = np.asarray(train_empty_labels, dtype = np.float32)
        
        valid_files = listdir("Data/val")
        valid_images, valid_piece_images, valid_piece_labels, valid_empty_labels = [], [], [], []
        
        # Validation dataset
        print("Loading validation dataset...")
        for imname in (valid_files[:use_up_to] if use_up_to != None else valid_files):
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, piece_images, one_hot_labels, empty_labels = gather_data(x, "Data/val/", crop_width, crop_height, not use_oc)
            if (use_oc):
                valid_images += images
                valid_empty_labels += empty_labels
            else:
                valid_piece_images += piece_images
                valid_piece_labels += one_hot_labels
        valid_images = np.asarray(valid_images, dtype = np.float32)
        valid_piece_images = np.asarray(valid_piece_images, dtype = np.float32)
        valid_piece_labels = np.asarray(valid_piece_labels, dtype = np.float32)
        valid_empty_labels = np.asarray(valid_empty_labels, dtype = np.float32)

        print("Datasets loaded in %.3f seconds." % (time.time() - start_time))

        # NOTE: ResNet and Inception were having problems so I'm using a simple CNN as a classifier for now just to see if I can get things
        # working.

    #==========================#
    # Occupancy Classification #
    #==========================#

    if (train and use_oc):
        # Model parameters
        #batch_size = 128
        #learning_rate = 1e-4

        # Instantiate model
        model = keras.Sequential()
        model.add(keras.Input(shape = (crop_height, crop_width, 3), name = "oc-input"))

        # Convolutional layers

        #occupancy_layer_amount = 3
        filter_index = 0
        #Multipying by 2 for each layer.

        for i in range(1, occupancy_layer_amount):
            print(i)
            model.add(keras.layers.Conv2D(filters = filters[filter_index], kernel_size = (5, 5), strides = (1, 1), name = f"oc-conv2d-{i}"))
            model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = f"oc-maxpool-{i}"))
            filter_index += 1


        '''
        model.add(keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), strides = (1, 1), name = "oc-conv2d-1"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "oc-maxpool-1"))

        model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1), name = "oc-conv2d-2"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "oc-maxpool-2"))

        model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), name = "oc-conv2d-3"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "oc-maxpool-3"))
        '''

        model.add(keras.layers.Flatten())

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(1024, name = "oc-dense-1"))
        model.add(keras.layers.Dropout(0.5, name = "oc-dropout-1"))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(256, name = "oc-dense-2"))
        model.add(keras.layers.Dropout(0.5, name = "oc-dropout-2"))

        model.add(keras.layers.Dense(2, activation = "softmax", name = "oc-predictions"))

        # # Make Base Model
        # base_model = ResNet101(weights = 'imagenet', input_shape = (crop_height, crop_width, 3), include_top = False, name = "oc-resnet101")
        # # Freeze base_model
        # base_model.trainable = True
        # inputs = keras.Input(shape = (crop_height, crop_width, 3))

        # # Calculate Outputs
        # x = base_model(inputs, training = True)
        # x = keras.layers.GlobalAveragePooling2D(name = "oc-pooling")(base_model.output)
        # x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(1024, name = "oc-dense")(x)
        # x = keras.layers.Dense(256, name = "oc-dense2")(x)
        # output = keras.layers.Dense(2, activation = "softmax", name = "oc-predictions")(x)

        # # Make Model
        # model = keras.Model(inputs = inputs, outputs = output)

        # Compile, Train, And Save Model
        model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                        loss = keras.losses.CategoricalCrossentropy(),
                        metrics = ['accuracy'])

        #num_epochs_occupany = 3
        model.fit(train_images, train_empty_labels,
                  validation_data = (valid_images, valid_empty_labels), epochs = num_epochs_occupancy, batch_size = batch_size)
        model.save("occupancy_classifier.keras", overwrite = True)
            
    #======================#
    # Piece Classification #
    #======================#

    if (train and not use_oc):  
        # Model parameters
        #batch_size = 128
        #learning_rate = 1e-4

        # Instantiate model
        model = keras.Sequential()
        model.add(keras.Input(shape = (crop_height, crop_width, 3), name = "pc-input"))

        # Convolutional layers

        #piece_layer_amount = 3
        filter_index = 0
        #Multipying by 2 for each layer.

        for i in range(1, piece_layer_amount):
            print(i)
            model.add(keras.layers.Conv2D(filters = filters[filter_index], kernel_size = (5, 5), strides = (1, 1), name = f"pc-conv2d-{i}"))
            model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = f"pc-maxpool-{i}"))
            filter_index += 1

        '''
        model.add(keras.layers.Conv2D(filters = 16, kernel_size = (5, 5), strides = (1, 1), name = "pc-conv2d-1"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "pc-maxpool-1"))

        model.add(keras.layers.Conv2D(filters = 32, kernel_size = (5, 5), strides = (1, 1), name = "pc-conv2d-2"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "pc-maxpool-2"))

        model.add(keras.layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = (1, 1), name = "pc-conv2d-3"))
        model.add(keras.layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = "pc-maxpool-3"))
        '''

        model.add(keras.layers.Flatten())

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(2048, name = "pc-dense-1"))
        model.add(keras.layers.Dropout(0.5, name = "pc-dropout-1"))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(1024, name = "pc-dense-2"))
        model.add(keras.layers.Dropout(0.5, name = "pc-dropout-2"))

        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dense(256, name = "pc-dense-3"))
        model.add(keras.layers.Dropout(0.5, name = "pc-dropout-3"))


        model.add(keras.layers.Dense(12, activation = "softmax", name = "pc-predictions"))

        # # Make Base Model
        # base_model = InceptionV3(weights = 'imagenet', input_shape = (crop_height, crop_width, 3), include_top = False, name = "pc-inception")
        # # Freeze base_model
        # base_model.trainable = True
        # inputs = keras.Input(shape = (crop_height, crop_width, 3))

        # # Calculate Outputs
        # x = base_model(inputs, training = True)
        # x = keras.layers.GlobalAveragePooling2D(name = "pc-pooling")(base_model.output)
        # x = keras.layers.Flatten()(x)
        # x = keras.layers.Dense(1024, name = "pc-dense")(x)
        # x = keras.layers.Dense(256, name = "pc-dense2")(x)
        # output = keras.layers.Dense(12, activation = "softmax", name = "pc-predictions")(x)

        # # Make Model
        # model = keras.Model(inputs = inputs, outputs = output)

        # Compile, Train, And Save Model
        model.compile(optimizer = keras.optimizers.Adam(learning_rate),
                      loss = keras.losses.CategoricalCrossentropy(),
                      metrics = ['accuracy'])
        
        #num_epochs_piece = 50
        model.fit(train_piece_images, train_piece_labels,
                  validation_data = (valid_piece_images, valid_piece_labels), epochs = num_epochs_piece, batch_size = batch_size)
        model.save("piece_classifier.keras", overwrite = True)
        
    # Testing
    if (not train):
        start_time = time.time()

        # For small tests of the AI, let's not load the whole dataset.
        use_up_to = 1000  # Set as None if you want to use the whole thing.
        
        test_files = listdir("Data/test")
        
        # Testing dataset
        print("Loading testing dataset...")
        test_images, test_empty_labels, test_piece_images, test_piece_labels = [], [], [], []
        for imname in (test_files[:use_up_to] if use_up_to != None else test_files):
            x = imname.split('.')
            if (x[1] == "json"): continue
            x = x[0]
            images, piece_images, one_hot_labels, empty_labels = gather_data(x, "Data/test/", crop_width, crop_height, not use_oc)
            if (use_oc):
                test_images += images
                test_empty_labels += empty_labels
            else:
                test_piece_images += piece_images
                test_piece_labels += one_hot_labels
        
        print("Datasets loaded in %.3f seconds." % (time.time() - start_time))
    
    # Test occupancy classifier
    if (not train and use_oc):
        model = keras.models.load_model("occupancy_classifier.keras")
        model.evaluate(np.asarray(test_images), np.asarray(test_empty_labels))
        
        # Choose images from the dataset randomly and predict.
        print("Randomly choosing test images to display.")
        score = 0
        test_amount = 500
        threshold = 0.8
        str_labels = ["Empty", "Not Empty"]
        for _ in range(test_amount):
            i = np.random.randint(0, len(test_images))
            cv.imshow("Test Image", np.asarray(test_images[i]))
            
            # Input img ---> img shape (100,100,3)
            # img = np.expand_dims(img, 0) ---> img shape now (1,100,100,3)
            # pred = model(img) ---> pred shape (1, 2)
            # pred = np.reshape(pred, -1) ---> pred shape now (2)
            # pred[0] = probability of 0th class, pred[1] = probability of 1st class
            # 0 class = empty, 1 class = occupied
            img = np.expand_dims(test_images[i], 0)
            pred = np.reshape(model(img), -1)
            print("Probability Distribution: " + str(pred))
            
            # Choose randomly or highest value.
            if (max(pred) >= threshold): label = np.argmax(pred)
            else: label = np.random.choice(2, 1, p = pred)[0]
            
            actual = np.argmax(test_empty_labels[i])
            print("Predicted: " + str_labels[label] + ", Actual: " + str_labels[actual])
            if (label == actual): score += 1
            #cv.waitKey()
        print("Testing had a score of %d/%d or %.3f accuracy!" % (score, test_amount, score/test_amount))
        return [score, test_amount, score/test_amount]
        
    # Test piece classifier
    if (not train and not use_oc):
        model = keras.models.load_model("piece_classifier.keras")
        model.evaluate(np.asarray(test_piece_images, np.float32), np.asarray(test_piece_labels, np.float32))
        
        # Choose images from the dataset randomly and predict.
        print("Randomly choosing test images to display.")
        score = 0
        test_amount = 500
        threshold = 0.8
        str_labels = "PRNBQKprnbqk"
        for _ in range(test_amount):
            # Choose image randomly
            i = np.random.randint(0, len(test_piece_images))
            cv.imshow("Test Image", np.asarray(test_piece_images[i]))
            
            # Input img ---> img shape (100,100,3)
            # img = np.expand_dims(img, 0) ---> img shape now (1,100,100,3)
            # pred = model(img) ---> pred shape (1, 12)
            # pred = np.reshape(pred, -1) ---> pred shape now (12)
            # pred[i] = probability of ith class
            # Classes are in order "PRNBQKprnbqk"
            img = np.expand_dims(test_piece_images[i], 0)
            pred = np.reshape(model(img), -1)
            print("Probability Distribution: " + str(pred))

            # Choose randomly or highest value.
            if (max(pred) >= threshold): label = np.argmax(pred)
            else: label = np.random.choice(12, 1, p = pred)[0]

            actual = np.argmax(test_piece_labels[i])
            print("Predicted: " + str_labels[label] + ", Actual: " + str_labels[actual])
            if (label == actual): score += 1
            #cv.waitKey()
        print("Testing had a score of %d/%d or %.3f accuracy!" % (score, test_amount, score/test_amount))
        return [score, test_amount, score/test_amount]




if __name__ == "__main__":

    path_occupancy = "results_occupancy.json"
    path_piece = "results_piece.json"
    
    occupancy_data = []
    piece_data = []


    train = False # Choose whether to train or test.
    use_oc = False # Training/testing occupancy classifier = true, otherwise train/test piece classifier


    num_epochs_piece = 50

    num_epochs_occupancy = 3


    piece_layer_amount = 3

    occupancy_layer_amount = 3


    #Starting with occupancy

    train = True
    use_oc = True


    learning_rate = 1e-4

    filters = [16, 32, 64]

    batch_size = 128



    use_oc = False
    index = 1
    for x in range(3, 7): #First param inclusive, second exclusive - Range of layers for this param
        for y in range(40, 71, 2): # Range of epochs

            print(f"Performing Version: {index}/{64}")

            #current_data = {}
            piece_layer_amount = x
            num_epochs_piece = y

            train = True
            train_and_view(train=train, use_oc=use_oc, num_epochs_piece=num_epochs_piece, piece_layer_amount=piece_layer_amount, learning_rate=learning_rate, filters=filters)

            train = False
            train_result = train_and_view(train=train, use_oc=use_oc, num_epochs_piece=num_epochs_piece, piece_layer_amount=piece_layer_amount, learning_rate=learning_rate, filters=filters)

            final_train_result = "Testing had a score of %d/%d or %.3f accuracy!" % (train_result[0], train_result[1], train_result[2])

            

            current_data = {
                "piece_layer_amount" : piece_layer_amount,
                "num_epochs_piece" : num_epochs_piece,
                "filters" : filters,
                "batch_size": batch_size,
                "dropout_param" : 0.5,
                "learning_rate" : learning_rate,
                "score" : train_result[0],
                "test_amount" : train_result[1],
                "accuracy" : "%.3f" % (train_result[0]/train_result[1]),
                "train_result" : final_train_result
            }

            piece_data.append(current_data)

            with open(path_piece, 'w') as json_file:
                json.dump(piece_data, json_file, indent=4)

            

            index += 1




    use_oc = True
    index = 1
    for x in range(3, 7): #First param inclusive, second exclusive - Range of layers for this param
        for y in range(3, 14): # Range of epochs

            print(f"Performing Version: {index}/{40}")

            #current_data = {}
            occupancy_layer_amount = x
            num_epochs_occupancy = y

            train = True
            train_and_view(train=train, use_oc=use_oc, num_epochs_occupancy=num_epochs_occupancy, occupancy_layer_amount=occupancy_layer_amount, learning_rate=learning_rate, filters=filters)

            train = False
            train_result = train_and_view(train=train, use_oc=use_oc, num_epochs_occupancy=num_epochs_occupancy, occupancy_layer_amount=occupancy_layer_amount, learning_rate=learning_rate, filters=filters)

            #print("Testing had a score of %d/%d or %.3f accuracy!" % (train_result[0], train_result[1], train_result[2]))


            final_train_result = "Testing had a score of %d/%d or %.3f accuracy!" % (train_result[0], train_result[1], train_result[2])
            

            current_data = {
                "occupancy_layer_amount" : occupancy_layer_amount,
                "num_epochs_occupancy" : num_epochs_occupancy,
                "filters" : filters,
                "batch_size" : batch_size,
                "dropout_param" : 0.5,
                "learning_rate" : learning_rate,
                "score" : train_result[0],
                "test_amount" : train_result[1],
                "accuracy" : "%.3f" % (train_result[0]/train_result[1]),
                "train_result" : final_train_result
            }

            occupancy_data.append(current_data)

            with open(path_occupancy, 'w') as json_file:
                json.dump(occupancy_data, json_file, indent=4)

            

            index += 1


    #train_and_view(train=train, use_oc=use_oc, num_epochs_piece=50, num_epochs_occupancy=3)


    #with open(path_occupancy, 'w') as json_file:
    #    json.dump(occupancy_data, json_file, indent=4)

