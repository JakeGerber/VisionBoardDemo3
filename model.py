from os import listdir
from os.path import isfile, join

import math
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt

import cv2 as cv
 
# import keras
# from keras.applications.inception_v3 import InceptionV3
# from keras import layers

# Model breakdown:
# Board Localization:
# - Canny Edge Detector to detect (most of) the lines.
# - DBSCAN clustering to group multiple lines that are close together with a single line.
# - Compute homography matrix and find the remaining lines.
# Occupancy Classifier:
# - CNN (100, 3, 3, 3) or ResNet
# Piece Recognition:
# - CNN (100, 3, 3, 3) or InceptionV3

# Returns the slope-intercept components for a given line
def slope_intercept(l):
    # Convert to y = mx + b for both lines.
    x1, x2 = l[0][0], l[1][0]
    y1, y2 = l[0][1], l[1][1]
    
    # Check for vertical line x=x11
    if (x2 == x1):  m, b = None, x1
    else:
        m = (y2-y1)/(x2-x1)
        # y = m(x-x0) + y0 --> y = mx + (y0 - m*x0)
        b = y1 - m * x1
    return m, b

# Returns the intersection point between two lines (each defined by the pair of points they join)
def intersection_point(l1, l2):
    m1, b1 = slope_intercept(l1)
    m2, b2 = slope_intercept(l2)

    # Compute intersection point
    if (m1 == m2): return None # Lines are parallel
    elif (m1 == None):
        x_inter = b1
        y_inter = m2 * x_inter + b2
    elif (m2 == None):
        x_inter = b2
        y_inter = m1 * x_inter + b1
    else:
        x_inter = (b1 - b2)/(m2 - m1)
        y_inter = m1 * x_inter + b1
    return (x_inter, y_inter)

# Given an image of a chessboard and its corresponding FEN string, split it into a collection of 64 tiles
# labeled by the pieces (or lack thereof) which occupy that tile.
# Returns a 64 element array of tuples corresponding to tiles to be passed into the piece classifier and the
# corresponding label for that tile.
def board_localization(image, fen):
    height, width, _ = image.shape
    # Canny edge detector followed by a Hough Transformation to roughly find all of the lines in the image.
    edges = cv.Canny(image, 200, 400)
    # Probabilistic Hough Transform
    hough_lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    # Plot the lines on the original image
    scale_factor = 20
    lines, angles = [], []
    for i in range(len(hough_lines)):
        l = hough_lines[i][0]
        # Line is given by (1 - t) * (l[0], l[1]) + t * (l[2], l[3]) for t = [-99, 99]
        # So extend line by doing t in [-scale_factor + 1, scale_factor]
        p1 = (scale_factor * l[0] - (scale_factor-1) * l[2], scale_factor * l[1] - (scale_factor-1) * l[3])
        p2 = (-(scale_factor-1) * l[0] + scale_factor * l[2], -(scale_factor-1) * l[1] + scale_factor * l[3])
        angle = math.atan2(l[3]-l[1], l[2]-l[0])
        lines.append([p1, p2])
        angles.append(angle)

    # Randomly sample two lines in order to find the horizontal and vertical axes
    axis_vec_1, axis_vec_2 = None, None
    dot_product_tolerance = 0.1
    while True:
        i = np.random.randint(0, len(lines))
        while True:
            j = np.random.randint(0, len(lines))
            if (i != j): break
        l1, l2 = lines[i], lines[j]
        # Slope vectors for each line
        vec1, vec2 = (l1[1][0] - l1[0][0], l1[1][1]-l1[0][1]), (l2[1][0] - l2[0][0], l2[1][1]-l2[0][1])
        mag1, mag2 = math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1]), math.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1])
        vec1, vec2 = (vec1[0]/mag1, vec1[1]/mag1), (vec2[0]/mag2, vec2[1]/mag2)
        # Dot product should be within tolerance of zero for lines to be orthogonal
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        if (abs(dot_product) < dot_product_tolerance):
            axis_vec_1, axis_vec_2 = vec1, vec2
            break
        
    line_group_1, line_group_2 = [], []
    for i in range(len(lines)):
        l = lines[i]
        vec = (l[1][0] - l[0][0], l[1][1]-l[0][1])
        mag = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
        vec = (vec[0]/mag, vec[1]/mag)
        dot_product = vec[0] * axis_vec_1[0] + vec[1] * axis_vec_1[1]
        if (1-abs(dot_product) < dot_product_tolerance): line_group_1.append(i)
        else:
            dot_product = vec[0] * axis_vec_2[0] + vec[1] * axis_vec_2[1]
            if (1-abs(dot_product) < dot_product_tolerance): line_group_2.append(i)

    for i in line_group_1:
        cv.line(im, lines[i][0], lines[i][1], (0, 255, 0), 3, cv.LINE_AA)
    for i in line_group_2:
        cv.line(im, lines[i][0], lines[i][1], (255, 0, 0), 3, cv.LINE_AA)
        
    # Compute the intersection points between the lines
    intersection_points = []
    for i in line_group_1:
        l1 = lines[i]
        for j in line_group_2:
            l2 = lines[j]
            inter_point = intersection_point(l1, l2)
            if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
                x, y = inter_point
                x, y = round(x), round(y)
                intersection_points.append((x, y))
                cv.circle(im, (x, y), 5, (0,0,255), -1)

    # cv.imshow("Canny Edge Detection", edges)
    cv.imshow("Chessboard with Detected Lines", im)
    cv.imwrite("test.png", im)
    cv.waitKey()

if __name__ == "__main__":
    # Gather and read the image.
    train_files = listdir("Data/train")
    while True:
        imname = str(int(np.random.randint(0, 4886)))
        imname += (4-len(imname))*"0"+".png"
        if (imname in train_files): break
    im = cv.imread("Data/train/" + imname)
    board_localization(im, None)
    
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

    #     # Compile, train, and save the dataset
    #     model.compile(optimizer = keras.optimizers.Adam(learning_rate),
    #                 loss = keras.losses.CategoricalCrossentropy(),
    #                 metrics = ['accuracy'])
    #     model.fit(train_data_x, train_data_y,
    #             validation_set = (valid_data_x, valid_data_y),
    #             batch_size = batch_size, validation_batch_size = batch_size)
    #     model.save(filepath, overwrite = True)

    # # Test the model on the test dataset.