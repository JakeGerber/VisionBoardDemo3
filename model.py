from os import listdir
from os.path import isfile, join

import math
import numpy as np
import scipy.cluster.hierarchy as cluster
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

# Returns the intersection point between two lines (each defined by the pair of points they join)
def intersection_point(l1, l2):
    # Convert to y = mx + b for both lines.
    x11, x12 = l1[0][0], l1[1][0]
    y11, y12 = l1[0][1], l1[1][1]
    
    # Check for vertical line x=x11
    if (x12 == x11):  m1, b1 = None, x11
    else:
        m1 = (y12-y11)/(x12-x11)
        # y = m(x-x0) + y0 --> y = mx + (y0 - m*x0)
        b1 = y11 - m1 * x11
    
    x21, x22 = l2[0][0], l2[1][0]
    y21, y22 = l2[0][1], l2[1][1]
    # Check for vertical line x=x21
    if (x22 == x21): m2, b2 = None, x21
    else:
        m2 = (y22-y21)/(x22-x21)
        b2 = y21 - m2 * x21

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

if __name__ == "__main__":
    # Gather and read the image.
    train_files = listdir("Data/train")
    while True:
        imname = str(int(np.random.randint(0, 4886)))
        imname += (4-len(imname))*"0"+".png"
        if (imname in train_files): break
    im = cv.imread("Data/train/" + imname)
    height, width, _ = im.shape
    
    # Canny edge detector followed by a Hough Transformation to roughly find all of the lines in the image.
    edges = cv.Canny(im, 200, 400)
    # Probabilistic Hough Transform
    hough_lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, None, 50, 10)
    
    # hough_lines = cv.HoughLines(edges, 1, np.pi / 180, 50, None, 50, 10)
    # print(str(len(hough_lines)) + " detected.")
    # hough_lines = [(l[0][0], l[0][1]) for l in hough_lines] # (r, theta) pairs for each polar line

    # def angle_metric(l1, l2): return min(l1[1], l2[1])  # Define the metric for the clustering algorithm.
    # linkage_matrix = cluster.linkage(hough_lines, method = 'single', metric = angle_metric)
    # # https://stackoverflow.com/questions/21638130/tutorial-for-scipy-cluster-hierarchy
    # clusters = cluster.fcluster(linkage_matrix, 2, criterion = 'maxclust')
    # print(clusters)

    # Plot the lines on the original image
    lines = []
    for i in range(len(hough_lines)):
        l = hough_lines[i][0]
        # Line is given by (1 - t) * (l[0], l[1]) + t * (l[2], l[3]) for t = [-99, 99]
        # So extend line by doing t in [-99, 100]
        p1 = (100 * l[0] - 99 * l[2], 100 * l[1] - 99 * l[3])
        p2 = (-99 * l[0] + 100 * l[2], -99 * l[1] + 100 * l[3])
        lines.append([p1, p2])
        cv.line(im, p1, p2, (0,255,0), 3, cv.LINE_AA)
    intersection_points = []
    for i in range(len(lines)):
        l1 = lines[i]
        for j in range(i + 1, len(lines)):
            l2 = lines[j]
            inter_point = intersection_point(l1, l2)
            if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
                x, y = inter_point
                x, y = round(x), round(y)
                intersection_points.append((x, y))
                cv.circle(im, (x, y), 5, (0,0,255), -1)

    #cv.imshow("Canny Edge Detection", edges)
    cv.imshow("Chessboard with Detected Lines", im)
    cv.imwrite("test.png", im)
    cv.waitKey()
    
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