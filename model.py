from os import listdir
import json

import math
import numpy as np
# import sklearn.cluster
# import matplotlib.pyplot as plt

import cv2 as cv

import tensorflow as tf
import keras
# from keras._tf_keras.keras.applications.inception_v3 import InceptionV3
from keras.applications import InceptionV3
# from keras.applications.InceptionV3 import InceptionV3
from keras import layers














# Model breakdown:
# Board Localization:
# - Canny Edge Detector to detect (most of) the lines.
# - DBSCAN clustering to group multiple lines that are close together with a single line.
# - Compute homography matrix and find the remaining lines.
# Occupancy Classifier:
# - CNN (100, 3, 3, 3) or ResNet
# Piece Recognition:
# - CNN (100, 3, 3, 3) or InceptionV3

# Given the bottom-right and top-left points of a rectangle (p1 and p2 respectively), compute the matrix
# mapping p1 to [s_x, 0] and p2 to [0, s_y]


















def compute_homography(p1, p2, s_x, s_y):
    H_inv = np.asarray([[p1[0]/s_x, p2[0]/s_y], [p1[1]/s_x, p2[1]/s_y]])
    return np.linalg.inv(H_inv)

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
def board_localization(image, piece_data):
    height, width, _ = image.shape
    # Canny edge detector followed by a Hough Transformation to roughly find all of the lines in the image.
    edges = cv.Canny(image, 250, 400, apertureSize = 3)
    # Probabilistic Hough Transform
    hough_lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, 80, 30, 10)

    # Plot the lines on the original image
    scale_factor = 25
    lines = []
    for i in range(len(hough_lines)):
        l = hough_lines[i][0]
        # Line is given by (1 - t) * (l[0], l[1]) + t * (l[2], l[3]) for t = [-99, 99]
        # So extend line by doing t in [-scale_factor + 1, scale_factor]
        p1 = (scale_factor * l[0] - (scale_factor-1) * l[2], scale_factor * l[1] - (scale_factor-1) * l[3])
        p2 = (-(scale_factor-1) * l[0] + scale_factor * l[2], -(scale_factor-1) * l[1] + scale_factor * l[3])
        lines.append([p1, p2])

    # Randomly sample two lines in order to find the horizontal and vertical axes
    axis1, axis2 = None, None
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
            axis1, axis2 = i, j
            axis_vec_1, axis_vec_2 = vec1, vec2
            break

    line_groups = [[], []]
    for i in range(len(lines)):
        l = lines[i]
        vec = (l[1][0] - l[0][0], l[1][1]-l[0][1])
        mag = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
        vec = (vec[0]/mag, vec[1]/mag)
        dot_product = vec[0] * axis_vec_1[0] + vec[1] * axis_vec_1[1]
        if (1-abs(dot_product) < dot_product_tolerance): line_groups[0].append(i)
        else:
            dot_product = vec[0] * axis_vec_2[0] + vec[1] * axis_vec_2[1]
            if (1-abs(dot_product) < dot_product_tolerance): line_groups[1].append(i)

    # Compute the intersection points between the lines and use DBSCAN to reduce the number of lines in each group
    new_line_groups = [[], []]
    for k in [0, 1]:
        point_to_line = {}
        intersection_points = []
        l2 = lines[axis2 if k == 0 else axis1]
        for i in line_groups[k]:
            l1 = lines[i]
            inter_point = intersection_point(l1, l2)
            if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
                x, y = inter_point
                x, y = round(x), round(y)
                point_to_line[(x, y)] = i
                intersection_points.append((x, y))
        intersection_points = np.asarray(intersection_points)
        clustering = sklearn.cluster.DBSCAN(eps = 10, min_samples = 2, metric = "euclidean").fit(intersection_points)
        seen = set()
        for i in range(len(intersection_points)):
            label = clustering.labels_[i]
            if (label not in seen):
                new_line_groups[k].append(point_to_line[tuple(intersection_points[i])])
                seen.add(label)
    line_groups = new_line_groups
    del new_line_groups, clustering, intersection_points, point_to_line

    for k in [0, 1]:
        for i in line_groups[k]:
            cv.line(im, lines[i][0], lines[i][1], (0, 255, 0) if k == 0 else (255, 0, 0), 2, cv.LINE_AA)

    # Find all intersection points between line groups
    intersection_points = []
    for i in line_groups[0]:
        l1 = lines[i]
        for j in line_groups[1]:
            l2 = lines[j]
            inter_point = intersection_point(l1, l2)
            if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
                x, y = inter_point
                x, y = round(x), round(y)
                intersection_points.append((x, y))

    # Compute the homography matrix and fix current lines / find remaining lines
    sample_1, sample_2 = [], []
    while len(sample_1) != 2:
        i = line_groups[0][np.random.randint(0, len(line_groups[0]))]
        if (i not in sample_1): sample_1.append(i)
    while len(sample_2) != 2:
        i = line_groups[1][np.random.randint(0, len(line_groups[1]))]
        if (i not in sample_2): sample_2.append(i)
    point_sample = []
    for i in sample_1:
        l1 = lines[i]
        for j in sample_2:
            l2 = lines[j]
            inter_point = intersection_point(l1, l2)
            if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
                x, y = inter_point
                x, y = round(x), round(y)
                point_sample.append((x, y))
                cv.circle(im, (x, y), 3, (0, 0, 255), 3, cv.LINE_AA)
    p1, p2, p3, p4 = point_sample
    min_x, max_x = min([p1[0], p2[0], p3[0], p4[0]]), max([p1[0], p2[0], p3[0], p4[0]])
    min_y, max_y = min([p1[1], p2[1], p3[1], p4[1]]), max([p1[1], p2[1], p3[1], p4[1]])
    s_x, s_y = 1, 1
    H = compute_homography((max_x, min_y), (min_x, max_y), s_x, s_y)

    # for piece in piece_data:
    #     cv.rectangle(im, (piece[2][0], piece[2][1]), (piece[2][0] + piece[2][2], piece[2][1] + piece[2][3]), (0, 0, 255), 2, cv.LINE_AA)
    cv.imshow("Chessboard with Detected Lines", im)
    cv.imwrite("test.png", im)
    cv.waitKey()

if __name__ == "__main__":
    # Gather and read the image.
    train_files = listdir("Data/train")
    while True:
        imname = str(int(np.random.randint(0, 4886)))
        imname += (4-len(imname))*"0"
        if (imname+".png" in train_files and imname+".json" in train_files): break
    im = cv.imread("Data/train/" + imname + ".png")
    metadata = json.load(open("Data/train/" + imname + ".json"))
    fen = metadata['fen']
    pieces = [(p['piece'], p['square'], p['box']) for p in metadata['pieces']]
    board_localization(im, pieces)
    

























    




    









