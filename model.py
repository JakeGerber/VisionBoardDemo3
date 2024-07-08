from os import listdir
import json
import time

import math
import numpy as np

import cv2 as cv
# import keras
# from keras.applications.inception_v3 import InceptionV3
# from keras import layers

# # Returns the slope-intercept components for a given line
# def slope_intercept(l):
#     # Convert to y = mx + b for both lines.
#     x1, x2 = l[0][0], l[1][0]
#     y1, y2 = l[0][1], l[1][1]
    
#     # Check for vertical line x=x11
#     if (x2 == x1):  m, b = None, x1
#     else:
#         m = (y2-y1)/(x2-x1)
#         # y = m(x-x0) + y0 --> y = mx + (y0 - m*x0)
#         b = y1 - m * x1
#     return m, b

# # Returns the intersection point between two lines (each defined by the pair of points they join)
# def intersection_point(l1, l2):
#     m1, b1 = slope_intercept(l1)
#     m2, b2 = slope_intercept(l2)

#     # Compute intersection point
#     if (m1 == m2): return None # Lines are parallel
#     elif (m1 == None):
#         x_inter = b1
#         y_inter = m2 * x_inter + b2
#     elif (m2 == None):
#         x_inter = b2
#         y_inter = m1 * x_inter + b1
#     else:
#         x_inter = (b1 - b2)/(m2 - m1)
#         y_inter = m1 * x_inter + b1
#     return (x_inter, y_inter)

# Given an image of a chessboard and its metadata, split it into a collection of 64 tiles
# labeled by the pieces (or lack thereof) which occupy that tile.
# Returns a 64 element array of tile images and their labels.
def board_localization(image, piece_data, corners, white_view):
    # # Canny edge detector followed by a Hough Transformation to roughly find all of the lines in the image.
    # edges = cv.Canny(image, 100, 150, apertureSize = 3)
    # # Probabilistic Hough Transform
    # hough_lines = cv.HoughLinesP(edges, 1, np.pi / 180, 50, 80, 30, 10)

    # # Extend each line.
    # scale_factor = 25
    # lines = []
    # for i in range(len(hough_lines)):
    #     l = hough_lines[i][0]
    #     # Line is given by (1 - t) * (l[0], l[1]) + t * (l[2], l[3]) for t = [-99, 99]
    #     # So extend line by doing t in [-scale_factor + 1, scale_factor]
    #     p1 = (scale_factor * l[0] - (scale_factor-1) * l[2], scale_factor * l[1] - (scale_factor-1) * l[3])
    #     p2 = (-(scale_factor-1) * l[0] + scale_factor * l[2], -(scale_factor-1) * l[1] + scale_factor * l[3])
    #     lines.append([p1, p2])

    # # Randomly sample two lines in order to find the horizontal and vertical axes
    # axis1, axis2 = None, None
    # axis_vec_1, axis_vec_2 = None, None
    # dot_product_tolerance = 0.1
    # while True:
    #     i = np.random.randint(0, len(lines))
    #     while True:
    #         j = np.random.randint(0, len(lines))
    #         if (i != j): break
    #     l1, l2 = lines[i], lines[j]
    #     # Slope vectors for each line
    #     vec1, vec2 = (l1[1][0] - l1[0][0], l1[1][1]-l1[0][1]), (l2[1][0] - l2[0][0], l2[1][1]-l2[0][1])
    #     mag1, mag2 = math.sqrt(vec1[0]*vec1[0] + vec1[1]*vec1[1]), math.sqrt(vec2[0]*vec2[0] + vec2[1]*vec2[1])
    #     vec1, vec2 = (vec1[0]/mag1, vec1[1]/mag1), (vec2[0]/mag2, vec2[1]/mag2)
    #     # Dot product should be within tolerance of zero for lines to be orthogonal
    #     dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
    #     if (abs(dot_product) < dot_product_tolerance):
    #         axis1, axis2 = i, j
    #         axis_vec_1, axis_vec_2 = vec1, vec2
    #         break

    # # Group the lines based on the axis representative.
    # line_groups = [[], []]
    # for i in range(len(lines)):
    #     l = lines[i]
    #     vec = (l[1][0] - l[0][0], l[1][1]-l[0][1])
    #     mag = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
    #     vec = (vec[0]/mag, vec[1]/mag)
    #     dot_product = vec[0] * axis_vec_1[0] + vec[1] * axis_vec_1[1]
    #     if (1-abs(dot_product) < dot_product_tolerance): line_groups[0].append(i)
    #     else:
    #         dot_product = vec[0] * axis_vec_2[0] + vec[1] * axis_vec_2[1]
    #         if (1-abs(dot_product) < dot_product_tolerance): line_groups[1].append(i)

    # # Compute the intersection points between the lines and use DBSCAN to reduce the number of lines in each group
    # new_line_groups = [[], []]
    # for k in [0, 1]:
    #     point_to_line = {}
    #     intersection_points = []
    #     l2 = lines[axis2 if k == 0 else axis1]
    #     for i in line_groups[k]:
    #         l1 = lines[i]
    #         inter_point = intersection_point(l1, l2)
    #         if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
    #             x, y = inter_point
    #             x, y = round(x), round(y)
    #             point_to_line[(x, y)] = i
    #             intersection_points.append((x, y))
    #     intersection_points = np.asarray(intersection_points)
    #     clustering = sklearn.cluster.DBSCAN(eps = 10, min_samples = 2, metric = "euclidean").fit(intersection_points)
    #     seen = set()
    #     for i in range(len(intersection_points)):
    #         label = clustering.labels_[i]
    #         if (label not in seen):
    #             new_line_groups[k].append(point_to_line[tuple(intersection_points[i])])
    #             seen.add(label)
    # line_groups = new_line_groups
    # del new_line_groups, clustering, intersection_points, point_to_line

    # for k in [0, 1]:
    #     for i in line_groups[k]:
    #         cv.line(im, lines[i][0], lines[i][1], (0, 255, 0) if k == 0 else (255, 0, 0), 2, cv.LINE_AA)

    # # Find all intersection points between line groups
    # intersection_points = []
    # for i in line_groups[0]:
    #     l1 = lines[i]
    #     for j in line_groups[1]:
    #         l2 = lines[j]
    #         inter_point = intersection_point(l1, l2)
    #         if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
    #             x, y = inter_point
    #             x, y = round(x), round(y)
    #             intersection_points.append((x, y))

    # # Compute the homography matrix and fix current lines / find remaining lines
    # sample_1, sample_2 = [], []
    # while len(sample_1) != 2:
    #     i = line_groups[0][np.random.randint(0, len(line_groups[0]))]
    #     if (i not in sample_1): sample_1.append(i)
    # while len(sample_2) != 2:
    #     i = line_groups[1][np.random.randint(0, len(line_groups[1]))]
    #     if (i not in sample_2): sample_2.append(i)
    # point_sample = []
    # for i in sample_1:
    #     l1 = lines[i]
    #     for j in sample_2:
    #         l2 = lines[j]
    #         inter_point = intersection_point(l1, l2)
    #         if (inter_point != None and 0 <= inter_point[0] < width and 0 <= inter_point[1] < height):
    #             x, y = inter_point
    #             x, y = round(x), round(y)
    #             point_sample.append((x, y))
    #             cv.circle(im, (x, y), 3, (0, 0, 255), 3, cv.LINE_AA)
    #for p in corners: cv.circle(im, p, 3, (0, 0, 255), 3, cv.LINE_AA)
    
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
    warped = cv.warpPerspective(im, A, (width, height))
    
    # Break into tiles
    x_prime, y_prime, t = np.dot(A, np.asarray([top_left[0], top_left[1], 1]))
    warped_top_left = (round(x_prime/t), round(y_prime/t))
    x_prime, y_prime, t = np.dot(A, np.asarray([bottom_right[0], bottom_right[1], 1]))
    warped_bottom_right = (round(x_prime/t), round(y_prime/t))
    # Pad tiles to include one tile to the left and right and two tiles above the chessboard.
    tiles = np.zeros((10, 10, 4))  # Tiles are represented by the top-left and bottom-right points.
    dx, dy = abs(warped_bottom_right[0] - warped_top_left[0]), abs(warped_bottom_right[1] - warped_top_left[1])
    sx, sy = dx/8, dy/8
    for i in range(-2, 8):
        y = warped_top_left[1] + sy * i
        next_y = warped_top_left[1] + sy * i + sy
        for j in range(-1, 9):
            x = warped_top_left[0] + sx * j
            next_x = warped_top_left[0] + sx * j + sx
            tiles[i][j] = (x, y, next_x, next_y)
    tiles = np.asarray(tiles).reshape(10, 10, 4)

    # Crop warped image
    crop_width, crop_height = 25, 50
    images, labels = [], []
    square_to_piece = {piece[1]: piece[0] for piece in piece_data}
    col_letters = 'abcdefgh'
    for piece_i in range(8):
        for piece_j in range(8):
            # Find the label
            square = col_letters[piece_j] + str(8-piece_i) if white_view else col_letters[7-piece_j] + str(piece_i + 1)
            if (square in square_to_piece): label = square_to_piece[square]
            else: label = 'E'
            labels.append(label)

            p1w, p2w, p3w, p4w = (tiles[piece_i-2][piece_j][0], tiles[piece_i-1][piece_j][1]), (tiles[piece_i-2][piece_j][2], tiles[piece_i-2][piece_j][1]), \
                            (tiles[piece_i][piece_j][0], tiles[piece_i][piece_j][3]), (tiles[piece_i][piece_j][2], tiles[piece_i][piece_j][3])

            min_xw, max_xw = int(min(p1w[0], p2w[0], p3w[0], p4w[0])), int(max(p1w[0], p2w[0], p3w[0], p4w[0]))
            min_yw, max_yw = int(min(p1w[1], p2w[1], p3w[1], p4w[1])), int(max(p1w[1], p2w[1], p3w[1], p4w[1]))
            crop = cv.resize(warped[min_yw:max_yw, min_xw:max_xw], (crop_width, crop_height))
            images.append(crop)
            print(square, label)
            cv.imshow("Crop", crop)
            cv.imshow("Original", im)
            cv.waitKey()
    return images, labels

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
    corners = metadata['corners']
    white_view = metadata['white_turn']
    images, labels = board_localization(im, pieces, corners, white_view)

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