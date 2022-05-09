# from tkinter import HIDDEN # do we need this?
import re
from cv2 import edgePreservingFilter
from matplotlib.pyplot import cla
import numpy as np
import math
from scipy.misc import face
from sklearn.svm import SVC, LinearSVC
from skimage.io import imread
import os

num_face_images = 0
num_nonface_images = 0
num_of_weak_classifiers = 10
alpha_vals = []
final_classifiers = []

def viola_jones(image_paths):
    '''
    Runs the entirety of the Viola-Jones algorithm configured for facial feature
    detection (helper methods expected). This helper method is called in run.py

    Params:
    - image: the image (could be a singular frame from a video)
    
    Returns:
    - an image of the same dimensions with rectangles overlayed over the faces
    '''

    num_imgs = len(image_paths)
    feature_threshold = 0.5
    feature_size = 5

    # file path of image
    # number of faces
    # [x1, y1, width, height] of each face
    bounding_boxes = np.empty()

    # for i in range(num_imgs):
    #     image = imread(image_paths[i])
    #     image = create_integral_image(image)
    #     for row in range(0, image.shape[0], ):
    #         for col in range(0, image.shape[1], ):
    #             overall_score = 0
    #             # loop through each possible feature
    #             for feature in range(0,5):
    #                 score = haar_like_features(image, feature, feature_size, (row, col))
                    
    #                 if (score < feature_threshold):
    #                     overall_score -= 1
    #                 else:
    #                     overall_score += 1
                
    #             if overall_score >= 0: # more likely than not to be a face
    #                 # store the bounding boxes
    #                 bounding_boxes = np.append([row, col, feature_size, feature_size])
    
    return bounding_boxes
    
                
                    
        

def haar_like_features(integral_image, feature, feature_size, start_position):
    '''
    Gets the Haar-like features. There are 4 different types of Haar features
    that are used in Viola-Jones to detect faces: 
    1: two verical rectangles, white on left and black on right
    2: two horizontal rectangles, white on top and black on bottom
    3: three vertical rectangles, white on either side and black in middle
    4: four squares in a checkerboard, white on bottom left and top right
    
    param: 
    - feature_size : tuple height by width
    - start_position: tuple of (int, int) representing the (row,col) starting location (top left)
    '''
    
    white = 0
    black = 0
    if feature == 1:
        black_end_pos = (start_position[0] + feature_size[0] / 2, start_position[1] + feature_size[1])
        black = quick_sum(integral_image, start_position, black_end_pos)

        white_start_pos = (start_position[0] + feature_size[0] / 2, start_position[1])
        white_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])
        white = quick_sum(integral_image, white_start_pos, white_end_pos)

    elif feature == 2:
        white_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1] / 2)
        white = quick_sum(integral_image, start_position, white_end_pos)

        black_start_pos = (start_position[0], start_position[1] + feature_size[1] / 2)
        black_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 3:
        white_one_end_pos = (start_position[0] + feature_size[0] / 3, start_position[1] + feature_size[1])

        black_start_pos = (start_position[0] + feature_size[0] / 3, start_position[1])
        black_end_pos = (start_position[0] + 2 * (feature_size[0] / 3), start_position[1] + feature_size[1])

        white_two_start_pos = (start_position[0] + 2 * (feature_size[0] / 3), start_position[1])
        white_two_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])

        white = quick_sum(integral_image, start_position, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 4:
        white_one_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1] / 3)

        black_start_pos = (start_position[0] , start_position[1] + feature_size[1] / 3)
        black_end_pos = (start_position[0] + feature_size[0], start_position[1] + 2 * (feature_size[1] / 3))

        white_two_start_pos = (start_position[0], start_position[1] + 2 * (feature_size[1] / 3))
        white_two_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])

        white = quick_sum(integral_image, start_position, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 5:
        middle_row = start_position[0] + feature_size[0] / 2
        middle_col = start_position[1] + feature_size[1] / 2

        white_one_end_pos = (middle_row, middle_col)
        white_two_start_pos = (middle_row, middle_col)
        white_two_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])

        black_one_start_pos = (start_position[0], middle_col)
        black_one_end_pos = (middle_row, start_position[1] + feature_size[1])
        black_two_start_pos = (middle_row, start_position[1])
        black_two_end_pos = (start_position[0] + feature_size, middle_col)

        white = quick_sum(integral_image, start_position, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, black_one_start_pos, black_one_end_pos) + quick_sum(integral_image, black_two_start_pos, black_two_end_pos)

    return black - white



def create_integral_image(image):
    '''
    Creates the integral image

    takes in normal image 
    returns integral image
    '''
    num_rows = np.shape(image)[0]
    num_cols = np.shape(image)[1]
    integral_image = np.zeros((num_rows, num_cols))
    temp_sum = 0
    for row in range(num_rows):
        for col in range(num_cols):
            if (row == 0):
                if (col == 0):
                    integral_image[row][col] = image[row][col]
                else:
                    integral_image[row][col] = image[row][col] + integral_image[row][col - 1]
            else:
                if (col == 0):
                    temp_sum = 0
                integral_image[row][col] = image[row][col] + integral_image[row - 1][col] + temp_sum
                temp_sum = temp_sum + image[row][col]
    return integral_image

def quick_sum(integral_image, top_left, width, height):
    '''
    Sums the value over a given area using an integral image
    topLeft (row , col)

    '''
    top_left_val = integral_image[top_left[0]][top_left[1]]
    top_right_val = integral_image[top_left[0]][top_left[1] + width]
    bottom_left_val = integral_image[top_left[0] + height][top_left[1]]
    bottom_right_val = integral_image[top_left[0] + height][top_left[1] + width]
    return (top_left_val + bottom_right_val - top_right_val - bottom_left_val)

def get_all_feats(image):
    '''
    Returns arrary of all possible features within an image
    '''
    possible_feats = []
    rows = image.shape[0]
    columns = image.shape[1]
    for width in range(1, columns+1):
        for height in range(1, rows + 1):
            col = 0
            while width + col < columns:
                row = 0
                while height + row < rows:

                    first_rec = [col, row, width, height]
                    right_rec = [col + width, row, width, height]
                    below_first = [col, row + height, width, height]
                    right_rec_2 = [col + (2 * width), row, width, height]
                    below_first_2 = [col, row + (2 * height), width, height]
                    bottom_right = [col + width, row + height, width, height]

                    if ((col + (2 * width)) < columns):
                        possible_feats.append(([right_rec], [first_rec], 2)) #feature type 2
                    
                    if ((row + (2 * height)) < rows):
                        possible_feats.append(([first_rec], [below_first], 1)) #feature type 1                
                
                    if ((col + (3 * width)) < columns):
                        possible_feats.append(([right_rec], [first_rec, right_rec_2], 4)) #feature type 4
                                      
                    if ((row + (3 * height)) < rows):
                        possible_feats.append(([below_first], [first_rec, below_first_2], 3)) #feature type 3
                                     
                    if (((col + (2 * width)) < columns) and ((row + (2 * height) < rows))):
                        possible_feats.append(([right_rec, below_first], [first_rec, bottom_right], 5)) #feature type 5

                    row = row + 1
                col = col + 1
    return np.array(possible_feats)

def get_all_values(features, integral_img_data):
    '''
    returns all of the values for all features in all imgs
    array is number of feats by num of imgs
    '''
    values = np.zeros((len(features), len(integral_img_data)))

    #go through features
    for feature in range (0, len(features)):
        feat = features[feature]
        #go through images
        for image in range (0, len(integral_img_data)):
            integral_img = integral_img_data[image][0] #get integral img
            pos_sum = 0
            neg_sum = 0
            #sum values
            for rectangle in feat[0]:
                pos_sum = pos_sum + quick_sum(integral_img, (rectangle[1], rectangle[0]), rectangle[2], rectangle[3])
            for rectangle in feat[1]:
                neg_sum = neg_sum + quick_sum(integral_img, (rectangle[1], rectangle[0]), rectangle[2], rectangle[3])
            values[feature][image] = pos_sum - neg_sum
    
    return values

def weak_classify(classifier, integral_img):
    '''
    Classifies weak classifiers that are in the form of [pos_regions, neg_regions, threshold, polarity]
    '''
    pos_sum = 0
    neg_sum = 0

    for rectangle in classifier[0]:
        pos_sum = pos_sum + quick_sum(integral_img, (rectangle[1], rectangle[0]), rectangle[2], rectangle[3])
    for rectangle in classifier[1]:
        neg_sum = neg_sum + quick_sum(integral_img, (rectangle[1], rectangle[0]), rectangle[2], rectangle[3])

    total_sum = pos_sum - neg_sum
    if (classifier[3] * total_sum < classifier[3] * classifier[2]):
        return 1
    else:
        return 0

def train_weak_classifiers(feature_values, gt_labels, features, weights):
    total_face = 0
    total_nonface = 0

    for img in range(0, len(gt_labels)):
        if gt_labels[img] == 1:
            total_face += weights[img]
        else:
            total_nonface += weights[img]
    
    classifiers = []
    num_of_features = len(features)

    #for all features classifiers
    for index in range(len(feature_values)):
        feature = feature_values[index]
        if len(classifiers) % 1000 == 0 and len(classifiers) != 0: #print intermediate message
            print("%d classifiers trained out of %d" % (len(classifiers), num_of_features))
        
        #bundle and sort data
        feat_data = []
        for index2 in range(len(weights)):
            feat_data.append((weights[index2], feature[index2], gt_labels[index2]))

        sorted_feats = sorted(feat_data, key=lambda val: val[1])
        
        curr_pos = 0 
        curr_neg = 0
        pos_weights = 0
        neg_weights = 0
        best_error = float('inf')
        best_feat = None
        best_threshold = None
        best_polar = None

        for feat_num in range(0, len(sorted_feats)):
            weight = sorted_feats[feat_num][0]
            feat = sorted_feats[feat_num][1]
            g_truth = sorted_feats[feat_num][2]
            e1 = neg_weights + total_face - pos_weights
            e2 = pos_weights + total_nonface - neg_weights
            error = min(e1, e2)
            if error < best_error:
                best_error = error
                best_feat = features[index]
                best_threshold = feat
                if (curr_pos > curr_neg):
                    best_polar = 1
                else:
                    best_polar = -1
            
            if g_truth == 1:
                curr_pos = curr_pos + 1
                pos_weights = pos_weights + weight
            else:
                curr_neg = curr_neg + 1
                neg_weights = neg_weights + weight
        
        curr_classifier = [best_feat[0], best_feat[1], best_threshold, best_polar]
        classifiers.append(curr_classifier)
    
    return classifiers 

def find_best_weak(classifiers, weights, interal_img_data):
    best_classifier = None
    best_error = float('inf')
    best_accuracy = None

    for classifier_num in range(0, len(classifiers)):
        curr_classifier = classifiers[classifier_num]
        curr_error = 0
        curr_accuracy = []
        for index in range(len(interal_img_data)):
            img_data = interal_img_data[index]
            curr_weight = weights[index]
            correct_val = abs(weak_classify(curr_classifier, img_data[0]) - img_data[1])
            curr_accuracy.append(correct_val)
            curr_error = curr_error + (curr_weight * correct_val)
        curr_error = curr_error / len(interal_img_data)
        if curr_error < best_error:
            best_classifier = curr_classifier
            best_error = curr_error
            best_accuracy = curr_accuracy
            
    return best_classifier, best_error, best_accuracy

def training(training_data, gt_labels):
    integral_imgs = []
    weights = np.zeros(len(training_data))
    for i in range(len(training_data)):

        #cerate integral images
        integral_imgs.append((create_integral_image(training_data[i][0]), training_data[i][1]))

        #figure out what to intialize weights to
        if (training_data[i][1] == 1):
            weights[i] = 1 / (2 * num_face_images)
        else:
            weights[i] = 1 / (2 * num_nonface_images)
    
    #get all possible features in img
    features = get_all_feats(integral_imgs[0][0])
    #get values of all features
    feature_values = get_all_values(features, integral_imgs)
    for l in range(num_of_weak_classifiers):
        weights = weights / np.linalg.norm(weights) #normailize weights
        weak_classifiers = train_weak_classifiers(feature_values, gt_labels, features, weights)
        classifier, error, accuracy = find_best_weak(weak_classifiers, weights, integral_imgs)
        beta = error / (1.0 - error)
        for val in range(len(accuracy)):
            weights[val] = weights[val] * (beta ** (1 - accuracy[val]))
        alpha = math.log(1.0/beta)
        alpha_vals.append(alpha)
        final_classifiers.append(classifier)



    



def adaboost_training(image):
    '''
    Conducts AdaBoost training
    '''
    return image

def cascading_classifiers(image):
    return image
    

def read_in_gt(gt_filepath):
    if "data" not in os.getcwd():
        os.chdir("data")
    if "wider_face_split" not in os.getcwd():
        os.chdir("wider_face_split")
    gt_file = open(gt_filepath, 'r')

    # count the number of images in the gt labels
    num_images = 0
    while True:
        line = gt_file.readline()

        if "0--" in line:
            num_images += 1
        
        if not line:
            break

    gt_labels = np.empty((1, num_images))

    while True:
        line = gt_file.readline()

        if "0--" in line:
            num_faces = gt_file.readline()
            faces_array = np.empty((num_faces, 4))
            
            line = gt_file.readline()
            # fill in data for each face
            # [x1, y1, width, height]
            while "0--" not in line:
                data = line.split()
                x1 = data[0]
                x2 = data[1]
                width = data[2]
                height = data[3]

                faces_array = np.append(faces_array, np.array((x1, x2, width, height)))
                line = gt_file.readline()
            
            # append array of faces to array of images
            gt_labels = np.append(gt_labels, faces_array)

        if not line:
            break

    print(gt_labels)


    
