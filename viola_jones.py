# from tkinter import HIDDEN # do we need this?
from cv2 import edgePreservingFilter
import numpy as np
from scipy.misc import face
from sklearn.svm import SVC, LinearSVC
from skimage.io import imread
import os

num_face_images = 0
num_nonface_images = 0

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
    - feature_size : tuple width by height
    - start_position: tuple of (int, int) representing the (x,y) starting location (top left)
    '''
    
    white = 0
    black = 0
    if feature == 1:
        white_end_pos = (start_position[0] + feature_size[0] / 2, start_position[1] + feature_size[1])
        white = quick_sum(integral_image, start_position, white_end_pos)

        black_start_pos = (start_position[0] + feature_size[0] / 2, start_position[1])
        black_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 2:
        white_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1] / 2)
        white = quick_sum(integral_image, start_position, white_end_pos)

        black_start_pos = (start_position[0], start_position[1] + feature_size[1] / 2)
        black_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 3:
        white_one_end_pos = (start_position[0] + feature_size[0] / 4, start_position[1] + feature_size[1])

        black_start_pos = (start_position[0] + feature_size[0] / 4, start_position[1])
        black_end_pos = (start_position[0] + 3 * (feature_size[0] / 4), start_position[1] + feature_size[1])

        white_two_start_pos = (start_position[0] + 3 * (feature_size[0] / 4), start_position[1])
        white_two_end_pos = (start_position[0] + feature_size[0], start_position[1] + feature_size[1])

        white = quick_sum(integral_image, start_position, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 4:
        middle_x = start_position[0] + feature_size[0] / 2
        middle_y = start_position[1] + feature_size[1] / 2

        black_one_end_pos = (middle_x, middle_y)
        black_two_start_pos = (middle_x, middle_y)
        black_two_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size[1])

        white_one_start_pos = (start_position[0], middle_y)
        white_one_end_pos = (middle_x, start_position[1] + feature_size[1])
        white_two_start_pos = (middle_x, start_position[1])
        white_two_end_pos = (start_position[0] + feature_size, middle_y)

        white = quick_sum(integral_image, white_one_start_pos, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, start_position, black_one_end_pos) + quick_sum(integral_image, black_two_start_pos, black_two_end_pos)

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

def quick_sum(integral_image, top_left, bottom_right):
    '''
    Sums the value over a given area using an integral image
    '''
    top_left_val = integral_image[top_left[0]][top_left[1]]
    top_right_val = integral_image[top_left[0]][bottom_right[1]]
    bottom_left_val = integral_image[bottom_right[0]][top_left[1]]
    bottom_right_val = integral_image[bottom_right[0]][bottom_right[1]]
    return (top_left_val + bottom_right_val - top_right_val - bottom_left_val)

def get_all_feats(image):
    '''
    Returns arrary of all possible features within an image
    '''
    rows = image.shape[0]
    columns = image.shape[1]
    for temp_Width in range(1, columns+1):
        for temp_Height in range(1, rows + 1):
            edgeW = 0
            while temp_Width + edgeW < columns:
                edgeH = 0
                while temp_Height + edgeH < rows:
                    
                    

def training(training_data, images):
    integral_imgs = []
    weights = np.zeros(len(training_data))
    for i in range(len(images)):

        #cerate integral images
        integral_imgs.append(create_integral_image(images[i]))

        #figure out what to intialize weights to
        if (len(training_data[i]) > 0):
            weights[i] = 1 / (2 * num_face_images)
        else:
            weights[i] = 1 / (2 * num_nonface_images)


    



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


    
