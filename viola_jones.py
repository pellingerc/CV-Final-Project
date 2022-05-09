import numpy as np
from scipy.misc import face
from sklearn.svm import SVC, LinearSVC
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
import os


def run_it(train_image_paths, test_image_paths, gt_file_path):
    train_image_feats = viola_jones(train_image_paths)
    test_image_feats  = viola_jones(test_image_paths)
    gt_labels = read_in_gt(gt_file_path)

    predictions = svm_classify(train_image_feats, gt_labels, test_iamge_feats)
    
    return predictions

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

    for i in range(num_imgs):
        image = imread(image_paths[i])
        image = create_integral_image(image)
        for row in range(0, image.shape[0], ):
            for col in range(0, image.shape[1], ):
                overall_score = 0
                # loop through each possible feature
                for feature in range(0,5):
                    score = haar_like_features(image, feature, feature_size, (row, col))
                    
                    if (score < feature_threshold):
                        overall_score -= 1
                    else:
                        overall_score += 1
                
                if overall_score >= 0: # more likely than not to be a face
                    # store the bounding boxes
                    bounding_boxes = np.append([row, col, feature_size, feature_size])
    
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
    - start_position: tuple of (int, int) representing the (x,y) starting location (top left)
    '''
    
    white = 0
    black = 0
    if feature == 1:
        white_end_pos = (start_position[0] + feature_size / 2, start_position[1] + feature_size)
        white = quick_sum(integral_image, start_position, white_end_pos)

        black_start_pos = (start_position[0] + feature_size / 2, start_position[1])
        black_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 2:
        white_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size / 2)
        white = quick_sum(integral_image, start_position, white_end_pos)

        black_start_pos = (start_position[0], start_position[1] + feature_size / 2)
        black_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 3:
        white_one_end_pos = (start_position[0] + feature_size / 4, start_position[1] + feature_size)

        black_start_pos = (start_position[0] + feature_size / 4, start_position[1])
        black_end_pos = (start_position[0] + 3 * (feature_size / 4), start_position[1] + feature_size)

        white_two_start_pos = (start_position[0] + 3 * (feature_size / 4), start_position[1])
        white_two_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size)

        white = quick_sum(integral_image, start_position, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, black_start_pos, black_end_pos)

    elif feature == 4:
        middle_x = start_position[0] + feature_size / 2
        middle_y = start_position[1] + feature_size / 2

        black_one_end_pos = (middle_x, middle_y)
        black_two_start_pos = (middle_x, middle_y)
        black_two_end_pos = (start_position[0] + feature_size, start_position[1] + feature_size)

        white_one_start_pos = (start_position[0], middle_y)
        white_one_end_pos = (middle_x, start_position[1] + feature_size)
        white_two_start_pos = (middle_x, start_position[1])
        white_two_end_pos = (start_position[0] + feature_size, middle_y)

        white = quick_sum(integral_image, white_one_start_pos, white_one_end_pos) + quick_sum(integral_image, white_two_start_pos, white_two_end_pos)
        black = quick_sum(integral_image, start_position, black_one_end_pos) + quick_sum(integral_image, black_two_start_pos, black_two_end_pos)

    return black - white



def create_integral_image(image):
    '''
    Creates the integral image
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
    top_left_val = integral_image[top_left[0]][top_left[1]]
    top_right_val = integral_image[top_left[0]][bottom_right[1]]
    bottom_left_val = integral_image[bottom_right[0]][top_left[1]]
    bottom_right_val = integral_image[bottom_right[0]][bottom_right[1]]
    return (top_left_val + bottom_right_val - top_right_val - bottom_left_val)


def adaboost_training(image):
    '''
    Conducts AdaBoost training
    '''
    return image

def cascading_classifiers(image):
    return image
    
def svm_classify(train_image_feats, train_labels, test_image_feats):
    final_labels = np.empty((np.shape(test_image_feats)[0], 1), dtype='U100')
    X = SVC(kernel="linear")
    X.fit(train_image_feats, train_labels)
    Z = X.predict(test_image_feats)
    
    return Z



def create_gt_labels():
    '''
    Reads in the images from the dataset and creates an array

    Returns:
    gt_labels_with_images:
        - n x 1 array of tuples, each tuple has the m x m image and an int
        - int vale is 1 if there is a face in image, 0 if not
        - note: to index, gt_labels_with_images[image index] gives tuple
                          gt_labels_with_images[image index][0] gives image
                          gt_labels_with_images[image index][1] gives integer label
    
    gt_labels:
        - n x 1 array of integers (same order as gt_labels_with_images)
        - int vale is 1 if there is a face in image, 0 if not

    '''
    print(os.getcwd())
    while "CV-Final-Project/" in os.getcwd():
        os.chdir("..")
    os.chdir("data/faces/train/face")

    gt_labels_with_images = np.empty((0,0), dtype=object)
    gt_labels = np.empty((0,0))

    # fill in faces
    face_image_paths = os.listdir()
    num_images = len(face_image_paths)

    for path in face_image_paths:
        image = imread(path)
        image_tuple = (image, 1)
        gt_labels_with_images = np.append(gt_labels_with_images, image_tuple)
        gt_labels = np.append(gt_labels, 1)

    # fill in non-faces
    os.chdir("../non-face")

    non_face_image_paths = os.listdir()
    num_images += len(non_face_image_paths)

    for path in non_face_image_paths:
        image = imread(path)
        image_tuple = (image, 0)
        gt_labels_with_images = np.append(gt_labels_with_images, image_tuple)
        gt_labels = np.append(gt_labels, 0)

    gt_labels_with_images = np.reshape(gt_labels_with_images, (num_images, 2))

    os.chdir("../..")
    
    return gt_labels_with_images, gt_labels
    

def read_in_gt(gt_filename):

    # go into directory with ground truth text files 
    # at CV-Final-Project/data/wider_face_split
    while "CV-Final-Project/" in os.getcwd():
        os.chdir("..")
    os.chdir("data")
    os.chdir("wider_face_split")

    gt_labels = np.empty((0,0,0))
    images = np.empty((0,0,0))
    num_images = 0

    gt_file = open(gt_filename, 'r')

    line = gt_file.readline()
    while True:

        if "--" in line:
            num_images += 1

            # move to directory with images at data/WIDER_train/images
            os.chdir("..")
            os.chdir("WIDER_train/images")

            # load image and put it into images array
            curr_image = rgb2gray(img_as_float(imread(line.strip())))
            images = np.append(images, curr_image)

            # return back to original directory
            os.chdir("..")
            os.chdir("..")
            os.chdir("wider_face_split")
            
            line = gt_file.readline() # num faces
            num_faces = int(line)
            faces_array = np.empty((0,0))

            print("num faces: ", line)
            
            # fill in data for each face
            # [x, y, width, height]
            while "--" not in line:
            
                data = line.split()
                
                if len(data) > 1:
                    x = data[0]
                    y = data[1]
                    width = data[2]
                    height = data[3]
                    
                    face_data = [[x, y, width, height]]
                
                    faces_array = np.append(faces_array, face_data)   
                
                # add in else statement here to update number of face pics global var

                line = gt_file.readline()

                if not line:
                    break
            
            faces_array = np.reshape(faces_array, (num_faces, 4))

            # append array of faces to array of images
            gt_labels = np.append(gt_labels, faces_array)
        
        if not line:
            break
    
    print(gt_labels)
    gt_file.close()
    
