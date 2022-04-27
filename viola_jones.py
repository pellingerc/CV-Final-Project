import numpy as np



def viola_jones(image):
    '''
    Runs the entirety of the Viola-Jones algorithm configured for facial feature
    detection (helper methods expected). This helper method is called in run.py

    Params:
    - image: the image (could be a singular frame from a video)
    
    Returns:
    - an image of the same dimensions with rectangles overlayed over the faces
    '''
    return image

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