from cmath import pi
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops
from scipy import ndimage

# an implementation of the Kanade–Lucas–Tomasi feature tracker 
# reference: https://en.wikipedia.org/wiki/Kanade%E2%80%93Lucas%E2%80%93Tomasi_feature_tracker
# written by Caroline (Coco) Kaleel during May 2022
# final project for CSCI 1430: Computer Vision, taught by Prof. James Tompkin
'''
Pseudocode:
- use viola jones to produce a space with necessary features to track within the image
- pull 'good features to track' (shi and tomasi) (eigenfeatures?)
- estimate geometric transform between nearby features, move box by that amount
'''

'''
TODO question for Tarun: how should we implement tracking? like should the first frame have 
viola jones implemented to detect, and the next 50 frames be klt, and then that process repeats? like
should it hand off? confusion
'''

def get_interest_points(initial_image, box_dims, tolerance=None):
    '''
    TODO: Similar to the Harris Corner detector, but modified by Shi and Tomasi. This algorithm should
    pick a set of coordinates within the bounding box plus the area specified by the tolerance.
    
    Current:
    Adapted from Homework 2, this is a Harris corner detector system that works within
    the specified bounding box.

    Params:
     - initial_image: the complete image containing a face
     - box_dims: first two indices: the upper lefthand coordinates of the bounding box that contains the face
                second two indices: the dimensions of the bounding box of the face. Together with the box_coords, the
                    image space that contains the face can be isolated
     - tolerance: (optional) the amount of space outside the bounding box to also check. Used to find
                    nearby space in a theoretically translated image, so that we can center around the
                    previous bounding box. TODO THIS IS A GUESS ADDITION TO THE ALGO
    '''
    cropped_image=initial_image[box_dims[0]:box_dims[0]+box_dims[2],box_dims[1]:box_dims[1]+box_dims[3]]

    cropped_image = filters.gaussian(cropped_image)

    xgradient = ndimage.sobel(cropped_image, axis=1)
    ygradient = ndimage.sobel(cropped_image, axis=0)

    xgradient_squared = np.square(xgradient)
    xy_gradient = np.multiply(xgradient, ygradient)
    ygradient_squared = np.square(ygradient)

    sig = 1
    x_gaussian = filters.gaussian(xgradient_squared, sig)
    y_gaussian = filters.gaussian(ygradient_squared, sig)
    xy_gaussian = filters.gaussian(xy_gradient, sig)

    alpha = 0.05

    cornerness = np.multiply(x_gaussian,y_gaussian)-np.square(xy_gaussian)-alpha*np.square(x_gaussian+y_gaussian)

    # threshhold = 0.005
    # C[C <= threshhold] = 0

    C_cutoff = feature.peak_local_max(cornerness, min_distance = 5, threshold_rel=0.005)

    xs = C_cutoff[:,1]
    ys = C_cutoff[:,0]
    
    xs = xs+box_dims[0]
    ys = ys+box_dims[1]
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys

def geometric_transform(gftt1, gftt2):
    '''
    This algorithm should calculate the approximate x and y difference between the two detected faces.
    It will take the results of good_features_to_track 1 and 2
    '''
    dx=0
    dy=0
    return dx, dy


def get_features(image, x, y, feature_width): #feature width usually 16
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: np array of computed features. It should be of size
            [len(x) * feature dimensionality] (for standard SIFT feature
            dimensionality is 128)

    '''

    # TODO: Your implementation here! See block comments and the project webpage for instructions
    
    # DONE STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels. np.gradient()
    # DONE STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    image = filters.gaussian(image)
    image = filters.gaussian(image)

    grad_x = filters.sobel_h(image)
    grad_y = filters.sobel_v(image)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))
    grad_ori = np.arctan2( grad_y, grad_x )

    # This is a placeholder - replace this with your features!
    features = np.zeros((len(x),128))

    #loop for every coordinate
    #for every 4/4 square in feature width (0 - feature_width/4 in x and 0-feature_width/4) aka (-feature_width/8 to feature_width/8 in x and y)

    for index in range(0,len(x)):
        xcoord = x[index]
        ycoord = y[index]

        for local_x_count in range(-1*feature_width//8, feature_width//8): #the number of the square we are calculating for relative to the keypoint
            for local_y_count in range(-1*feature_width//8, feature_width//8):
                for x_offset in range(0,4): #how far off the local square is
                    for y_offset in range (0,4):
                        # # find the current orientation and magnitude at the local x
                        # current_grad_ori = grad_mag[xcoord+local_x_count*4+x_offset][ycoord+local_y_count*4+y_offset]
                        # current_grad_mag = grad_ori[xcoord+local_x_count*4+x_offset][ycoord+local_y_count*4+y_offset]
                        indexX = ycoord+local_x_count*4+x_offset
                        indexY = xcoord+local_y_count*4+y_offset
                        if (indexX < len(grad_ori) and indexY < len(grad_ori[0])):

                            current_grad_ori = grad_mag[indexX][indexY]
                            current_grad_mag = grad_ori[indexX][indexY]
                            #TODO does this work now??

                            # convert radian calculation into bin number through modulus division:
                            bin_number = (8*current_grad_ori/(2 * np.pi)) % 8

                            # find the number of the feature by multiplying out the offsets and adding the bin_number
                            # there is a photo of where this calculation came from in my writeup:
                            feature_number = (int)(8*(local_x_count+feature_width//8)+32*(local_y_count+feature_width//8)+bin_number)

                            features[index][feature_number]+=current_grad_mag
    
    features = features/np.linalg.norm(features)
    
    return features