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

def klt(initial_image, new_image, bounding_box_coords, bounding_box_dims):
    '''
    The main method for an implementation of the Kanade-Lucas-Tomasi feature tracker. Pulls
    important features from within the bounding box.
    '''
    dx = 0
    dy = 0
    return dx, dy

def good_features_to_track(initial_image, box_coords, box_dims, tolerance=None):
    '''
    TODO: Similar to the Harris Corner detector, but modified by Shi and Tomasi. This algorithm should
    pick a set of coordinates within the bounding box plus the area specified by the tolerance.
    
    Current:
    Adapted from Homework 2, this is a Harris corner detector system that works within
    the specified bounding box.

    Params:
     - initial_image: the complete image containing a face
     - box_coords: the upper lefthand coordinates of the bounding box that contains the face
     - box_dims: the dimensions of the bounding box of the face. Together with the box_coords, the
                    image space that contains the face can be isolated
     - tolerance: (optional) the amount of space outside the bounding box to also check. Used to find
                    nearby space in a theoretically translated image, so that we can center around the
                    previous bounding box. TODO THIS IS A GUESS ADDITION TO THE ALGO
    '''
    cropped_image=initial_image[box_coords[0]:box_coords[0]+box_dims[0],box_coords[1]:box_coords[1]+box_dims[1]]

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
    
    xs = xs+box_coords[0]
    ys = ys+box_coords[1]
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