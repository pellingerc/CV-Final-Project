from cmath import pi
import cv2
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

    C_cutoff = feature.peak_local_max(cornerness, min_distance = 5, threshold_rel=0.001) #0.005

    xs = C_cutoff[:,1]
    ys = C_cutoff[:,0]
    
    xs = xs+box_dims[0]
    ys = ys+box_dims[1]
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys

def get_next_points(last_image, current_image, last_xs, last_ys, window_size):
    '''
    Based on the Kanade-Lucas sparse optical flow algorithm, this algorithm will project where
    the previous points will end up. This is done through gradient calculations as specified
    by in-line comments.
    Params:
     - last_image: the frame previous to the current one, from which the last_xs and last_ys were calculated
     - current_imaeg: the current frame, for which new points are being solved
     - last_xs: x values of points of interest on the last image frame, from get_interest_points
     - last_ys: y values of points of interest on the last image frame, from get_interest_points
     - window_size: the amount of space around each point of interest to examine for the optical flow algorithm
    Returns:
     - new_xs: the new x values of the interest points
     - new_ys: the new y values of the interest points
    '''
    # last_image = filters.gaussian(last_image)
    last_image = cv2.GaussianBlur(last_image,(5,5),0.2)
    # current_image = filters.gaussian(current_image)

    new_xs = np.zeros(len(last_xs))
    new_ys = np.zeros(len(last_ys))

    # I_t = I(x,y,t+1)-I(x,y,t)
    temporal_derivative = (current_image-last_image)
    x_gradient = ndimage.sobel(last_image, axis=1)
    y_gradient = ndimage.sobel(last_image, axis=0)
    # x_gradient, y_gradient = np.gradient(last_image)

    # for each interest point, calculate the projected interest point
    for i in range(0, len(last_xs)):
        last_xs[i] = int(last_xs[i])
        last_ys[i] = int(last_ys[i])
        if (last_xs[i] >= window_size/2 and last_ys[i]>= window_size/2 and last_xs[i] < np.shape(last_image)[0]-window_size/2 and last_ys[i] < np.shape(last_image)[1]-window_size/2):
            AtransA = np.zeros((2,2)) # eventually become the A^T * A left half of the least squares solution equation
            Atransb = np.zeros(2)
            
            # look at indices in the window_size x window_size window around the point (or as far as you can go) 
            # TODO is this still accurate with the bounding?
            xrange = (np.arange((max(0, last_xs[i]-int(window_size/2))), (min(last_image.shape[0]-1,last_xs[i]+int(window_size/2))))).astype(int)
            yrange = np.arange((max(0, last_ys[i]-int(window_size/2))), (min(last_image.shape[1]-1,last_ys[i]+int(window_size/2)))).astype(int)

            # add to the left half of the equation
            AtransA[0][0] = np.sum(np.multiply(x_gradient[xrange,yrange], x_gradient[xrange,yrange]))
            AtransA[0][1] = np.sum(np.multiply(x_gradient[xrange,yrange], y_gradient[xrange,yrange]))
            AtransA[1][0] = np.sum(np.multiply(x_gradient[xrange,yrange], y_gradient[xrange,yrange]))
            AtransA[1][1] = np.sum(np.multiply(y_gradient[xrange,yrange], y_gradient[xrange,yrange]))

            #add to the right half of the equation
            Atransb[0] = np.sum(np.multiply(x_gradient[xrange,yrange],temporal_derivative[xrange,yrange]))
            Atransb[1] = np.sum(np.multiply(y_gradient[xrange,yrange],temporal_derivative[xrange,yrange]))
            
            # calculate the least squares and solve for the x vector by inverting the left side of the equation and multiplying it to the right
            uv = [0,0]
            det = np.linalg.det(AtransA)
            if (np.linalg.det(AtransA)!=0):
                L = np.zeros((2,2))
                L[0][0] = AtransA[1][1]/det
                L[0][1] = -AtransA[0][1]/det
                L[1][0] = -AtransA[1][0]/det
                L[1][1] = AtransA[0][0]/det
                # uv = np.matmul(np.linalg.inv(AtransA),(-1*Atransb))

                # uv = [L[0][0]*Atransb[0]+L[0][1]*Atransb[1], L[1][0]*Atransb[0]+L[1][1]*Atransb[1]]

            # add the new vectors
            # TODO flip 0 and 1 back bc this makes no sense
            # new_xs[i] = int(last_xs[i]-(10*uv[0]))
            # new_ys[i] = int(last_ys[i]-10*uv[1])
            new_xs[i] = last_xs[i] + uv[0]
            new_ys[i] = last_ys[i] + uv[1]
            # print(uv)

    return new_xs, new_ys

def new_bounding_box(new_xs, new_ys, oldBBdims):
    '''
    Shifts the bounding box to the average of the new xs and ys
    Params:
     - new_xs: the x-coordinates of the interest points
     - new_ys: the y-coordinates of the interest points
     - oldBBdims: (x, y, width, height) of old bounding boxes
    Returns:
     - new_bounding_box_dims: (x, y, width, height) of new bounding box
    '''

    avg_x = np.average(new_xs)
    avg_y = np.average(new_ys)

    old_center_x = oldBBdims[0][0]+int(oldBBdims[0][2]/2)
    old_center_y = oldBBdims[0][1]+int(oldBBdims[0][3]/2)

    dx = old_center_x-avg_x
    dy = old_center_y-avg_y

    return [[oldBBdims[0][0]-dx, oldBBdims[0][1]-dy, oldBBdims[0][2], oldBBdims[0][3]]]