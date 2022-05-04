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

def good_features_to_track(initial_image, box_coords, box_dims, tolerance):
    '''
    Similar to the Harris Corner detector, but modified by Shi and Tomasi. This algorithm should
    pick a set of coordinates within the bounding box plus the area specified by the tolerance.
    Params:
     - initial_image: the complete image containing a face
     - box_coords: the upper lefthand coordinates of the bounding box that contains the face
     - box_dims: the dimensions of the bounding box of the face. Together with the box_coords, the
                    image space that contains the face can be isolated
     - tolerance: (optional) the amount of space outside the bounding box to also check. Used to find
                    nearby space in a theoretically translated image, so that we can center around the
                    previous bounding box. TODO THIS IS A GUESS ADDITION TO THE ALGO
    '''
    return None

def geometric_transform(gftt1, gftt2):
    '''
    '''
    dx=0
    dy=0
    return dx, dy