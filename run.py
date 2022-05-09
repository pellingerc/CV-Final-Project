# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import cv2
import argparse
import time
from cv2 import rectangle

import numpy as np

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage import io, img_as_float
from skimage.color import rgb2gray

import viola_jones as vj
import cheat_face_detection as cheat

from tracking import get_interest_points, get_features


def main():
    """
    Reads in the arguments to determine if video should be on, then runs the Viola Jones face detection
    algorithm and overlays a bounding box rectangle if a face is in fact detected.

    Command line usage: python run.py [-v | --video <"on" or "off" (no quotes/brackets)>]

    -v | --video - flag - not required. specifies if the program should be run with 
                                    live video or on a static image. default on.

    KILL PROGRAM WITH SIGNAL INTERRUPTS (EX: cmd/ctrl c or z to suspend)

    """

    # create the command line parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video",
                        required=True,
                        choices=["on","off"],
                        help="Specify whether video \"on\" or to use a static image with \"off\"")
    parser.add_argument('-l', '--load', default='True', help='Boolean for either loading existing vocab (True) or creating new one (False)')
    parser.add_argument('-d', '--data', default='../data', help='Filepath to the data directory')
    args = parser.parse_args()

    videoOn = (args.video == "on")

    if not videoOn:
        ## TODO: call viola_jones on static image
        training_data, gt_label, num_face_images, num_nonface_images = vj.create_gt_labels()
        alpha_vals, final_classifiers = vj.training(training_data, gt_label, num_face_images, num_nonface_images)
        print("camera off")
        
        correct = 0
        for x, y in training_data:
            correct += 1 if vj.classify(x, alpha_vals, final_classifiers) == y else 0
        print("Classified %d out of %d test examples" % (correct, len(training_data)))

    else:
        ## TODO: call viola_jones on live video
        print("camera on")
        live_viola_jones(videoOn)

    

    


###### code from HW2 life fourier transform (liveFFT2.py)

class live_viola_jones():
    """
    This function shows the live Viola-Jones facial detection on a continuous stream of 
        images captured from a built-in or attached camera.

    Adapted by Caroline (Coco) Kaleel from James Tompkin's demo on live Fourier transformation 
        live video code
    """

    wn = "Viola Jones Algorithm Running!"
    im = 0
    videoOn = True

    def __init__(self, videoOn):
        self.videoOn = videoOn
        # Camera device
        # If you have more than one camera, you can access them by cv2.VideoCapture(1), etc.
        self.vc=None;
        
        if self.videoOn:
            self.vc = cv2.VideoCapture(0)
        
        if self.videoOn and not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.videoOn = False

        if self.videoOn == False:
            # No camera!
            self.gray = cv2.cvtColor(io.imread('coco_with_thumbs_cropped.png'), cv2.COLOR_BGR2GRAY)
            self.im = rgb2gray(img_as_float(io.imread('coco_with_thumbs_cropped.png'))) # One of our intrepid TAs (Yuanning was one of our HTAs for Spring 2019)
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)

        # run algorithms on static image
        if not self.videoOn:
            boundingBoxDims = cheat.cheat_face_detection((255*self.gray).astype(np.uint8))

            self.im = self.overlay_interest_points(boundingBoxDims, self.im)
            self.im = self.add_rect(boundingBoxDims, self.im)

            cv2.imshow(self.wn, (np.fliplr(self.im)*255).astype(np.uint8))
            cv2.waitKey(0)
            return

        # Main loop
        while True:
            a = time.perf_counter()
            self.camimage_vj()
            # call the algorithm on the image
            # boundingBoxDims = vj.viola_jones( self.im )
            # boundingBoxDims = [[10, 10, 100, 100]]
            boundingBoxDims = cheat.cheat_face_detection((255*self.gray).astype(np.uint8))

            #TODO remove: chris and emily's implementation shouldn't have this issue
            boundingBoxDims = self.temporarily_scoot_boundingBox(boundingBoxDims)

            if (len(boundingBoxDims) != 0):
                self.im = self.overlay_interest_points(boundingBoxDims, self.im)

                #overlay the image with a red, 2px thick rectangle of viola jones shape

                self.im = self.add_rect(boundingBoxDims, self.im)

                self.klt(boundingBoxDims)
                
                ##press key when ready for algorithm to continue
                cv2.waitKey(0)

            cv2.imshow(self.wn, (np.fliplr(self.im)*255).astype(np.uint8)) # faster alternative
            
            ## TODO implement object tracking algorithm so this can go back down to 1 ms
            cv2.waitKey(100)
            print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))
    
    
        if self.videoOn:
            # Stop camera
            self.vc.release()

    def klt(self, initialBoundingBoxDims):
        '''
        Do tracking with SIFT for a while, then return to object detection loop.
        '''
        # TODO fix so that it works for multiple faces
        xs1, ys1 = get_interest_points(self.im, initialBoundingBoxDims[0])

        # bounding box for whole image
        whole_im_bb = [0,0,int(self.im.shape[1]),int(self.im.shape[0])]
        for i in range(0, 100):

            self.camimage_vj()
            xs2, ys2 = get_interest_points(self.im,whole_im_bb)

            self.im = self.overlay_interest_points([whole_im_bb], self.im)
            self.im = self.add_rect(initialBoundingBoxDims, self.im)

            cv2.imshow(self.wn, (np.fliplr(self.im)*255).astype(np.uint8)) # faster alternative
    
            cv2.waitKey(100)


    def overlay_interest_points(self, boundingBoxDims, image):
        '''
        Adds interest points to the image based on the Harris Corner Detection algorithm in tracking.py, which is
        adapted from Homework 2.
        Params:
         - boundingBoxDims: [x, y, width, height] of bounding box
         - image: the COMPLETE uncroppped image that in theory contains a face
        Returns:
         - the image with red dots overlayed on the interest points
        '''
        if image is None:
            return image
        
        temp_image = image
        for (x,y,width,height) in boundingBoxDims:
            xs, ys = get_interest_points(image, [x,y,width,height])

            for i in range(len(xs)):
                temp_image = cv2.circle(image, (xs[i],ys[i]), radius=1, color=(255, 255, 0), thickness=-1)
        return temp_image

    def temporarily_scoot_boundingBox(self, boundingBoxDims):
        # TODO remove when we implement chris and emily's version
        temp_array = boundingBoxDims
        for x in range(len(boundingBoxDims)):
            width = boundingBoxDims[x][2]
            if (boundingBoxDims[x][0]-int(0.5*width)>=0):
                temp_array[x] = [boundingBoxDims[x][0]-int(0.5*width),boundingBoxDims[x][1],width,boundingBoxDims[x][3]]
            else:
                temp_array[x]=boundingBoxDims[x]
        return temp_array

    def add_rect(self, boundingBoxDims, image):
        '''
        Adds the rectangle specified in the dimensions to the image.
        Params:
         - boundingBoxDims: m*4 array of bounding rectangles: [[x,y,width,height],[x,y,width,height],...]
         - the original image
        Returns:
         - image
        '''
        temp_image = image
        for (x,y,width,height) in boundingBoxDims:
            temp_image = cv2.rectangle(temp_image, (x,y), (x+width,y+height), (255,0,0), 2)
        return temp_image



    def camimage_vj(self):
        '''
        In the case that the videoCamera should be on, read in the image from the video camera
        '''
        if self.videoOn:
            # Read image
            rval, origImage = self.vc.read()
            # Convert to grayscale and crop to square
            # (not necessary as rectangular is fine; just easier for didactic reasons)
            im = img_as_float(rgb2gray(origImage))
            self.gray = cv2.cvtColor(origImage, cv2.COLOR_BGR2GRAY)
            # Note: some cameras across the class are returning different image sizes
            # on first read and later on. So, let's just recompute the crop constantly.
            
            if im.shape[1] > im.shape[0]:
                cropx = int(3*(im.shape[1]-im.shape[0])/4)
                cropy = 0
            elif im.shape[0] > im.shape[1]:
                cropx = 0
                cropy = int(3*(im.shape[0]-im.shape[1])/4)

            self.im = im[cropy:im.shape[0]-cropy, cropx:im.shape[1]-cropx] 

        # Set size
        width = self.im.shape[1]
        height = self.im.shape[0]
        cv2.resizeWindow(self.wn, width*2, height*2)

        return


if __name__ == '__main__':
    main()