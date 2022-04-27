# Local Feature Stencil Code
# Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech with Henry Hu <henryhu@gatech.edu>
# Edited by James Tompkin
# Adapted for python by asabel and jdemari1 (2019)

import csv
import cv2
import sys
import argparse
import time

import numpy as np
import scipy.io as scio

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from skimage import io, filters, feature, img_as_float32, img_as_float
from skimage.transform import rescale
from skimage.color import rgb2gray

import viola_jones as vj


def main():
    """
    Reads in the data,

    Command line usage: python main.py [-v | --video <"on" or "off" (no quotes/brackets)>]

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
    args = parser.parse_args()

    videoOn = (args.video == "on")

    if not videoOn:
        ## TODO: call viola_jones on static image
        print("camera off")
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
        self.vc = cv2.VideoCapture(0)
        if not self.vc.isOpened():
            print( "No camera found or error opening camera; using a static image instead." )
            self.videoOn = False

        if self.videoOn == False:
            # No camera!
            self.im = rgb2gray(img_as_float(io.imread('coco_with_thumbs_cropped.png'))) # One of our intrepid TAs (Yuanning was one of our HTAs for Spring 2019)
        else:
            # We found a camera!
            # Requested camera size. This will be cropped square later on, e.g., 240 x 240
            ret = self.vc.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
            ret = self.vc.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

        # Set the size of the output window
        cv2.namedWindow(self.wn, 0)

        # Main loop
        while True:
            a = time.perf_counter()
            self.camimage_vj()
            print('framerate = {} fps \r'.format(1. / (time.perf_counter() - a)))
    
    
        if self.videoOn:
            # Stop camera
            self.vc.release()


    def camimage_vj(self):
        
        if self.videoOn:
            # Read image
            rval, im = self.vc.read()
            # Convert to grayscale and crop to square
            # (not necessary as rectangular is fine; just easier for didactic reasons)
            im = img_as_float(rgb2gray(im))
            # Note: some cameras across the class are returning different image sizes
            # on first read and later on. So, let's just recompute the crop constantly.
            
            if im.shape[1] > im.shape[0]:
                cropx = int((im.shape[1]-im.shape[0])/2)
                cropy = 0
            elif im.shape[0] > im.shape[1]:
                cropx = 0
                cropy = int((im.shape[0]-im.shape[1])/2)

            self.im = im[cropy:im.shape[0]-cropy, cropx:im.shape[1]-cropx]

        # Set size
        width = self.im.shape[1]
        height = self.im.shape[0]
        cv2.resizeWindow(self.wn, width*2, height*2)

        # call the algorithm on the image
        imVJ = vj.viola_jones( self.im )
        
        cv2.imshow(self.wn, (imVJ*255).astype(np.uint8)) # faster alternative
        
        ## TODO implement object tracking algorithm so this can go back down to 1 ms
        cv2.waitKey(60)

        return


if __name__ == '__main__':
    main()