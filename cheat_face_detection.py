import os
import cv2
import numpy as np

def cheat_face_detection(image):
    '''
    Returns the cv2 version of the face detection results; use for approximating speed and object tracking
    '''
    
    cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(cascPathface)
    
    faces = faceCascade.detectMultiScale(image,
                                         scaleFactor=1.1,
                                         minNeighbors=5,
                                         minSize=(60, 60),
                                         flags=cv2.CASCADE_SCALE_IMAGE)

    return faces