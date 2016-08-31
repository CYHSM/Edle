"""Detect Face Methods."""
#  Copyright 2016-present Markus Frey

import numpy as np
import dlib
import cv2
import os
import time
import sys


def detectFaceCascade(inputImgPath,showit):
    """Detects the face in a picture with help of haarcascades

    See also:
        http://docs.opencv.org/3.1.0/d7/d8b/tutorial_py_face_detection.html
    """
    # 1.) Get pre-trained XML classifier for face and eye
    opencvpath = '/home/marx/miniconda3/envs/Python3.5.1/share/OpenCV/haarcascades'
    face_cascade = cv2.CascadeClassifier(os.path.join(opencvpath, 'haarcascade_frontalface_default.xml'))
    #eye_cascade = cv2.CascadeClassifier(os.path.join(opencvpath, 'haarcascade_eye.xml'))
    profile_cascade = cv2.CascadeClassifier(os.path.join(opencvpath, 'haarcascade_profileface.xml'))

    # 2.) Read Image and convert to grayscale
    img = cv2.imread(inputImgPath)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 3.) Detect Face and return ROI
    # scaleFactor – Parameter specifying how much the image size is reduced at each image scale.
    # minNeighbors – Parameter specifying how many neighbors each candidate rectangle should have to retain it.
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30,30))
    facesProfile = profile_cascade.detectMultiScale(gray, scaleFactor=1.1,
                                          minNeighbors=5, minSize=(30,30))

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        # roi_gray = gray[y:y+h, x:x+w]
        # roi_color = img[y:y+h, x:x+w]
        # eyes = eye_cascade.detectMultiScale(roi_gray)
        # for (ex,ey,ew,eh) in eyes:
        #     cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    for (x,y,w,h) in facesProfile:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)

    if showit:
        cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('img',img)
        cv2.waitKey(0)

    return faces


def detectFaceDlib(inputImgPath,showit):
    # 1.) Get dlib Detector and read image
    detector = dlib.get_frontal_face_detector()
    img = cv2.imread(inputImgPath)
    img = cv2.resize(img, (0,0), fx=0.3, fy=0.3)

    # 2.) Get Faces
    dets, scores, idx = detector.run(img)

    #3.) Show
    for i, d in enumerate(dets):
        print("Detection {}, score: {}, face_type:{}".format(
            d, scores[i], idx[i]))
        l,t,r,b = d.left(),d.top(),d.right(),d.bottom()
        cv2.rectangle(img, (l, b), (r,t), (0, 255, 0), 2)

    if showit:
        cv2.namedWindow('img2', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('img2', img)
        cv2.waitKey(0)

    return dets,scores,idx
