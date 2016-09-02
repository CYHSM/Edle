"""Main script for testing and starting this project"""

import os.path
from pathlib import Path
import numpy as np
import scipy.misc
import cv2
#My Modules
from Edle.util import util
from Edle.facedetection import facedetection as fd

# 1.1) Define Paths
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR,'data')
CLASSIFIER_DIR = os.path.join(DATA_DIR,'classifier/clf.pkl')
INCEPTION_MODEL_DIR = os.path.join(DATA_DIR,'inceptionmodel','classify_image_graph_def.pb')
FULL_IMAGES_DIR = os.path.join(DATA_DIR,'images/full')
FULL_IMAGES_LABELS_DIR = os.path.join(FULL_IMAGES_DIR,'labels')
FACE_DETECTED_IMAGES_DIR = os.path.join(DATA_DIR,'images/facedetected')
#------------------------------(.1.1)

# 1.2) Define Variables
retrain_classifier = True
use_face_detected_images = True
redetect_faces = True
#------------------------------(.1.2)


# 2.) Extract Faces from Images
if use_face_detected_images:
    if redetect_faces:
        filenames, texts, labels, unique_labels = util._find_image_files(FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR)
        for fn in filenames:
            cropped_image,dets,scores,idx = fd.detect_face_dlib(fn)
            if cropped_image is not None:
                this_cropped_path = os.path.join(FACE_DETECTED_IMAGES_DIR,Path(fn).parts[-2],Path(fn).parts[-1])
                cv2.imwrite(this_cropped_path, cropped_image)
                print('Found face in ',fn)
            else:
                print('Did not find face in ',fn,'...skipping!')
    else:
        filenames, texts, labels, unique_labels = util._find_image_files(FACE_DETECTED_IMAGES_DIR)
#------------------------------(.2)
