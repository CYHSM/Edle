"""Main script for testing and starting this project"""

import os.path
from pathlib import Path
import numpy as np
import scipy.misc
import cv2
from sklearn.externals import joblib
#My Modules
from Edle.util import util
from Edle.facedetection import facedetection as fd
from Edle.faceclassification import faceclassification as fc
from Edle.faceclassification import retrain

# 1.1) Define Paths
DATA_DIR,CLASSIFIER_DIR,CLASSIFIER_DIR_LABELS,INCEPTION_MODEL_DIR,FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,FACE_DETECTED_IMAGES_DIR,FACE_DETECTED_IMAGES_LABELS_DIR = util.get_absolute_paths() 
#------------------------------(.1.1)

# 1.2) Define Variables
retrain_classifier = True
use_face_detected_images = True
redetect_faces = False
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
    picture_folder = FACE_DETECTED_IMAGES_DIR

else:
    picture_folder = FULL_IMAGES_DIR
#------------------------------(.2)


# 3.) Retrain classifier with features from inception model (tensorflow)
if retrain_classifier:
    retrain.main(picture_folder, CLASSIFIER_DIR, CLASSIFIER_DIR_LABELS, steps=1000)
else:
    clf = fc.load_best_classifier(CLASSIFIER_DIR)
#------------------------------(.3)

