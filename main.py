"""Main script for testing and starting this project"""

import os.path
from pathlib import Path
import numpy as np
import scipy.misc
import cv2
from sklearn.externals import joblib
import tensorflow as tf
#My Modules
from Edle.util import util
from Edle.facedetection import facedetection as fd
from Edle.faceclassification import faceclassification as fc
from sklearn import cross_validation, svm, preprocessing
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
# 1.1) Define Paths
DATA_DIR,CLASSIFIER_DIR,INCEPTION_MODEL_DIR,VGG_FACE_MODEL_DIR,FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,FACE_DETECTED_IMAGES_DIR,FACE_DETECTED_IMAGES_LABELS_DIR = util.get_absolute_paths()
#------------------------------(.1.1)

# 1.2) Define Variables
retrain_classifier = True
use_face_detected_images = True
redetect_faces = False
#------------------------------(.1.2)

# 2.) Extract Faces from Images
if use_face_detected_images:
    if redetect_faces:
        filenames, texts, labels, unique_labels = util._find_image_files(FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,shuffle=False)
        for fn in filenames:
            cropped_image,dets,scores,idx = fd.detect_face_dlib(fn)
            if cropped_image is not None:
                this_cropped_path = os.path.join(FACE_DETECTED_IMAGES_DIR,Path(fn).parts[-2],Path(fn).parts[-1])
                cv2.imwrite(this_cropped_path, cropped_image)
                print('Found face in ',fn)
            else:
                print('Did not find face in ',fn,'...skipping!')
    filenames, texts, labels, unique_labels = util._find_image_files(FACE_DETECTED_IMAGES_DIR,FACE_DETECTED_IMAGES_LABELS_DIR,shuffle=False)

else:
    filenames, texts, labels, unique_labels = util._find_image_files(FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,shuffle=False)
#------------------------------(.2)

fc.load_vgg_graph()
graph = tf.get_default_graph()
#graph.get_tensor_by_name('import/data:0')
#ops = graph.get_operations()
#ops[0].name
# 3.) Retrain classifier with features from inception model (tensorflow)
if retrain_classifier:
    #Get Features
    #fc.load_graph(VGG_FACE_MODEL_DIR) #For better performance load beforehand
    features,f2,f3 = fc.get_feature_vector_from_vgg(filenames, VGG_FACE_MODEL_DIR)
    #Train Classifier
    clf, _ = fc.get_best_classifier(features, labels, unique_labels=unique_labels, save=True, classifier_path=CLASSIFIER_DIR)
else:
    features,f2,f3 = fc.get_feature_vector_from_vgg(['/home/marx/Pictures/bla.jpg'], VGG_FACE_MODEL_DIR)
    clf = fc.load_best_classifier(CLASSIFIER_DIR)
    y_pred = clf.predict_proba(features)
    print(y_pred)
#------------------------------(.3)

#TESTS
v = '/home/marx/Pictures/bla.jpg'
features_test,f2t,f3t = fc.get_feature_vector_from_vgg([v], VGG_FACE_MODEL_DIR)
v2 = '/home/marx/Documents/GitHubProjects/Edle/data/images/facedetected/markus/20160402_151425.jpg'
features_test2,f2t2,f3t2 = fc.get_feature_vector_from_vgg([v2], VGG_FACE_MODEL_DIR)

features_standardized = preprocessing.scale(features.astype(float),axis=1)
features_standardized = f2
np.std(features[2])

clf = svm.SVC(C=10, kernel='linear', probability=True)
clf.fit(features_standardized,labels)

y_pred = clf.predict(features_standardized)
y_pred
y_test = labels
print(classification_report(y_test, y_pred, target_names=unique_labels))


clf.predict(f2t)
clf.predict_proba(f2t)

len(features[0])
np.std(features_standardized[0])
labels
%matplotlib qt
plt.plot(features[1])
plt.show()
np.max(features[0])
np.argmax(features[0])
