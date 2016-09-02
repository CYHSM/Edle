"""Implements the classification of a face based on the vgg-face"""

from sklearn import cross_validation, svm, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import datetime
import numpy as np
import os.path
import tensorflow as tf
import glob
import cv2
from time import time
from sklearn.externals import joblib
from pathlib import Path
#My Modules
from Edle.facedetection import facedetection as fd

##########################################################
#######Get Features from Inception model with TF##########
##########################################################
def load_inception_graph(MODEL_PATH):     
    #1.) Create inception graph from .pb file
    inception_path = MODEL_PATH
    with tf.gfile.FastGFile(inception_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

def get_feature_vector(image):
    """Gets the feature vector from the second to last layer of the inception dnn model.
    
    Args: 
        image : Path to image or to a folder with images or list of image paths or numpy image
    
    Returns:
        Feature Vector for this image
    """
  
    def return_feature_from_session(tensor_name,image_data):
        #3.) Run session and get feature
        with tf.Session() as sess:
            next_to_last_tensor = sess.graph.get_tensor_by_name(tensor_name)
            
            #Check if image from folder or as numpy array
            time_bf = datetime.datetime.now()
            if isinstance(image_data,np.ndarray):
                next_to_last_feature_vector = sess.run(next_to_last_tensor,
                               {'DecodeJpeg:0': image_data})
            else:
                next_to_last_feature_vector = sess.run(next_to_last_tensor,
                               {'DecodeJpeg/contents:0': image_data})
            time_af = datetime.datetime.now()
            print('Ran in ',(time_af-time_bf).total_seconds(),' seconds')
            next_to_last_feature_vector = np.squeeze(next_to_last_feature_vector)
            #print(next_to_last_feature_vector)
            return next_to_last_feature_vector
    
    #2.) Get Data from image or folder
    next_to_last_feature_vector = []
    #If List
    if isinstance(image,np.ndarray):
        next_to_last_feature_vector = return_feature_from_session('pool_3:0',image)
    elif isinstance(image,list):        
        for p in image:
            image_data = tf.gfile.FastGFile(p, 'rb').read()
            this_feature_vector = return_feature_from_session('pool_3:0',image_data)
            next_to_last_feature_vector.append(this_feature_vector)
        next_to_last_feature_vector = np.vstack(next_to_last_feature_vector)
    #If Path
    elif os.path.isfile(image):
        image_data = tf.gfile.FastGFile(image, 'rb').read()
        next_to_last_feature_vector = return_feature_from_session('pool_3:0',image_data)
    #If Folder
    elif os.path.isdir(image):
        for p in glob.glob(image+'/*.jpg'):
            image_data = tf.gfile.FastGFile(p, 'rb').read()
            this_feature_vector = return_feature_from_session('pool_3:0',image_data)
            next_to_last_feature_vector.append(this_feature_vector)
        next_to_last_feature_vector = np.vstack(next_to_last_feature_vector)

    else:
        tf.logging.fatal('File does not exist %s', image)    
            
        
    #4.) Return features
    return next_to_last_feature_vector
        

##########################################################
##############Classification of Features##################
##########################################################
def get_best_classifier(features, labels, unique_labels=[], save=True, classifier_path=[]):
    """Classify the features from the pretrained vgg dnn model"""

    # 1.) Standardize
    features = preprocessing.scale(features.astype(float))
                
    # 2.) Perform Grid Search         
    t0 = time()
    param_grid = {'C': [1e-1, 1, 1e1, 1e2, 1e3, 5e3, 1e4, 5e4, 1e5] }
    clf = GridSearchCV(svm.SVC(kernel='linear',probability=True), param_grid)
    clf = clf.fit(features, labels)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    clf_best = clf.best_estimator_
    print(clf_best)
    
    #print(clf.predict_proba(features))

    y_pred = clf_best.predict(features)
    y_test = labels
    if not unique_labels:
       unique_labels = range(np.max(labels))
       unique_labels = ["{:01d}".format(x) for x in unique_labels]
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    print(confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)+1)))
    
    #3.) Save Classifier
    #print(base_dir)
    if save:
        joblib.dump(clf_best, classifier_path)
    
    # 4.) Return classifier scores
    return clf_best, clf

def load_best_classifier(classifier_path):
    #1.) Load Classifier
    clf = joblib.load(classifier_path)
    #2.) Return Classifier
    return clf
    
def classify_new_image(image, clf, unique_labels = []):
    """Classifies a new image
    
    Args: 
        image : Path to image or to a folder with images or list of image paths or numpy image
    
    Returns:
        Classification result for this image
    """

    #1.) Detect Face
    cropped_image,dets,scores,idx = fd.detect_face_dlib(image)
    
    if cropped_image is None:
        print('Found no Face')
        return None,None,None
    else:
        print('Found a Face')
    
    #2.) Get feature vector / Can return a 2d array if image is folder with images
    features = get_feature_vector(cropped_image)
    
    #3.) Predict on new features
    y_pred = clf.predict_proba(features.reshape(1,-1))
    y_pred_index = np.argmax(y_pred)
    
    if unique_labels:
        result_label = unique_labels[y_pred_index]
    else:
        result_label = y_pred_index
        
    return y_pred_index, result_label, y_pred
    

##########################################################
#######Get Data from Deep Net with MATLAB API#############
##########################################################
def getFeatureVector_from_matlab(inputImgPath):
    """Gets the feature vector of layer 6 or 7 from matlab matconvnet"""

    # 1.) Start Matlab Engine
    eng = matlab.engine.start_matlab('-nojvm')

    # 2.) Go to path
    eng.cd('/home/marx/Dropbox/Privat/Programming/PrivateProj/Medium/CNNFaceUbuntu/getFeaturesFromVgg/')

    # 3.) Run m-file and extract features
    features = eng.getFeaturesFromVgg(inputImgPath)

    # 4.) Return
    return features

def getFeatureVectors_from_matlab(imgDir):
    """Gets the features from all images in imgDir"""

    # 1.) Start Matlab Engine
    eng = matlab.engine.start_matlab('-nojvm')

    # 2.) Go to path
    eng.cd('/home/marx/Dropbox/Privat/Programming/PrivateProj/Medium/CNNFaceUbuntu/getFeaturesFromVgg/')

    # 3.) Run m-file and extract features
    features, labels = eng.getFeaturesFromVggAll(imgDir, nargout=2)

    # 4.) Return
    return features, labels

##########################################################
##############HELPER FUCNTIONS############################
##########################################################
def convertMatlabToNumpy(matlabIn):
    np_a = np.array(matlabIn._data.tolist())
    np_a = np_a.reshape(matlabIn.size)
    np_a = np_a.reshape((matlabIn.size[1], matlabIn.size[0])).transpose()
    return np_a