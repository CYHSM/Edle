"""Implements the classification of a face based on the vgg-face"""

from sklearn import cross_validation, svm, preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import numpy as np
import os.path
import tensorflow as tf
import glob
from time import time

##########################################################
#######Get Features from Inception model with TF##########
##########################################################
def get_feature_vector(image_path):
    """Gets the feature vector from the second to last layer of the inception dnn model.
    
    Args: 
        image_path : Path to image or to a folder with images or list of image paths
    
    Returns:
        Feature Vector for this image
    """
    # Define Variables and Constants
    MODEL_PATH = '/home/marx/Documents/GitHubProjects/Edle/FaceClassification/InceptionModel'
     
    #1.) Create inception graph from .pb file
    inception_path = os.path.join(MODEL_PATH,'classify_image_graph_def.pb')
    with tf.gfile.FastGFile(inception_path,'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
        
    def return_feature_from_session(tensor_name,image_data):
        #3.) Run session and get feature
        with tf.Session() as sess:
            next_to_last_tensor = sess.graph.get_tensor_by_name(tensor_name)
            next_to_last_feature_vector = sess.run(next_to_last_tensor,
                               {'DecodeJpeg/contents:0': image_data})
            next_to_last_feature_vector = np.squeeze(next_to_last_feature_vector)
            print(next_to_last_feature_vector)
            return next_to_last_feature_vector
    
    #2.) Get Data from image or folder
    next_to_last_feature_vector = []
    #If List
    if isinstance(image_path,list):        
        for p in image_path:
            image_data = tf.gfile.FastGFile(p, 'rb').read()
            this_feature_vector = return_feature_from_session('pool_3:0',image_data)
            next_to_last_feature_vector.append(this_feature_vector)
        next_to_last_feature_vector = np.vstack(next_to_last_feature_vector)
    #If Path
    elif os.path.isfile(image_path):
        image_data = tf.gfile.FastGFile(image_path, 'rb').read()
        next_to_last_feature_vector = return_feature_from_session('pool_3:0',image_data)
    #If Folder
    elif os.path.isdir(image_path):
        for p in glob.glob(image_path+'/*.jpg'):
            image_data = tf.gfile.FastGFile(p, 'rb').read()
            this_feature_vector = return_feature_from_session('pool_3:0',image_data)
            next_to_last_feature_vector.append(this_feature_vector)
        next_to_last_feature_vector = np.vstack(next_to_last_feature_vector)
    else:
        tf.logging.fatal('File does not exist %s', image_path)    
            
        
    #4.) Return features
    return next_to_last_feature_vector
        


##########################################################
##############Classification of Features##################
##########################################################
def classify_features(features, labels, unique_labels=[]):
    """Classify the features from the pretrained vgg dnn model"""

    # 1.) Standardize
    features = preprocessing.scale(features.astype(float))
                
    # 2.) Perform Grid Search         
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(svm.SVC(kernel='rbf', class_weight='balanced',probability=True), param_grid)
    clf = clf.fit(features, labels)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    #print(clf.predict_proba(features))

    y_pred = clf.predict(features)
    y_test = labels
    if not unique_labels:
       unique_labels = range(np.max(labels))
       unique_labels = ["{:01d}".format(x) for x in unique_labels]
    print(classification_report(y_test, y_pred, target_names=unique_labels))
    print(confusion_matrix(y_test, y_pred, labels=range(len(unique_labels)+1)))
    
    # 4.) Return classifier scores
    return clf

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