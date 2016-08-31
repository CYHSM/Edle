"""Implements the classification of a face based on the vgg-face"""

import matlab.engine
from sklearn import cross_validation, svm, preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
import os.path
import tensorflow as tf

##########################################################
#######Get Features from Inception model with TF##########
##########################################################
def get_feature_vector(image_path):
    """Gets the feature vector from the second to last layer of the inception dnn model.
    
    Args: 
        image_path : Path to image or to a folder with images
    
    Returns:
        Feature Vector for this image
    """
    
    #1.) Get Data from image
    if not tf.gfile.Exists(image_path):
        tf.logging.fatal('File does not exist %s', image_path)
    image_data = tf.gfile.FastGFile(image_path, 'rb').read()
    
    #2.) Create inception graph from .pb file
    inception_path = os.path.join('classify_image_graph_def.pb')
    with tf.gfile.FastGFile(inception_path) as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    
    #3.) Run session and get feature
    with tf.Session() as sess:
        next_to_last_tensor = sess.graph.get_tensor_by_name('pool_3:0')
        next_to_last_feature_vector = sess.run(next_to_last_tensor,
                           {'DecodeJpeg/contents:0': image_data})
        next_to_last_feature_vector = np.squeeze(next_to_last_feature_vector)
        
    #4.) Return features
    return next_to_last_feature_vector
        


##########################################################
##############Classification of Features##################
##########################################################
def classify_pretrained_features(features, labels):
    """Classify the features from the pretrained vgg dnn model"""

    # 1.) Define classifier and standardize
    classifier = svm.SVC(kernel='linear', C=1)
    #features = StandardScaler().fit_transform(features)
    features = preprocessing.scale(features)

    # 2.) Perform cross validation
    scores = cross_validation.cross_val_score(classifier,features,labels, cv=cross_validation.LeaveOneOut(n=labels.size), scoring = 'accuracy')

    # 3.) Show results
    print(scores)


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