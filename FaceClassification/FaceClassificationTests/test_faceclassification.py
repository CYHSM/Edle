from FaceClassification.faceclassification import getFeatureVector, getFeatureVectors
from FaceClassification.faceclassification import classifyPretrainedFeatures
from FaceClassification.faceclassification import convertMatlabToNumpy
import os
from pathlib import Path
from sklearn.externals import joblib
import numpy as np

# 1.) Check if saved, else load from Matlab
reload = True
baseDir = os.path.dirname(__file__)
featuresFn = baseDir+'/ClassifierData/features.pkl'
labelsFn = baseDir+'/ClassifierData/labels.pkl'
if os.path.isfile(featuresFn) and os.path.isfile(labelsFn) and not reload:
    features = joblib.load(featuresFn)
    labels = joblib.load(labelsFn)
else:
    #2.) Get Features from Tensorflow
    testImageFolder = '/home/marx/Dropbox/Privat/Programming/PrivateProj/Medium/CNNFace/faces'
    
    # 2.) Load directly from Matlab
    baseDirParent = Path(baseDir).parent
    #testImageFolder = os.path.join(str(baseDirParent), 'FaceDetectionTests/testImages')
    testImageFolder = '/home/marx/Dropbox/Privat/Programming/PrivateProj/Medium/CNNFace/faces'
    features, labels = getFeatureVectors(testImageFolder)
    features = convertMatlabToNumpy(features)
    labels = np.squeeze(convertMatlabToNumpy(labels))
    joblib.dump(features, featuresFn)
    joblib.dump(labels, labelsFn)

# 3.) Try to classify
classifyPretrainedFeatures(features,labels)


###########################################################
#####################OLD###################################
###########################################################

# 2.) Load Images from Folder
# featureMatrix = []
# baseDir = os.path.dirname(__file__)
# baseDir = Path(baseDir).parent
# testImageFolder = os.path.join(str(baseDir),'FaceDetectionTests/testImages')
# for folder in os.listdir(testImageFolder):
#     print(folder)
#     testNameFolder = os.path.join(testImageFolder,folder)
#     for image in os.listdir(testNameFolder):
#         image_file = os.path.join(testNameFolder,image)
#         print(image_file)
#         features = getFeatureVector(image_file)
#         featureMatrix.append(features)