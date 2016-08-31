#Imports
from FaceDetection.facedetection import detectFaceCascade,detectFaceDlib
import os

# 1.) Load Images from Folder
baseDir = os.path.dirname(__file__)
testImageFolder = os.path.join(baseDir,'testImages')
for folder in os.listdir(testImageFolder):
    print(folder)
    if folder != 'andi':
        testNameFolder = os.path.join(testImageFolder,folder)
        for image in os.listdir(testNameFolder):
            image_file = os.path.join(testNameFolder,image)
            print(image_file)
            #detectFaceCascade(image_file)
            dets, scores, idx = detectFaceDlib(image_file, True)