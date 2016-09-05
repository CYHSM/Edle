"""Checks the vgg face dnn with the matlab implementation"""

import tensorflow as tf
import numpy as np
from skimage.transform import resize

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#My Modules
from Edle.faceclassification import faceclassification as fc
#%matplotlib inline

picture_path = '/home/marx/Documents/GitHubProjects/Edle/data/images/facedetected/markus/20160402_151407.jpg'
# image = mpimg.imread(picture_path)
# image[0,0,0]
# np.max(image)
# image_r = resize(image, (224, 224), preserve_range=True)
# plt.imshow(image)
# plt.show()
#
# image_r[0,4,0]
# image_r.shape
# mean_image = [129.1863, 104.7624, 93.5940]
# imagesub = image_r - mean_image
# imagesub[0,0,0]
# plt.imshow(imagesub)
# plt.show()
# np.min(imagesub)
# np.max(imagesub)

#
feature_python,_,_ = fc.get_feature_vector_from_vgg([picture_path])
print(feature_python[0])
print(feature_python[-100:])
gg = feature_python[0]
print(gg[0:10])
print(gg[0])
max(gg)
min(gg)
np.max(gg)
np.argmax(gg)


# feature_python2,_,_ = fc.get_feature_vector_from_vgg([picture_path])
# gg2 = feature_python2[0]
# print(gg2[0:10])
