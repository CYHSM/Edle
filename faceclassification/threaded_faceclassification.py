# import the necessary packages
from threading import Thread
import cv2
import numpy as np
# My Modules
from Edle.facedetection import facedetection as fd
from Edle.faceclassification import faceclassification as fc


class ThreadedFaceClassification:

    def __init__(self, unique_labels, classifier_path, inception_path):
        # initialize the Threaded Face Classification class
        self.success = False
        self.result = []
        self.result_proba = []
        self.image = []
        self.unique_labels = unique_labels
        self.classifier_path = classifier_path
        self.ready = True
        self.clf = []
        self.inception_path = inception_path

        # Load Classifier and init graph
        self.clf = fc.load_best_classifier(classifier_path)
        self.t = Thread(target=self.classify, args=())
        self.t.start()
    def start(self, image):
        self.ready = False  # set to false during calculation
        self.image = image
        self.classify()
        return self

    def classify(self):
        if isinstance(self.image,np.ndarray):
            y_pred_index, result_label, y_pred = fc.classify_new_image(
                self.image, self.clf, unique_labels=self.unique_labels)
        else:
            fc.load_vgg_graph()  # For better performance load beforehand
            y_pred_index = None
        if y_pred_index is None:
            self.sucess = False
            self.ready = True
        else:
            self.ready = True
            self.success = True
            self.result = result_label
            self.result_proba = y_pred

    def read(self):
        # return the frame most recently read
        return self.result
