"""Runs Edle in live mode with a webcam"""

from imutils.video import WebcamVideoStream
from imutils.video import FPS
import cv2
#My Modules
from Edle.util import util
from Edle.faceclassification.threaded_faceclassification import ThreadedFaceClassification


# 1.1) Define Paths and get unique labels
DATA_DIR,CLASSIFIER_DIR,INCEPTION_MODEL_DIR,VGG_FACE_MODEL_DIR,FULL_IMAGES_DIR,FULL_IMAGES_LABELS_DIR,FACE_DETECTED_IMAGES_DIR,FACE_DETECTED_IMAGES_LABELS_DIR = util.get_absolute_paths() 
unique_labels = util.get_unique_labels(FACE_DETECTED_IMAGES_LABELS_DIR)
#------------------------------(.1.1)


# 2.) Start WebcamVideoStream (from imutils module)
vs = WebcamVideoStream(src=0).start()
fps = FPS().start()
#------------------------------(.2)

# 2.1) Start ThreadedFaceClassification
tfc = ThreadedFaceClassification(classifier_path=CLASSIFIER_DIR,unique_labels=unique_labels,inception_path=INCEPTION_MODEL_DIR)
#------------------------------(.2.1)


# 2.) Start loop and continously acquire frames which are then processed by the thread 
while fps._numFrames < 5000:
    # grab the frame from the threaded video stream
    frame = vs.read()
    #frame = cv2.resize(frame, (299, 299)) 
    
    #Classify
    if tfc.ready:
        tfc.start(frame)
        print(tfc.result)
        print(tfc.result_proba)
    
    # check to see if the frame should be displayed to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # update the FPS counter
    fps.update()
#------------------------------(.2)
    
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
