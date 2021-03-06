""""Controls the video in and output"""

from imutils.video import WebcamVideoStream
from imutils.video import FPS
import imutils
import cv2
from Edle.faceclassification import faceclassification as fc
from threading import Thread

def capture_frame_from_camera():
    """Captures a frame from the webcam in a threaded environment"""
    
    #1.) Start WebcamVideoStream (from imutils module)
    vs = WebcamVideoStream(src=0).start()
    fps = FPS().start()

    #2.) Start thread for image processing
#    fc.load_inception_graph()
#    clf = fc.load_best_classifier()
#    frame = []
    #
    #ip = Thread(target=fc.classify_new_image,args=(frame,clf))
    
    # loop over some frames...this time using the threaded stream
    while fps._numFrames < 1000:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        #frame = imutils.resize(frame, width=400)
     
        #Classify
        y_pred_index, result_label, y_pred = fc.classify_new_image(frame,clf)
        print(y_pred)
        
        # check to see if the frame should be displayed to our screen
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
    
        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    
    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
        
capture_frame_from_camera()