import os
import cv2
from PIL import Image
from ObjectDetector import Detector
#from Statistics import *

# frame
currentframe = 1
object_detector = Detector()

# Read the video from specified path
cam = cv2.VideoCapture("C:\\Users\\User\\Desktop\\Experiment\\VIRAT.mp4")   # VIRAT
cam.set(cv2.CAP_PROP_FPS, 30)
fps = cam.get(cv2.CAP_PROP_FPS)

while True:
    # reading from frame
    ret, frame = cam.read()
    if ret:
        frame = Image.fromarray(frame, 'RGB')
        # video is still left continue creating images
        object_detector.ROI(frame)
        currentframe += 1
    else:
        break

# Release all space and windows once done
object_detector.release()
cam.release()
cv2.destroyAllWindows()