# Importing all necessary libraries
# %matplotlib inline
import cv2
import numpy as np
from scipy.spatial import distance
from yolo import YOLO

out = cv2.VideoWriter('Covid19Analyser.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (1280, 720))


class Detector:
    def __init__(self):
        self.yolo_obj = YOLO()

    def initialize(self):
        self.safecount = 0
        self.unsafecount = 0
        self.boundings = list()
        self.centroid = list()

    def ROI(self, frame):
        self.initialize()
        ori_frame = np.asarray(frame)
        image, BoundingList = self.yolo_obj.detect_image(frame)
        l = [a['class'] for a in BoundingList]
        self.totalcount = l.count('person')

        if self.totalcount == 0:
            self.boundary(ori_frame)
            cv2.imshow('COVID-19 analyser', ori_frame)
            out.write(ori_frame)
            cv2.waitKey(10)
        else:
            for Bounding_box in BoundingList:
                bound = list()
                if Bounding_box['class'] == 'person':
                    top = Bounding_box['top']
                    bottom = Bounding_box['bottom']
                    left = Bounding_box['left']
                    right = Bounding_box['right']
                    bound.append(left)
                    bound.append(top)
                    bound.append(right)
                    bound.append(bottom)
                    self.cal_centroid(bound)
            self.show_video(ori_frame)

    def cal_centroid(self, bound):
        cen = (int((bound[0] + bound[2]) / 2), int((bound[1] + bound[3]) / 2))
        self.centroid.append(cen)
        self.boundings.append(bound)

    def show_video(self, img):
        for i in range(len(self.boundings)):
            startcentroid = self.centroid[i]
            startnode = self.boundings[i]
            j = i
            while j < len(self.boundings):
                col = self.cal_distance(startcentroid, self.centroid[j])
                if col == (0, 255, 0):
                    cv2.line(img, startcentroid, self.centroid[j], col, 1)
                    self.safecount += 1
                elif col == (0, 0, 255):
                    cv2.line(img, startcentroid, self.centroid[j], col, 1)
                    self.unsafecount += 1
                else:
                    col = (0, 0, 255)
                cv2.rectangle(img, (startnode[0], startnode[1]), (startnode[2], startnode[3]), col, 2)
                cv2.rectangle(img, (self.boundings[j][0], self.boundings[j][1]),
                              (self.boundings[j][2], self.boundings[j][3]), col, 2)
                j += 1

        self.boundary(img)
        cv2.imshow('COVID-19 analyser', img)
        out.write(img)
        cv2.waitKey(10)

    def cal_distance(self, start, end):
        if start == end:
            return 0
        else:
            d = distance.euclidean(start, end)
            if np.sqrt(d) > 12:
                return (0, 255, 0)
            else:
                return (0, 0, 255)

    def boundary(self, img):
        cv2.putText(img, "Social Distancing Analyser (COVID-19)", (300, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.rectangle(img, (20, 60), (540, 140), (0, 0, 0), 2)
        cv2.putText(img, "Colour of Bounding box shows severity of risk and", (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 186, 244), 2)
        cv2.putText(img, "Connecting lines shows distance between people", (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 186, 244), 2)
        cv2.putText(img, "GREEN: SAFE ", (100, 70 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "RED: UNSAFE", (300, 70 + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.rectangle(img, (700, 65), (1250, 100 + 40), (0, 0, 0), 2)
        cv2.putText(img, "Total number of people in frame: " + str(self.totalcount), (730, 20 + 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        if self.safecount != 0:
            self.safecount += 1
        if self.unsafecount != 0:
            self.unsafecount += 1
        cv2.putText(img, "Number of people maintaining safe distance: " + str(self.safecount), (730, 20 + 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(img, "Number of people having unsafe distance: " + str(self.unsafecount), (730, 20 + 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 150), 2)

    def release(self):
        out.release()
        print("Done!!")
