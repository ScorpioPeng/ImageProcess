import cv2
import numpy as np

from tracker import *

# Create tracker object

cap = cv2.VideoCapture("视频3.mp4")

# Object detection from Stable camera
object_detector = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=16)

while True:
    ret, frame = cap.read()
    height, width, _ = frame.shape


    # 1. Object Detection
    mask = object_detector.apply(frame)
    # _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    mask = cv2.erode(mask,np.ones((2,2), np.uint8),iterations=10)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=10)
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # # if contour is too small, just ignore it
        if cv2.contourArea(contour) < 50:
            continue

        # 计算最小外接矩形（非旋转）
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x+w, round(y+h*0.6)), (255,255,0), 2)


    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(30)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()