import cv2
import mediapipe as mp
import math
import numpy as np
import time
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
detector = HandDetector()
offset = 20
imgSize = 300
folder = "data/M"
counter = 0
while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand["bbox"]
        imgwhite = np.ones((imgSize, imgSize, 3), dtype="uint8")*255
        imgcrop = img[y-offset:y+h+offset, x-offset:x+w+offset]
        imgcropshape = imgcrop.shape
        aspectratio = h / w
        if aspectratio > 1:
            k = imgSize / h
            wcal = math.ceil(k * w)
            imgResize = cv2.resize(imgcrop, (wcal, imgSize))
            imgResizeShape = imgResize.shape
            wgap = math.ceil((imgSize - wcal)/2)
            imgwhite[:, wgap:wcal+wgap] = imgResize
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgSize, hcal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgResize
        # imgwhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
        cv2.imshow("imgwhite", imgwhite)
        # cv2.imshow("imgcrop", imgcrop)
    cv2.imshow("detection", img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f"{folder}/{time.time()}.jpg", imgwhite)
        print(counter)