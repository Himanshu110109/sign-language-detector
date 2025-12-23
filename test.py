import cv2
import mediapipe as mp
import math
import numpy as np
import time
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow
from cvzone.ClassificationModule import Classifier
from cvzone.HandTrackingModule import HandDetector

cap = cv2.VideoCapture(0)
mpdraw = mp.solutions.drawing_utils
mpface = mp.solutions.face_mesh
mpmesh = mpface.FaceMesh()
detector = HandDetector()
dot_color = (0, 255, 0)
mesh_color = (0, 255, 0)
drawing_spec_dots = mpdraw.DrawingSpec(color=dot_color, thickness=1, circle_radius=2)
drawing_spec_mesh = mpdraw.DrawingSpec(color=mesh_color, thickness=1, circle_radius=1)
classifier = Classifier("model/keras_model.h5", "model/labels.txt")
offset = 20
imgSize = 300
folder = "data/B"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "s", "T", "U", "V", "W", "X", "Y", "Z"]
while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mpmesh.process(imgRGB)
    if faces.multi_face_landmarks:
        for facelms in faces.multi_face_landmarks:
            mpdraw.draw_landmarks(img, facelms, mpface.FACEMESH_TESSELATION, drawing_spec_dots, drawing_spec_mesh)
    imgoutput = img.copy()
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
            predictions, index = classifier.getPrediction(imgwhite, draw=False)
            print(predictions, index)
        else:
            k = imgSize / w
            hcal = math.ceil(k * h)
            imgResize = cv2.resize(imgcrop, (imgSize, hcal))
            imgResizeShape = imgResize.shape
            hgap = math.ceil((imgSize - hcal) / 2)
            imgwhite[hgap:hcal + hgap, :] = imgResize
            predictions, index = classifier.getPrediction(imgwhite, draw=False)
        cv2.rectangle(imgoutput, (x-offset, y-offset-50), (x-offset+50, y-offset-50+50), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgoutput, labels[index], (x, y-30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 2 )
        cv2.rectangle(imgoutput, (x-offset,y-offset), (x+w+offset, y+h+offset), (0, 255, 0), 4)
        # imgwhite[0:imgResizeShape[0], 0:imgResizeShape[1]] = imgResize
        # cv2.imshow("imgwhite", imgwhite)
        # cv2.imshow("imgcrop", imgcrop)
    cv2.imshow("detection", imgoutput)
    key = cv2.waitKey(1)