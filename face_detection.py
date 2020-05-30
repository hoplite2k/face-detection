import cv2
from imutils.video import VideoStream
import imutils
import numpy as np
import time
import os
#path for model and prototxt of pre-trained caffemodel
modelpath = './pre_trained_models/model/res10_300x300_ssd_iter_140000.caffemodel'
prototxtpath = './pre_trained_models/prototxt/deploy.prototxt.txt'

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxtpath, modelpath)

print("[INFO] starting video stream...")
cam = VideoStream(0).start()
time.sleep(1)

while(True):
    #one frame at a time
    frame = cam.read()
    #get frames
    frame = cv2.flip(frame, +1)
    frame = imutils.resize(frame, width=680)
    (h,w) = frame.shape[:2]

    #create blobs of frame(input pre-processing)
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300,300)), 1.0, (300,300), (104.0, 177.0, 123.0))

    #feed blob as input to the model and get detections
    net.setInput(blob)
    detections = net.forward()

    #bounding box
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence < 0.5:
            continue
        box = detections[0, 0, i, 3:7] * np.array([w,h,w,h])
        (x,y,ex,ey) = box.astype("int")
        cv2.rectangle(frame, (x,y), (ex,ey), (0,0,255), 2)

    #display screen
    cv2.imshow("Cam", frame)
    key = cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
cam.stop()
