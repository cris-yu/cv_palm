#!/usr/bin/env python
#手掌识别
import cv2
import os

hand_Cascade = cv2.CascadeClassifier("cascade.xml")
hand_Cascade.load('xml/cascade.xml')
cap = cv2.VideoCapture(0)

while True:
    flag+=1
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rect = hand_Cascade.detectMultiScale(        #主要修改以下参数
        gray,
        scaleFactor=1.1,
        minNeighbors=70,
        minSize=(2,2),
        flags = cv2.IMREAD_GRAYSCALE
    )
    for (x, y, w, h) in rect:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


