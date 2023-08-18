import cv2
import numpy as np
import matplotlib.pyplot as plt

def DetectionHumanFace():
    img = cv2.imread('res/img001.png')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade='data/haarcascade_frontalface_default.xml'
    face=  cv2.CascadeClassifier(face_cascade)

    eye_cascade='data/haarcascade_eye.xml'
    eye= cv2.CascadeClassifier(eye_cascade)
    faces=face.detectMultiScale(gray)

    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_eye=gray[y:y+h,x:x+w]
        eyes=eye.detectMultiScale(roi_eye)
        for (ex,ey,ew,eh) in eyes:
            cv2.circle(img[y:y+h,x:x+w],(int(ex+ew/2),int(ey+eh/2)),int(max(ew,eh)/2),(0,255,0),2)
    return img

def DetectionCatFace():
    img = cv2.imread('res/cat002.jpg')
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    face_cascade='data/haarcascade_frontalcatface.xml'
    face=  cv2.CascadeClassifier(face_cascade)

    faces=face.detectMultiScale(gray)
    print('Get null in here')
    for x,y,w,h in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    return img

cv2.imshow('face',DetectionCatFace())
cv2.waitKey(0)

