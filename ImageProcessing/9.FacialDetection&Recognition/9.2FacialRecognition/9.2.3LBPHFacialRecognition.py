import cv2
import numpy as np
img11=cv2.imread('res/xl11.jpg',0)
img12=cv2.imread('res/xl12.jpg',0)
img13=cv2.imread('res/x113.jpg',0)
img21=cv2.imread('res/x121.jpB',0)
img22=cv2.imread('res/x122.jpg',0)
img23=cv2.imread( 'res/xl23.jpg',0)

train_images=[img11,img12,img13,img21,img22,img23]
labels=np. array([0,0,0,1,1,1])
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.train(train_images, labels)
testimg=cv2.imread( 'test2.jpg ',0)
label, confidence=recognizer.predict(testimg)
print('匹配序号',label)
print('可信度',confidence)