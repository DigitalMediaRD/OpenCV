import cv2
import numpy as np
img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, img2 = cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY)  # 二值化阈值处理
c, h = cv2.findContours(img2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 查找轮廓
ep=[0.1,0.05,0.01]

arcl=cv2.arcLength(c[0],True)
img3 = np.zeros(img.shape, np.uint8) + 255
img3=cv2.drawContours(img3,c,-1,(0,0,255),2)
cv2.imshow('DrawContours', img3)

def Moments():

    for n in range (len(c)):
        m=cv2.moments(c[n])
        print('轮廓%s的矩'%n,m)
        print('轮廓%s的面积' % n, m['m00'])

def Length():
    for n in range (len(c)):
        m=cv2.arcLength(c[n],True)
        print('轮廓%s的长度'%n,m)


def Area():
    for n in range (len(c)):
        m=cv2.contourArea(c[n])
        print('轮廓%s的面积'%n,m)


def PolyDP():
    for n in range (3):
        eps=ep[n]*arcl
        img4=img3.copy()
        app=cv2.approxPolyDP(c[0],eps,True)
        img4=cv2.drawContours(img4,[app],-1,(100,20,55),2)
        cv2.imshow('appro %.2f '%ep[n], img4)




def ConveHull():
    hull=cv2.convexHull(c[1])
    print('True',hull)
    hull2 = cv2.convexHull(c[1],returnPoints=False)
    print('False',hull2)
    cv2.polylines(img3,[hull],True,(255,0,0),2)
    cv2.imshow('ConveHull',img3)



def Rect():
    ret=cv2.boundingRect(c[0])
    print(ret)
    pt1=(ret[0],ret[1])
    pt2 = (ret[0]+ret[2], ret[1]+ret[3])
    cv2.rectangle(img3,pt1,pt2,(255,0,0),2)
    cv2.imshow('ConveHull',img3)

def DrawAreaRect():
    ret=cv2.minAreaRect(c[0])
    rect=cv2.boxPoints(ret)
    rect=np.int0(rect)
    cv2.drawContours(img3,[rect],-1,(0,0,255),2)
    cv2.imshow('DrawAreaRect',img3)
DrawAreaRect()
cv2.waitKey(0)
cv2.destroyAllWindows()