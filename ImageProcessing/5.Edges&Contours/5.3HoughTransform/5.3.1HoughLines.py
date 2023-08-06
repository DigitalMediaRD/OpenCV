import cv2
import numpy as np
img = cv2.imread('res/img001.png')
cv2.imshow('original', img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges=cv2.Canny(gray,50,150,apertureSize=3)
lines=cv2.HoughLines(edges,1,np.pi/180,150)
linesP=cv2.HoughLinesP(edges,1,np.pi/180,150,minLineLength=100,maxLineGap=10)
img3=img.copy()
print(lines)

def HoughLine(lines):
    for line in lines:
        rho,theta=line[0]
        a=np.cos(theta)
        b=np.sin(theta)
        x0,y0=a*rho,b*rho
        pt1=(int(x0+1000*(-b)),int(y0+1000*(a)))
        pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
        cv2.line(img,pt1,pt2,(0,0,255),2)

def HoughLineP(lines):
    for line in lines:
        x1,y1,x2,y2=line[0]
        cv2.line(img3,(x1,y1),(x2,y2),(0,0,255),2)

HoughLine(lines)
#HoughLineP(linesP)
cv2.imshow('HoughLines',img3)
cv2.waitKey(0)
cv2.destroyAllWindows()
