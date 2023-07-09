import cv2
import numpy as np
img=cv2.imread('res/img001.png')
cv2.imshow('show',img)
height=img.shape[0]
width=img.shape[1]
dsize=(width,height)


def Perspective():
    input = np.float32([[0, 0], [width - 10, 0], [0, height - 10], [width-1, height - 1]])
    output = np.float32([[50, 50], [width - 50, 80], [50, height - 100], [width-100, height - 10]])
    m=cv2.getPerspectiveTransform(input,output)
    img2 = cv2.warpPerspective(img, m, dsize)
    cv2.imshow('Perspective', img2)
    cv2.waitKey(0)


Perspective()