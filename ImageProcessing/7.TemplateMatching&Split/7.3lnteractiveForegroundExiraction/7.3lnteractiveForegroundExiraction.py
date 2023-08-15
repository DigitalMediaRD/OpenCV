import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('res/img002.png')
cv2.imshow('Original',img)
mask=np.zeros(img.shape[:2],np.uint8)
bg=np.zeros((1,65),np.float64)
fg=np.zeros((1,65),np.float64)
rect=(50,50,400,300)


def Original(img):
    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img=img*mask2[:,:,np.newaxis]
    cv2.imshow('grabCut',img)
    cv2.waitKey(0)

def EditedImg(img):
    imgmask=cv2.imread('res/img002.png')
    cv2.imshow('mask img',imgmask)
    mask2=cv2.cvtColor(imgmask,cv2.COLOR_RGB2GRAY,dstCn=1)
    mask[mask2==0]=0
    mask[mask2 == 255] = 1
    cv2.grabCut(img, mask, rect, bg, fg, 5, cv2.GC_INIT_WITH_RECT)
    mask2=np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img=img*mask2[:,:,np.newaxis]
    cv2.imshow('grabCut',img)
    cv2.waitKey(0)

EditedImg(img)
'''

cv2.GC_INIT_WITH_MASK
cv2.GC_EVAL
cv2.GC_EVAL_FREEZE_MODEL
'''