import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img002.png')
cv2.imshow('Original',img)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray=np.float32(gray)
dst=cv2.cornerHarris(gray,8,7,0.01)

r,dst=cv2.threshold(dst,0.01*dst.max(),255,0)

dst=np.uint8(dst)

r,l,s,cxys=cv2.connectedComponentsWithStats(dst)

cif=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,100,0.001)

corners=cv2.cornerSubPix(gray,np.float32(cxys),(5,5),(-1,-1),cif)

res=np.hstack((cxys,corners))
res=np.int0(res)

img[res[:,1],res[:,0]]=[0,0,255]

img[res[:,3],res[:,2]]=[0,255,0]

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
plt.imshow(img)
plt.axis('off')
plt.show()