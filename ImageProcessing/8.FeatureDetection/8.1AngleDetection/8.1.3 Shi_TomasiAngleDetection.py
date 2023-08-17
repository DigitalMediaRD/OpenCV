import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread('res/img002.png')
cv2.imshow('Original',img)
gray=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
gray=np.float32(gray)
corners=cv2.goodFeaturesToTrack(gray,6,0.1,100)
corners=np.int0(corners)
for i in corners:
    x,y=i.ravel()
    cv2.circle(img,(x,y),4,(0,0,255),-1)

img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis('off')
plt.show()