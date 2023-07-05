import cv2
import numpy as np
img=np.zeros((2,8),dtype=np.uint8)
print(img)
n=0
while True:
    cv2.imshow('GrayImg',img)
    n+=20
    img[:,:]=n

    key=cv2.waitKey(200)
    if key==27:
        break
