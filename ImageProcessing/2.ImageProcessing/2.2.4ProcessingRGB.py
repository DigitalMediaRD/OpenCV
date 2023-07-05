import cv2
import numpy as np
img=np.zeros((2,8,3),dtype=np.uint8)
print(img)
r0=0
r1=1
r2=2
while True:
    img[:80, :, r0] = 255
    img[80:160, :, r1] = 255
    img[160:, :, r2] = 255
    cv2.imshow('GolorImg',img)
    key = cv2.waitKey(200)
    img[:,:,:]=0
    t=r0
    r0=r1
    r1=r2
    r2=t


    if key==27:
        break