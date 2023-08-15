import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png')
cv2.imshow('original', img1)
temp=cv2.imread('res/template.png')
cv2.imshow('template',temp)

img1gray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY,dstCn=1)
tempgray=cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY,dstCn=1)
h,w=tempgray.shape
res=cv2.matchTemplate(img1gray,tempgray,cv2.TM_SQDIFF)
plt.imshow(res,cmap='gray')
plt.title('Matching Result')
plt.axis('off')
plt.show()
min_val,max_val,min_Loc,max_Loc=cv2.minMaxLoc(res)
top_left=min_Loc
bottom_right=(top_left[0]+w,top_left[1]+h)
cv2.rectangle(img1,top_left,bottom_right,(255,0,0),2)
cv2.imshow('Edtected Rang',img1)
cv2.waitKey(0)
'''
cv2.TM_CCORR_NORMED
cv2.TM_SQDIFF_NORMED
cv2.TM_CCOEFF_NORMED
'''