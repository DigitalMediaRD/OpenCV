import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('res/img001.png',cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('res/img002.png',cv2.IMREAD_GRAYSCALE)

orb=cv2.ORB_create()

kp1,des1=orb.detectAndCompute(img1,None)
kp2,des2=orb.detectAndCompute(img2,None)

bf=cv2.BFMatcher_create(cv2.NORM_HAMMING,crossCheck=True)

ms=bf.match(des1,des2)
ms=sorted(ms,key=lambda x:x.distance)

matchesMask=None

if len(ms)>10:
    querypts = np.float32([kp1[m.queryIdx].pt    for m in ms ] ).reshape(-1, 1, 2)
    trainpts = np.float32([kp1[m.trainIdx].pt    for m in ms ] ).reshape(-1, 1, 2)
    retv, mask = cv2.findHomography( querypts,trainpts, cv2.RANSAC)  # 计算最佳匹配结果的掩模，用于绘制匹配结果
    matchesMask = mask.ravel().tolist()
    h, w = img1.shape
    pts = np.float32([[0, 0],[0,h-1],[w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)  # 执行向量的透视矩阵转换,获得查询图像在训练图像中的位置
    dst = cv2.perspectiveTransform(pts, retv)
    img2 = cv2.polylines(img2, [np.int32(dst)], True, (255, 255, 255), 5)

img3 =cv2.drawMatches(img1,kp1,img2,kp2,ms, None,matchColor =(0,255,0),singlePointColor =None,matchesMask = matchesMask,
                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

plt.imshow(img3)
plt.axis('off')
plt.show()