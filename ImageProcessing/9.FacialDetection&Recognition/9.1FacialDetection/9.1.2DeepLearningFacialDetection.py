import cv2
import numpy as np
import matplotlib.pyplot as plt

dp='face_detector/deploy.prototxt'
rsf='face_detector/res10_300x300_ssd_iter_140000_fp16.caffemodel'
dnnnet =cv2.dnn.readNetFromCaffe( dp,rsf)

img = cv2.imread( "res/img001.png")
h, w = img. shape[:2]
blobs = cv2.dnn.blobFromImage(img,1.0,(300,300),[104.,117.,123.],False, False)

dnnnet.setInput(blobs)
detections = dnnnet.forward()
faces = 0
for i in range(0,detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.8:
        faces += 1
    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    x1, y1, x2, y2 = box.astype("int")
    y = y1 - 10 if y1 - 10 > 10 else y1 + 10
    text = "%.3f" % (confidence * 100) + '%'
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(img, text, (x1 + 20,y),    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0, 255),2)

cv2.imshow('face',img)
cv2.waitKey(0)