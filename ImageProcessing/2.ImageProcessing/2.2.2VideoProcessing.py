import cv2
import numpy as np

vc=cv2.VideoCapture('res/video001.mp4')
fps=vc.get(cv2.CAP_PROP_FPS)
size=(int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)),int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)))
print(size)

def ReadVideo():
    success,frame=vc.read()
    print(size)
    while success:
        cv2.imshow('myvideo',frame)
        success,frame=vc.read()
        key=cv2.waitKey(1)
        print(key)
        if key==27:
            break
    vc.release()


def WriteVideo():
    vw=cv2.VideoWriter('res/videoWrite.mp4',cv2.VideoWriter_fourcc(*'mp4v'),fps,size)
    success, frame = vc.read()
    print(size)
    while success:
        cv2.imshow('writevideo', frame)
        key = cv2.waitKey(10)
        vw.write(frame)
        success,frame=vc.read()
        #print(success)
    vc.release()

WriteVideo()

def CameraCapture():
    vc=cv2.VideoCapture(0)
    fps=30
    size = (int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(vc.get(cv2.CAP_PROP_FRAME_WIDTH)))
    vw = cv2.VideoWriter('res/videoWrite.mp4', cv2.VideoWriter_fourcc('X','2','6','4'), fps, size)
    success, frame = vc.read()
    while success:
        vw.write(frame)
        cv2.imshow('myCamera', frame)
        key = cv2.waitKey()
        print(key)
        if key == 27:
            break
        success,frame=vc.read()
    vc.release()


