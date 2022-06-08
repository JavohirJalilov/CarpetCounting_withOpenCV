import cv2
import numpy as np

def empty(a):
    pass

path = 'data/frame10.png'
cv2.namedWindow('TrackBars')
cv2.resizeWindow('TrackBars',640,240)
cv2.createTrackbar('Hue min','TrackBars',0,179,empty)
cv2.createTrackbar('Hue max','TrackBars',179,179,empty)
cv2.createTrackbar('Sat min','TrackBars',0,255,empty)
cv2.createTrackbar('Sat max','TrackBars',255,255,empty)
cv2.createTrackbar('val min','TrackBars',0,255,empty)
cv2.createTrackbar('val max','TrackBars',255,255,empty)
while True:
    img = cv2.imread(path)
    w,h = img.shape[:2]
    img = cv2.resize(img, (h//2,w//2))

    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos('Hue min','TrackBars')
    h_max = cv2.getTrackbarPos('Hue max','TrackBars')
    s_min = cv2.getTrackbarPos('Sat min','TrackBars')
    s_max = cv2.getTrackbarPos('Sat max','TrackBars')
    v_min = cv2.getTrackbarPos('val min','TrackBars')
    v_max = cv2.getTrackbarPos('val max','TrackBars')
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    cv2.imshow('orginal',img)
    cv2.imshow('HSV',imgHSV)
    cv2.imshow('Mask',mask)
    cv2.waitKey(1)