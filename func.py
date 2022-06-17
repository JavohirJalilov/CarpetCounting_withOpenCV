import numpy as np
import matplotlib.pyplot as plt
import cv2


def kernal(x,y):
    k = np.ones((x,y),dtype=np.uint8)
    return k

def max_area_idx(contours):
  area_list = []
  for i in range(len(contours)):
    area_list.append(cv2.contourArea(contours[i]))
  area_list = np.array(area_list)
  mx_idx = area_list.argmax()
  return mx_idx