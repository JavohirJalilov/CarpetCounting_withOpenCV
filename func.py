import numpy as np
import matplotlib.pyplot as plt
import cv2


def kernal(x,y):
    k = np.ones((x,y),dtype=np.uint8)
    return k