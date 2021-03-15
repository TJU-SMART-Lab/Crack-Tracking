import cv2
from config import *
import numpy as np


def find_tip(img, crack_direction):
    black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(black, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]

    leftmost = tuple(cnt[:, 0][cnt[:, :, 0].argmin()])
    rightmost = tuple(cnt[:, 0][cnt[:, :, 0].argmax()])

    if crack_direction == '0':
        tip = 0
    elif crack_direction == '1':
        tip = 1
    elif crack_direction == '2':
        tip = leftmost
    elif crack_direction == '3':
        tip = rightmost
    else:
        tip = "error"

    return tip


if __name__ == "__main__":
    gray = cv2.imread("./mask_test.png")
    # print(find_tip(gray))
