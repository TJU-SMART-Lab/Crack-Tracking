import cv2
from config import *
import numpy as np


def refresh_config():
    global img_path, crack_direction, img_time, cycle, alarm_time, alarm_min_distance, alarm_max_distance
    img_path = config['DEFAULT']['img_path']  # 文件路径
    crack_direction = config['DEFAULT']['crack_direction']  # 上下左右 默认右
    img_time = config['DEFAULT']['img_time']  # min/张
    cycle = config['DEFAULT']['cycle']  # 圈/min

    alarm_time = config['alarm']['alarm_time']  # min
    alarm_min_distance = config['alarm']['alarm_min_distance']  # μm
    alarm_max_distance = config['alarm']['alarm_max_distance']  # μm


def find_tip(img):
    refresh_config()
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
    print(find_tip(gray))
