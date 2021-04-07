# -*- coding:utf-8 -*-
import cv2
import numpy as np


def find_tip(img, crack_direction):
    global binary

    img = cv2.resize(img, (512, 512))
    black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(black, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return 256, 256
    cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]

    leftmost = tuple(cnt[:, 0][cnt[:, :, 0].argmin()])
    rightmost = tuple(cnt[:, 0][cnt[:, :, 0].argmax()])
    topmost = tuple(cnt[:, 0][cnt[:, :, 1].argmin()])
    bottommost = tuple(cnt[:, 0][cnt[:, :, 1].argmax()])

    if crack_direction == '0':
        tip = topmost
    elif crack_direction == '1':
        tip = bottommost
    elif crack_direction == '2':
        tip = leftmost
    elif crack_direction == '3':
        tip = rightmost
    else:
        tip = rightmost

    return tip


"""    
最大连通域判断
img = cv2.resize(img, (512, 512))
black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(black, 127, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt = contours[np.argmax([cv2.contourArea(cnt) for cnt in contours])]

leftmost = tuple(cnt[:, 0][cnt[:, :, 0].argmin()])
rightmost = tuple(cnt[:, 0][cnt[:, :, 0].argmax()])

"""

"""
最大连通域面积的八分之一去噪
img = cv2.resize(img, (512, 512))
black = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(black, 127, 255, 0)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
cnt_max = np.max([cv2.contourArea(cnt) for cnt in contours])
# for cnt in contours:
#     if cv2.contourArea(cnt) < cnt_max / 8:  # Area<Area_max/8的连通域为噪点
#         cv2.drawContours(binary, [cnt], 0, 0, cv2.FILLED)   # 轮廓填充
[cv2.drawContours(binary, [cnt], 0, 0, cv2.FILLED) for cnt in contours if cv2.contourArea(cnt) < cnt_max / 8]
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

left = []
right = []
top = []
bottom = []
for cnt in contours:  # contours = {list:l};cnt = {ndarray:(p,1,2)}  # todo 介于优雅和丑陋之间
    left.append(cnt[:, 0][cnt[:, :, 0].argmin()])
    right.append(tuple(cnt[:, 0][cnt[:, :, 0].argmax()]))
    top.append(tuple(cnt[:, 0][cnt[:, :, 1].argmin()]))
    bottom.append(tuple(cnt[:, 0][cnt[:, :, 1].argmax()]))
leftmost = tuple(left[np.array(left)[:, 0].argmin()])
rightmost = tuple(right[np.array(left)[:, 0].argmax()])
topmost = tuple(top[np.array(left)[:, 0].argmin()])
bottommost = tuple(bottom[np.array(left)[:, 0].argmax()])

"""


if __name__ == "__main__":
    binary = []
    gray = cv2.imread("../image/20210402-163424-189.jpg_mask.png")
    print(find_tip(gray, '3'))
    cv2.imshow("binary", binary)
    cv2.waitKey(0)
