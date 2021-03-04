import os
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from model.my_unet_model import my_unet
from utils.visual import show_plot
from utils.tip import find_tip
from config import *

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

path_to_watch = img_path
before = dict([(f, None) for f in os.listdir(path_to_watch)])

if __name__ == '__main__':
    while 1:
        time.sleep(1)
        after = dict([(f, None) for f in os.listdir(path_to_watch)])
        added = [f for f in after if f not in before]
        removed = [f for f in before if f not in after]
        before = after
        if added:
            image_path = img_path + "/" + added[0]
            img = Image.open(image_path).convert('L')
            img = img.resize((512, 512), Image.ANTIALIAS)
            X_predict = np.array([np.array(img) / 255])
            model = my_unet((512, 512, 1))
            model.load_weights('model/1608616054_3250.hdf5')
            result = model.predict(X_predict)
            print(np.shape(result))
            plt.figure(figsize=(1, 1), dpi=512)
            plt.imshow(np.squeeze(result[0])[:, :, 1], cmap="gray")
            plt.axis(False)
            plt.savefig("./image/mask/" + added[0] + "_mask.png", bbox_inches='tight', pad_inches=0)
            gray = cv2.imread("./image/mask/" + added[0] + "_mask.png")
            tip = find_tip(gray)
            print(tip)

            # show_plot(X_predict, result, result, "./image/mask/" + added[0] + "_mask.png")
