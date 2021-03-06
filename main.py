import os
import threading
import time
import cv2
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
from model.my_unet_model import my_unet
from utils.visual import show_plot
from utils.tip import find_tip
from config import *

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

working_status = 0


def refresh_config():
    global img_path, crack_direction, img_time, cycle, alarm_time, alarm_min_distance, alarm_max_distance
    img_path = config['DEFAULT']['img_path']  # 文件路径
    crack_direction = config['DEFAULT']['crack_direction']  # 上下左右 默认右
    img_time = config['DEFAULT']['img_time']  # min/张
    cycle = config['DEFAULT']['cycle']  # 圈/min

    alarm_time = config['alarm']['alarm_time']  # min
    alarm_min_distance = config['alarm']['alarm_min_distance']  # μm
    alarm_max_distance = config['alarm']['alarm_max_distance']  # μm


def main():
    path_to_watch = img_path
    before = dict([(f, None) for f in os.listdir(path_to_watch)])
    model = my_unet((512, 512, 1))
    model.load_weights("model/1608616054_3250.hdf5")
    while working_status == 1:
        time.sleep(1)
        after = dict([(f, None) for f in os.listdir(path_to_watch)])
        added = [f for f in after if f not in before]
        removed = [f for f in before if f not in after]
        before = after
        if added:
            image_path = img_path + "/" + added[0]
            l = tk.Label(window, text='你好！this is Tkinter', bg='green', font=('Arial', 12), width=30, height=2)
            l.grid()
            task_temp = "./image/mask/" + added[0] + "_mask.png"
            img = Image.open(image_path).convert('L')
            img = img.resize((512, 512), Image.ANTIALIAS)
            x_predict = np.array([np.array(img) / 255])
            result = model.predict(x_predict)
            plt.figure(figsize=(1, 1), dpi=512)
            plt.imshow(np.squeeze(result[0])[:, :, 1], cmap="gray")
            plt.axis(False)
            plt.savefig(task_temp, bbox_inches='tight', pad_inches=0)
            gray = cv2.imread(task_temp)
            tip = find_tip(gray)
            print(tip)

            if os.path.exists(task_temp):
                os.remove(task_temp)
            # show_plot(X_predict, result, result, "./image/mask/" + added[0] + "_mask.png")
    print("stop")
    return


def start():
    global working_status
    if working_status == 0:
        refresh_config()
        working_status = 1
        T = threading.Thread(target=main)
        T.setDaemon(True)
        T.start()


def stop():
    global working_status
    working_status = 0


def settings():
    img_path_ = tk.StringVar()
    img_path_.set(img_path)
    direction_ = tk.StringVar()
    direction_.set(crack_direction)
    img_time_ = tk.StringVar()
    img_time_.set(img_time)
    cycle_ = tk.StringVar()
    cycle_.set(cycle)
    alarm_time_ = tk.StringVar()
    alarm_time_.set(alarm_time)
    alarm_min_distance_ = tk.StringVar()
    alarm_min_distance_.set(alarm_min_distance)
    alarm_max_distance_ = tk.StringVar()
    alarm_max_distance_.set(alarm_max_distance)

    def file():
        f = tk.filedialog.askdirectory()
        img_path_.set(f)

    def settings_save():
        config.set('DEFAULT', 'img_path', img_path_.get())
        config.set('DEFAULT', 'crack_direction', direction_.get())
        config.set('DEFAULT', 'img_time', img_time_.get())
        config.set('DEFAULT', 'cycle', cycle_.get())
        config.set('alarm', 'alarm_time', alarm_time_.get())
        config.set('alarm', 'alarm_min_distance', alarm_min_distance_.get())
        config.set('alarm', 'alarm_max_distance', alarm_max_distance_.get())
        with open('./config.ini', 'w') as ini:
            config.write(ini)
        refresh_config()
        setting.destroy()

    setting = tk.Toplevel(window)
    setting.geometry('600x400')
    setting.title("设置")
    e1 = tk.Entry(setting, textvariable=img_path_, show=None, font=('Arial', 14))  # 显示成明文形式
    e1.grid()
    b1 = tk.Button(setting, text="选择", command=file)
    b1.grid()
    r1 = tk.Radiobutton(setting, text='上', variable=direction_, value='0')
    r1.grid()
    r2 = tk.Radiobutton(setting, text='下', variable=direction_, value='1')
    r2.grid()
    r3 = tk.Radiobutton(setting, text='左', variable=direction_, value='2')
    r3.grid()
    r4 = tk.Radiobutton(setting, text='右', variable=direction_, value='3')
    r4.grid()
    # s1 = tk.Spinbox(setting, values=([h for h in range(10)]))
    # s1.grid()
    save_btn = tk.Button(setting, text="确定", command=settings_save)
    save_btn.grid()


if __name__ == '__main__':
    refresh_config()
    window = tk.Tk()
    window.title('裂纹追踪系统 v0.01')
    window.geometry('500x300')
    btn = tk.Button(window, text="开始", command=start)
    btn.grid()
    btn = tk.Button(window, text="结束", command=stop)
    btn.grid()
    btn = tk.Button(window, text="设置", command=settings)
    btn.grid()
    window.mainloop()
