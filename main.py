import os
import sys
import threading
import time
import objgraph
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from config import *
from logger import logger
from model.my_unet_model import my_unet
from utils.tcpip import tcpIp
from utils.tip import find_tip

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

working_status = 0
tips = []
tip = (256, 256)
times_up = 0
times_right = 0
fError = open("except_error.log", 'a')


def watching(limit):
    global tips, working_status, tip
    working_status = 1
    tips.clear()
    count = 0
    x_Acc = 0.0  # 累计移动，绝对值
    y_Acc = 0.0
    path_to_watch = config.get("DEFAULT", "img_path")
    try:
        before = dict([(f, None) for f in os.listdir(path_to_watch)])
    except FileNotFoundError:
        working_status = 0
        print("文件路径不存在")
        return
    model = my_unet((512, 512, 1))
    model.load_weights("model/1608616054_3250.hdf5")
    while working_status == 1:
        time.sleep(1)
        after = dict([(f, None) for f in os.listdir(path_to_watch)])  # 判断文件变化
        added = [f for f in after if f not in before]
        removed = [f for f in before if f not in after]
        before = after
        if added:
            image_path = config.get("DEFAULT", "img_path") + "/" + added[0]
            # image = Image.open(image_path)
            # Photo = ImageTk.PhotoImage(image)
            # Lab['image'] = Photo
            task_temp = "./image/mask/" + added[0] + "_mask.png"
            try:
                img = Image.open(image_path).convert('L')
                img = img.resize((512, 512), Image.ANTIALIAS)
                x_predict = np.array([np.array(img) / 255])
                result = model.predict(x_predict)
                plt.figure(figsize=(1, 1), dpi=512)
                plt.imshow(np.squeeze(result[0])[:, :, 1], cmap="gray")
                plt.axis(False)
                plt.savefig(task_temp, bbox_inches='tight', pad_inches=0)
                plt.close()
                gray = cv2.imread(task_temp)
                tip = find_tip(gray, config.get("DEFAULT", "crack_direction"))
                count += 1
                print(tip)
            except OSError:
                print('跳过了一张图片')  # 图片有可能无法读取 原因未知

            if count:
                # gc.collect()  # 强制进行垃圾回收
                objgraph.show_most_common_types(limit=50)  # 打印出对象数目最多的 50 个类型信息
            if os.path.exists(task_temp):
                os.remove(task_temp)
            # show_plot(X_predict, result, result, "./image/mask/" + added[0] + "_mask.png")
        if limit == 'inf':
            if added:
                x = (256 - tip[0]) / float(config.get('Correction', 'times_right'))
                y = (tip[1] - 256) / float(config.get('Correction', 'times_up'))
                x_Acc += x
                y_Acc += y
                print(x, y)
                tcpIp(x_Acc, y_Acc)
        else:
            tips.append(tip)
            if count >= limit:
                working_status = 0
                break
    print('stop')


class Window(tk.Tk):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("裂纹追踪系统 v0.04")
        self.geometry("1000x600")
        self.btn_start = tk.Button(self, text="开始",
                                   state='normal' if config.get('Correction', 'last_corr') else 'disabled',
                                   command=self.start)
        self.btn_start.grid()
        self.btn_stop = tk.Button(self, text="结束", state='disabled', command=self.stop)
        self.btn_stop.grid()
        self.btn_set = tk.Button(self, text="设置", command=lambda: self.SettingWindow(self))
        self.btn_set.grid()
        self.btn_corr = tk.Button(self, text="校准", command=lambda: self.CorrectWindow(self))
        self.btn_corr.grid()
        self.btn_open = tk.Button(self, text="打开文件夹",
                                  command=lambda: os.system('start ' + config.get("DEFAULT", "img_path")))
        self.btn_open.grid()

    def start(self):
        self.btn_start['state'] = 'disabled'
        self.btn_stop['state'] = 'normal'
        self.btn_set['state'] = 'disabled'
        self.btn_corr['state'] = 'disabled'
        T = threading.Thread(target=lambda: watching('inf'))
        T.setDaemon(True)
        T.start()

    def stop(self):
        global working_status, tips
        working_status = 0
        self.btn_start['state'] = 'normal'
        self.btn_stop['state'] = 'disabled'
        self.btn_set['state'] = 'normal'
        self.btn_corr['state'] = 'normal'
        tips = []

    class CorrectWindow(tk.Toplevel):
        def __init__(self, parent):
            super().__init__(parent)
            self.up_ = tk.DoubleVar()
            self.right_ = tk.DoubleVar()
            self.e1 = tk.Entry(self, textvariable=self.right_, show=None, font=('Arial', 14))
            self.e2 = tk.Entry(self, textvariable=self.up_, show=None, font=('Arial', 14))  # up_=向上移动距离 right_=向右移动距离
            self.corr_finish = tk.Button(self, text="完成", command=self.confirm)

            self.parent = parent

            self.geometry('600x400')
            self.title("校准")
            self.grab_set()
            self.protocol('WM_DELETE_WINDOW', self.corr_close)
            self.corr_start_btn = tk.Button(self, text="开始校准", command=self.corr_start)
            self.corr_start_btn.grid()

        def corr_close(self):
            global working_status
            working_status = 0
            self.destroy()

        def corr_start(self):
            tips.clear()
            self.corr_start_btn.destroy()
            T = threading.Thread(target=lambda: self.parent.watching(2))
            T.setDaemon(True)
            T.start()

            self.e1.grid()
            self.e2.grid()
            self.corr_finish.grid()

        def confirm(self):
            global times_up, times_right
            times_right = (tips[1][0] - tips[0][0]) / self.right_.get()
            times_up = (tips[1][1] - tips[0][1]) / (0 - self.up_.get())
            print(times_right, times_up)
            config.set('Correction', 'times_up', str(times_up))
            config.set('Correction', 'times_right', str(times_right))
            config.set('Correction', 'last_corr', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            window.btn_start['state'] = "normal"
            with open('./config.ini', 'w') as ini:
                config.write(ini)
            self.destroy()

    class SettingWindow(tk.Toplevel):
        def __init__(self, parent):
            super().__init__(parent)
            self.geometry('600x400')
            self.title("设置")
            self.grab_set()

            self.img_path = tk.StringVar()
            self.img_path.set(config.get("DEFAULT", "img_path"))
            self.direction = tk.StringVar()
            self.direction.set(config.get("DEFAULT", "crack_direction"))
            self.img_time = tk.StringVar()
            self.img_time.set(config.get("DEFAULT", "img_time"))
            self.cycle_ = tk.StringVar()
            self.cycle_.set(config.get("DEFAULT", "cycle"))
            self.alarm_time = tk.StringVar()
            self.alarm_time.set(config.get("alarm", "alarm_time"))
            self.alarm_min_distance = tk.StringVar()
            self.alarm_min_distance.set(config.get("alarm", "alarm_min_distance"))
            self.alarm_max_distance = tk.StringVar()
            self.alarm_max_distance.set(config.get("alarm", "alarm_max_distance"))

            self.e_img = tk.Entry(self, textvariable=self.img_path, show=None, font=('Arial', 14))
            self.e_img.grid()
            self.b1 = tk.Button(self, text="选择", command=self.file)
            self.b1.grid()
            self.r1 = tk.Radiobutton(self, text='上', variable=self.direction, value='0')
            self.r1.grid()
            self.r2 = tk.Radiobutton(self, text='下', variable=self.direction, value='1')
            self.r2.grid()
            self.r3 = tk.Radiobutton(self, text='左', variable=self.direction, value='2')
            self.r3.grid()
            self.r4 = tk.Radiobutton(self, text='右', variable=self.direction, value='3')
            self.r4.grid()
            # self.s1 = tk.Spinbox(self, values=([h for h in range(10)]))
            # self.s1.grid()
            self.save_btn = tk.Button(self, text="确定", command=self.settings_save)
            self.save_btn.grid()

        def file(self):
            f = tk.filedialog.askdirectory()
            self.img_path.set(f)

        def settings_save(self):
            config.set('DEFAULT', 'img_path', self.img_path.get())
            config.set('DEFAULT', 'crack_direction', self.direction.get())
            config.set('DEFAULT', 'img_time', self.img_time.get())
            config.set('DEFAULT', 'cycle', self.cycle_.get())
            config.set('alarm', 'alarm_time', self.alarm_time.get())
            config.set('alarm', 'alarm_min_distance', self.alarm_min_distance.get())
            config.set('alarm', 'alarm_max_distance', self.alarm_max_distance.get())
            with open('./config.ini', 'w') as ini:
                config.write(ini)
            self.destroy()

    def report_callback_exception(self, exc_type, exc_value, exc_traceback):
        logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def handle_exception(exc_type, exc_value=None, exc_traceback=None):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))


def handle_thread_exception(exc):
    logger.error("Uncaught exception", exc_info=(exc.exc_type, exc.exc_value, exc.exc_traceback))


if __name__ == '__main__':
    sys.excepthook = handle_exception
    threading.excepthook = handle_thread_exception
    config = Config()
    window = Window()
    # image = Image.open(None)
    # Photo = ImageTk.PhotoImage(image)
    # Lab = tk.Label(window, image=Photo)
    # Lab.grid()
    window.mainloop()
    fError.close()
