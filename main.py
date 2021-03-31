import os
import threading
import time
import cv2
import tkinter as tk
import numpy as np
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from model.my_unet_model import my_unet
from utils.tip import find_tip
from config import *
from utils.tcpip import tcpIp

# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

working_status = 0
tips = []
times_up = 0
times_right = 0


def correct():
    def corr_start():
        tips.clear()
        corr_start.destroy()
        T = threading.Thread(target=lambda: watching(2))
        T.setDaemon(True)
        T.start()

        def confirm():
            global times_up, times_right
            times_up = (tips[1][1] - tips[0][1]) / (0 - up_.get())
            times_right = (tips[1][0] - tips[0][0]) / right_.get()
            print(times_up, times_right)
            config.set('Correction', 'times_up', str(times_up))
            config.set('Correction', 'times_right', str(times_right))
            config.set('Correction', 'last_corr', time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
            btn_start['state'] = "normal"
            with open('./config.ini', 'w') as ini:
                config.write(ini)
            corr_win.destroy()

        right_ = tk.DoubleVar()
        up_ = tk.DoubleVar()
        e1 = tk.Entry(corr_win, textvariable=right_, show=None, font=('Arial', 14))
        e1.grid()
        e2 = tk.Entry(corr_win, textvariable=up_, show=None, font=('Arial', 14))    # up_=向上移动距离 right_=向右移动距离
        e2.grid()
        corr_finis = tk.Button(corr_win, text="完成", command=confirm)
        corr_finis.grid()

    corr_win = tk.Toplevel(window)
    corr_win.geometry('600x400')
    corr_win.title("校准")
    corr_win.grab_set()
    corr_start = tk.Button(corr_win, text="开始校准", command=corr_start)
    corr_start.grid()


def watching(limit):
    global tips
    count = 0
    x_Acc = 0.0  # 累计移动，绝对值
    y_Acc = 0.0
    path_to_watch = config.get("DEFAULT", "img_path")
    before = dict([(f, None) for f in os.listdir(path_to_watch)])
    model = my_unet((512, 512, 1))
    model.load_weights("model/1608616054_3250.hdf5")
    while 1:
        time.sleep(1)
        after = dict([(f, None) for f in os.listdir(path_to_watch)])
        added = [f for f in after if f not in before]
        removed = [f for f in before if f not in after]
        before = after
        if added:
            image_path = config.get("DEFAULT", "img_path") + "/" + added[0]
            # image = Image.open(image_path)
            # Photo = ImageTk.PhotoImage(image)
            # Lab['image'] = Photo
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
            tip = find_tip(gray, config.get("DEFAULT", "crack_direction"))
            tips.append(tip)
            count += 1
            print(tip)
            print(tips)

            if os.path.exists(task_temp):
                os.remove(task_temp)
            # show_plot(X_predict, result, result, "./image/mask/" + added[0] + "_mask.png")
        if limit == 'inf':
            if added:
                x = (tips[-1][0] - 256) / float(config.get('Correction', 'times_right'))
                y = (256 - tips[-1][1]) / float(config.get('Correction', 'times_up'))
                x_Acc += x
                y_Acc += y
                print(x, y)
                tcpIp(x_Acc, y_Acc)
            if working_status == 0:
                break
        else:
            if count >= limit:
                break
    return


def start():
    global working_status
    if working_status == 0:
        # refresh_config()
        working_status = 1
        btn_start['state'] = 'disabled'
        btn_stop['state'] = 'normal'
        btn_set['state'] = 'disabled'
        btn_corr['state'] = 'disabled'
        T = threading.Thread(target=lambda: watching('inf'))
        T.setDaemon(True)
        T.start()


def stop():
    global working_status
    working_status = 0
    btn_start['state'] = 'normal'
    btn_stop['state'] = 'disabled'
    btn_set['state'] = 'normal'


def settings():
    img_path = tk.StringVar()
    img_path.set(config.get("DEFAULT", "img_path"))
    direction = tk.StringVar()
    direction.set(config.get("DEFAULT", "crack_direction"))
    img_time = tk.StringVar()
    img_time.set(config.get("DEFAULT", "img_time"))
    cycle_ = tk.StringVar()
    cycle_.set(config.get("DEFAULT", "cycle"))
    alarm_time = tk.StringVar()
    alarm_time.set(config.get("alarm", "alarm_time"))
    alarm_min_distance = tk.StringVar()
    alarm_min_distance.set(config.get("alarm", "alarm_min_distance"))
    alarm_max_distance = tk.StringVar()
    alarm_max_distance.set(config.get("alarm", "alarm_max_distance"))

    def file():
        f = tk.filedialog.askdirectory()
        img_path.set(f)

    def settings_save():
        config.set('DEFAULT', 'img_path', img_path.get())
        config.set('DEFAULT', 'crack_direction', direction.get())
        config.set('DEFAULT', 'img_time', img_time.get())
        config.set('DEFAULT', 'cycle', cycle_.get())
        config.set('alarm', 'alarm_time', alarm_time.get())
        config.set('alarm', 'alarm_min_distance', alarm_min_distance.get())
        config.set('alarm', 'alarm_max_distance', alarm_max_distance.get())
        with open('./config.ini', 'w') as ini:
            config.write(ini)
        setting.destroy()

    setting = tk.Toplevel(window)
    setting.geometry('600x400')
    setting.title("设置")
    setting.grab_set()
    e_img = tk.Entry(setting, textvariable=img_path, show=None, font=('Arial', 14))  # 显示成明文形式
    e_img.grid()
    b1 = tk.Button(setting, text="选择", command=file)
    b1.grid()
    r1 = tk.Radiobutton(setting, text='上', variable=direction, value='0', state='disable')
    r1.grid()
    r2 = tk.Radiobutton(setting, text='下', variable=direction, value='1', state='disable')
    r2.grid()
    r3 = tk.Radiobutton(setting, text='左', variable=direction, value='2')
    r3.grid()
    r4 = tk.Radiobutton(setting, text='右', variable=direction, value='3')
    r4.grid()
    # s1 = tk.Spinbox(setting, values=([h for h in range(10)]))
    # s1.grid()
    save_btn = tk.Button(setting, text="确定", command=settings_save)
    save_btn.grid()


if __name__ == '__main__':
    config = Config()
    window = tk.Tk()
    window.title('裂纹追踪系统 v0.01')
    window.geometry('1000x600')
    btn_start = tk.Button(window, text="开始", state='disabled', command=start)
    btn_start.grid()
    if config.get('Correction', 'last_corr'):
        btn_start['state'] = 'normal'
    btn_stop = tk.Button(window, text="结束", state='disabled', command=stop)
    btn_stop.grid()
    btn_set = tk.Button(window, text="设置", command=settings)
    btn_set.grid()
    btn_corr = tk.Button(window, text="校准", command=correct)
    btn_corr.grid()
    # image = Image.open(None)
    # Photo = ImageTk.PhotoImage(image)
    # Lab = tk.Label(window, image=Photo)
    # Lab.grid()

    window.mainloop()
