# 将原始数据转换成图片的形式
import os
import shutil

import numpy as np
from matplotlib import pyplot as plt


# 单个原数据进行预测
def transforms_image():
    temp_root = '../regression/data/temp_origin_data'
    origin_root = '../regression/data/origin_data'

    for filepath, dirnames, filenames in os.walk(temp_root):
        for filename in filenames:
            temp = []
            time = []
            tt = 0
            f = open(temp_root + '/' + filename, encoding='utf-8')
            for line in f:
                temp.append(np.float64(line.strip().split(" ")[0]))
                one = np.int64(line.strip().split(" ")[2].split(":")[0])
                two = np.int64(line.strip().split(" ")[2].split(":")[1])
                three = np.int64(line.strip().split(" ")[2].split(":")[2])
                four = np.int64(line.strip().split(" ")[2].split(":")[3])
                time_temp = (((one * 60 + two) * 60 + three) * 1000) + four
                if time == []:
                    time.append(0)
                    tt = time_temp
                else:
                    time.append((time_temp - tt) / 1000.0)
            plt.plot(time, temp,  color='#7EB6D1', markersize=12)
            # color=(136/255, 158/255, 170/255) #C8DBE4 #7EB6D1
            name = "../regression/data/figure_data/" + filename.split('.')[0] + '.jpg'
            # print(name)
            plt.savefig(name)
            plt.close()
            f.close()
            shutil.copy(temp_root + '/' + filename, origin_root + '/' + filename)

            # os.remove(temp_root + '/' + filename)
            img = "../regression/data/figure_data/" + filename.split('.')[0] + '.jpg'
            shutil.copy(img, "../demo/1.jpg")
            f.close()
            # os.remove(temp_root + '/' + filename)


# 多个原数据进行预测
def mul_transforms_image(temp_root):
    filename = temp_root.split("/")[-1]
    temp = []
    time = []
    tt = 0
    f = open(temp_root, encoding='utf-8')
    for line in f:
        temp.append(np.float64(line.strip().split(" ")[0]))
        one = np.int64(line.strip().split(" ")[2].split(":")[0])
        two = np.int64(line.strip().split(" ")[2].split(":")[1])
        three = np.int64(line.strip().split(" ")[2].split(":")[2])
        four = np.int64(line.strip().split(" ")[2].split(":")[3])
        time_temp = (((one * 60 + two) * 60 + three) * 1000) + four
        if time == []:
            time.append(0)
            tt = time_temp
        else:
            time.append((time_temp - tt) / 1000.0)
    plt.cla()
    plt.plot(time, temp,  color='#7EB6D1', markersize=12)
    # color=(136/255, 158/255, 170/255) #C8DBE4 #7EB6D1
    name = "../regression/data/figure_data/" + filename.split('.')[0] + '.jpg'
    plt.savefig(name)
    plt.close()
    shutil.copy("../regression/data/figure_data/" + filename.split('.')[0] + '.jpg', "../demo/1.jpg")
    f.close()