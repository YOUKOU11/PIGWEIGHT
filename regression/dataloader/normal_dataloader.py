import os
from regression.dataloader import DataLoader
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema


# 获取x,y数据
def gain_x_y_data():
    x = []
    y = []

    root = '../regression/data/temp_origin_data/'
    for filepath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            f = open(root + filename, encoding='utf-8')
            x_temp = []
            y_temp = []
            for line in f:
                y_temp.append(np.float64(line.strip().split(" ")[0]))
                one = np.int64(line.strip().split(" ")[2].split(":")[0])
                two = np.int64(line.strip().split(" ")[2].split(":")[1])
                three = np.int64(line.strip().split(" ")[2].split(":")[2])
                four = np.int64(line.strip().split(" ")[2].split(":")[3])
                time = (((one * 60 + two) * 60 + three) * 1000) + four
                x_temp.append(time)

            x_time_temp = []
            for i in range(len(x_temp)):
                if i == 0:
                    x_time_temp.append(0.0)
                else:
                    x_time_temp.append((x_temp[i] - x_temp[0]) / 1000.0)
            x.append(x_time_temp)
            y.append(y_temp)
            f.close()
            os.remove(root + '/' + filename)

    return x, y


# 获取x,y, true_weight数据
def gain_x_y_true_weight_data():
    x = []
    y = []
    true_weight = []

    weight_root = "D:/Python项目/regression/data/weight/true_weight.txt"
    weight = []
    f = open(weight_root, encoding='GBK')
    for line in f:
        temp = []
        line = line.rstrip("\n")
        temp.append(line)
        weight.append(temp)

    root = 'D:/Python项目/regression/data/test_data/normal/'
    for filepath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            f = open(root + filename, encoding='utf-8')
            x_temp = []
            y_temp = []
            for line in f:
                y_temp.append(np.float64(line.strip().split(" ")[0]))
                one = np.int64(line.strip().split(" ")[2].split(":")[0])
                two = np.int64(line.strip().split(" ")[2].split(":")[1])
                three = np.int64(line.strip().split(" ")[2].split(":")[2])
                four = np.int64(line.strip().split(" ")[2].split(":")[3])
                time = (((one * 60 + two) * 60 + three) * 1000) + four
                x_temp.append(time)

            x_time_temp = []
            for i in range(len(x_temp)):
                if i == 0:
                    x_time_temp.append(0.0)
                else:
                    x_time_temp.append((x_temp[i] - x_temp[0]) / 1000.0)
            x.append(x_time_temp)
            y.append(y_temp)

            for i in range(len(weight)):
                temp = []
                file_name = weight[i][0].split(",")[0]
                weight_data = weight[i][0].split(",")[1]
                if file_name == filename:
                    temp.append(np.float_(weight_data))
                    true_weight.append(temp)

    return x, y, true_weight


# 第一次获取持续时间最长的波峰、最大最小值、中位数
def gain_first_data(x, y):

    seg_x = []
    seg_y = []
    maximum_value = []  # 记录极大值的值
    minimum_value = []  # 记录极小值的值
    max_weight = np.max(y)
    α = 0.73
    seg_crest = max_weight * α  # 分割线
    max_value = argrelextrema(np.array(y), np.greater)[0]   # 获取极大值的下标
    min_value = argrelextrema(np.array(y), np.less)[0]  # 获取极小值的下标
    for i in range(len(y)):
        seg_x.append(x[i])
        if y[i] >= seg_crest:
            seg_y.append(y[i])
        # else:
        #     seg_y.append(np.nan)
    for i in range(len(max_value)):
        if y[max_value[i]] >= seg_crest:
            maximum_value.append(y[max_value[i]])

    # plt.plot(x, y, 'blue', label="origin_data", linestyle='--')
    # plt.plot(x, seg_y, 'green', label="first_data", linestyle='--')

    if maximum_value == []:
        maximum_value.append(seg_y[0])

    for i in range(len(min_value)):
        if y[min_value[i]] >= seg_crest:
            minimum_value.append(y[min_value[i]])

    max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_second_data(x, seg_y, maximum_value, minimum_value)

    return max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest


# 第二次获取持续时间最长的波峰、最大最小值、中位数
def gain_second_data(x, y, maximum_value, minimum_value):
    α = 0.77
    mean_maximum_value = np.mean(maximum_value)
    # print(mean_maximum_value, "====")

    if minimum_value == []:
        mean_minimum_value = mean_maximum_value
    else:
        mean_minimum_value = np.mean(minimum_value)
    mean_seg_crest = (mean_maximum_value + mean_minimum_value) / 2.0
    weight_seg_crest = mean_seg_crest * α

    seg_y = []
    maximum_value = []  # 记录极大值的值
    max_value = argrelextrema(np.array(y), np.greater)[0]  # 获取极大值的下标
    for i in range(len(y)):
        if y[i] >= weight_seg_crest:
            seg_y.append(y[i])
        # else:
        #     seg_y.append(np.nan)
    for i in range(len(max_value)):
        if y[max_value[i]] >= weight_seg_crest:
            maximum_value.append(y[max_value[i]])

    # plt.plot(x, seg_y, 'red', label="second_data", linestyle='-')
    # plt.xlabel("time")
    # plt.ylabel("weight")
    # plt.legend(loc=4)
    # plt.savefig("44.jpg", bbox_inches='tight')
    # plt.close()
    # plt.show()  # 显示预测值与测试值曲线

    mean_crest = np.mean(seg_y)  # 获取大于分界线值的平均值
    max_crest = np.max(seg_y)  # 获取最大值

    # 获取众数
    vals, counts = np.unique(seg_y, return_counts=True)
    mode_crest = vals[np.argmax(counts)]

    # 获取中位数
    seg_y.sort()
    median_crest = seg_y[(len(seg_y) - 1) // 2]

    mean_maximum_value = np.mean(maximum_value)
    if max_crest * 0.9 > mean_crest:
        max_crest = max_crest * 0.2 + mean_crest * 0.8
    else:
        max_crest = (max_crest + mean_crest) / 2.0

    return max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest


# # 第一次获取持续时间最长的波峰、最大最小值、中位数
# def gain_first_data(x, y):
#
#     seg_x = []
#     seg_y = []
#     maximum_value = []  # 记录极大值的值
#     minimum_value = []  # 记录极小值的值
#     max_weight = np.max(y)
#     α = 0.73
#     seg_crest = max_weight * α  # 分割线
#     max_value = argrelextrema(np.array(y), np.greater)[0]   # 获取极大值的下标
#     min_value = argrelextrema(np.array(y), np.less)[0]  # 获取极小值的下标
#     for i in range(len(y)):
#         if y[i] >= seg_crest:
#             seg_x.append(x[i])
#             seg_y.append(y[i])
#     for i in range(len(max_value)):
#         if y[max_value[i]] >= seg_crest:
#             maximum_value.append(y[max_value[i]])
#     if maximum_value == []:
#         maximum_value.append(seg_y[0])
#
#     for i in range(len(min_value)):
#         if y[min_value[i]] >= seg_crest:
#             minimum_value.append(y[min_value[i]])
#
#     max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_second_data(seg_y, maximum_value, minimum_value)
#
#     return max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest
#
#
# # 第二次获取持续时间最长的波峰、最大最小值、中位数
# def gain_second_data(y, maximum_value, minimum_value):
#     α = 0.77
#     mean_maximum_value = np.mean(maximum_value)
#     # print(mean_maximum_value, "====")
#
#     if minimum_value == []:
#         mean_minimum_value = mean_maximum_value
#     else:
#         mean_minimum_value = np.mean(minimum_value)
#     mean_seg_crest = (mean_maximum_value + mean_minimum_value) / 2.0
#     weight_seg_crest = mean_seg_crest * α
#
#     seg_y = []
#     maximum_value = []  # 记录极大值的值
#     max_value = argrelextrema(np.array(y), np.greater)[0]  # 获取极大值的下标
#     for i in range(len(y)):
#         if y[i] >= weight_seg_crest:
#             seg_y.append(y[i])
#     for i in range(len(max_value)):
#         if y[max_value[i]] >= weight_seg_crest:
#             maximum_value.append(y[max_value[i]])
#
#     mean_crest = np.mean(seg_y)  # 获取大于分界线值的平均值
#     max_crest = np.max(seg_y)  # 获取最大值
#
#     # 获取众数
#     vals, counts = np.unique(seg_y, return_counts=True)
#     mode_crest = vals[np.argmax(counts)]
#
#     # 获取中位数
#     seg_y.sort()
#     median_crest = seg_y[(len(seg_y) - 1) // 2]
#
#     mean_maximum_value = np.mean(maximum_value)
#     if max_crest * 0.9 > mean_crest:
#         max_crest = max_crest * 0.2 + mean_crest * 0.8
#     else:
#         max_crest = (max_crest + mean_crest) / 2.0
#
#     return max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest


def normal_dataloader(type='train', path=None):
    if type == 'train':
        x, y, true_weight = gain_x_y_true_weight_data()
        label = ['max_crest', 'median_crest', 'mode_crest', 'mean_maximum_value', 'mean_crest', 'true_weight']
    elif type == 'predict':
        x, y = DataLoader.gain_x_y_data(path)
        label = ['max_crest', 'median_crest', 'mode_crest', 'mean_maximum_value', 'mean_crest']

    crest_segment = []
    for i in range(len(y)):
        temp = []
        max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_first_data(x[i], y[i])
        temp.append(max_crest)
        temp.append(median_crest)
        temp.append(mode_crest)
        temp.append(mean_maximum_value)
        temp.append(mean_crest)
        if type == 'train':
            temp.append(true_weight[i][0])
        crest_segment.append(temp)

    crest_segment = pd.DataFrame(data=crest_segment, columns=label)

    return crest_segment


def normal_dataloader_test(type='train', path=None):
    if type == 'train':
        x, y, true_weight = gain_x_y_true_weight_data()
        label = ['max_crest', 'median_crest', 'mode_crest', 'mean_maximum_value', 'mean_crest', 'true_weight']
    elif type == 'predict':
        x, y, true_weight = DataLoader.gain_x_y_true_weight_data(path)
        label = ['max_crest', 'median_crest', 'mode_crest', 'mean_maximum_value', 'mean_crest']

    crest_segment = []
    for i in range(len(y)):
        temp = []
        max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_first_data(x[i], y[i])
        temp.append(max_crest)
        temp.append(median_crest)
        temp.append(mode_crest)
        temp.append(mean_maximum_value)
        temp.append(mean_crest)
        if type == 'train':
            temp.append(true_weight[i][0])
        crest_segment.append(temp)

    crest_segment = pd.DataFrame(data=crest_segment, columns=label)

    return crest_segment, true_weight