import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.signal import argrelextrema
import os

from regression.dataloader import DataLoader

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


# 获取x,y数据
def gain_x_y_data():
    x = []
    y = []

    root = 'D:/Python项目/regression/data/predict_data/lame/'
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
    print(len(weight))

    root = 'D:/Python项目/regression/data/test_data/lame/'
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

    print(len(true_weight))

    return x, y, true_weight


# 获取波峰段
def gain_coiled_crest_segment(x, y):    # 这里就是整段的波峰及其时间，该函数是获取分割线及其大于分割线的数据
    """
    先获取分割线，即极大值和极小值的平均值作为分割线，然后获取分割线以上的值，并获取持续世间最长的波峰段，最后就获取最大值，平均值等
    :param x: 时间
    :param y: 数据
    :return:
    """
    crest_segment_x = []
    crest_segment_y = []
    # print(y)

    # 记录最大最小值
    max_value = np.max(y)
    min_value = np.min(y)
    # print(min_value)

    maximum = argrelextrema(np.array(y), np.greater)[0]  # 获取极大值的下标
    minimum = argrelextrema(np.array(y), np.less)[0]  # 获取极大值的下标

    # 将所有极大值和极小值都记录起来
    maximum_y = []
    minimum_y = []
    for i in range(len(maximum)):
        maximum_y.append(y[maximum[i]])
    for i in range(len(minimum)):
        # if y[minimum[i]] >= max_value * 0.9:
            minimum_y.append(y[minimum[i]])

    # 更新极值
    if len(maximum_y) == 0:
        maximum_value = (max_value + min_value) / 2
    else:
        maximum_value = np.mean(maximum_y)
    if len(minimum_y) == 0:
        minimum_value = max_value * 0.4 + min_value * 0.6
    else:
        minimum_value = np.mean(minimum_y)

    # 获取平均极值
    mean_extremum_value = (maximum_value + minimum_value) / 2
    # mean_extremum_value = np.min(minimum_value)
    # print(mean_extremum_value)

    coiled_crest_segment = [0, 0]  # 记录最长波峰段的开始下标和结束下标

    # 获取最长波峰段
    first = 0
    last = len(y)
    continue_time = 0
    book = 0

    # temp_y = []
    # for i in range(len(y)):
    #     if y[i] < mean_extremum_value:
    #         temp_y.append(i)
    # temp_first = temp_y[0]
    # temp_last = temp_y[1]
    # temp_continue_time = temp_last - temp_first
    # for i in range(len(temp_y)):
    #     if i >= 2:
    #         if temp_y[i] - temp_last > temp_continue_time:
    #             temp_continue_time = temp_y[i] - temp_last
    #             temp_first = temp_last
    #             temp_last = temp_y[i]
    # first = temp_first
    # last = temp_last
    # continue_time = temp_continue_time
    # # print(temp_first, temp_last, temp_continue_time)

    for i in range(len(y)):
        # if y[i] == max_value:
        #     y[i] = y[i] * 0.95
        if y[i] >= mean_extremum_value and book == 0 and i != len(y) - 1:
            first = i
            book = 1
        elif y[i] < mean_extremum_value and book == 1 or book == 1 and i == len(y) - 1:
            last = i
            book = 0
            if last - first > continue_time:
                coiled_crest_segment[0] = first
                coiled_crest_segment[1] = last
                continue_time = last - first

    # 记录最长波峰段的最大最小值
    # crest_segment_max_value = np.max(y[temp_first: temp_last+1])
    # crest_segment_min_value = np.min(y[temp_first: temp_last+1])

    crest_segment_after_y = y[first: last+1]
    crest_segment_after_x = x[first: last+1]

    # plt.plot(crest_segment_after_x, crest_segment_after_y, 'red', label="second_data")
    # plt.legend(loc=4)
    # plt.xlabel("time")
    # plt.ylabel("weight")
    # plt.savefig("2.jpg", bbox_inches='tight')
    # plt.close()
    # plt.show()  # 显示预测值与测试值曲线

    # print(crest_segment_after_y)

    # for i in range(continue_time + 1):
    #     if y[coiled_crest_segment[0] + i] > max_value * 0.95:
    #         crest_segment_y.append(max_value * 0.93)
    #     elif y[coiled_crest_segment[0] + i] < max_value * 0.95 * 0.95:
    #         crest_segment_y.append(y[coiled_crest_segment[0] + i] * 1.06)
    #     else:
    #         crest_segment_y.append(y[coiled_crest_segment[0] + i])
    #
    #     crest_segment_x.append(x[coiled_crest_segment[0] + i])

    for i in range(len(crest_segment_after_y)):
        if crest_segment_after_y[i] > max_value * 0.97:
            crest_segment_y.append(crest_segment_after_y[i] * 0.97)
        elif crest_segment_after_y[i] < max_value * 0.95 * 0.95:
            crest_segment_y.append(crest_segment_after_y[i] * 1.06)
        else:
            crest_segment_y.append(crest_segment_after_y[i])
        crest_segment_x.append(crest_segment_after_x[i])


    mean_crest = np.mean(crest_segment_y) * 0.95  # 获取大于分界线值的平均值
    max_crest = np.max(crest_segment_y)   # 获取最大值


    # # 获取众数
    vals, counts = np.unique(crest_segment_y, return_counts=True)
    mode_crest = vals[np.argmax(counts)]

    # 获取中位数
    crest_segment_y.sort()
    median_crest = crest_segment_y[(len(crest_segment_y) - 1) // 2]

    mean_maximum_value = np.mean(maximum_value) * 0.95

    mode_crest = (mode_crest + mean_maximum_value) / 2
    # mean_maximum_value = mode_crest

    # print(max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest)

    return max_crest, median_crest, mean_maximum_value, mean_crest, mode_crest


def gain_crest_segment(x, y):   # 获取波峰段
    maximum_value = np.max(y)   # 获取原始数据的最大值
    crest_segment_x = []        # 记录波峰段开始时间和结束时间
    crest_segment_y = []        # 记录波峰段开始位置和结束位置
    α = 0.89

    index = []

    last_index = len(y) - 1
    for i in range(len(y)):     # 从开头往后遍历，获取极值，即确定开始位置
        if y[i] >= maximum_value * α and y[i] > y[i+1]:
            index.append(i)
            break
    for i in range(len(y)):     # 从后面往前遍历，获取极值，即确定结束位置
        if y[last_index] >= maximum_value * α and y[last_index] > y[last_index - 1]:
            index.append(last_index + 1)
            break
        last_index -= 1

    for i in range(index[1] - index[0]):    # 存储波峰段及其时间
        crest_segment_x.append(x[index[0] + i])
        crest_segment_y.append(y[index[0] + i])

    # plt.plot(x, y, 'blue', label="origin_data")
    # plt.plot(crest_segment_x, crest_segment_y, 'green', label="first_data")
    # plt.legend(loc=4)
    # plt.xlabel("number")
    # plt.ylabel("weight")
    # plt.show()  # 显示预测值与测试值曲线

    return gain_coiled_crest_segment(crest_segment_x, crest_segment_y)


# 第一次获取持续时间最长的波峰、最大最小值、中位数
def gain_first_data(x, y):

    seg_x = []
    seg_y = []
    maximum_value = []  # 记录极大值的值
    minimum_value = []  # 记录极小值的值
    max_weight = np.max(y)
    α = 0.75
    seg_crest = max_weight * α  # 分割线
    max_value = argrelextrema(np.array(y), np.greater)[0]   # 获取极大值的下标
    min_value = argrelextrema(np.array(y), np.less)[0]  # 获取极小值的下标
    for i in range(len(y)):
        if y[i] >= seg_crest:
            seg_x.append(x[i])
            seg_y.append(y[i])
    for i in range(len(max_value)):
        if y[max_value[i]] >= seg_crest:
            maximum_value.append(y[max_value[i]])

    # plt.plot(range(len(y)), y, 'blue', label="origin_data")
    # plt.plot(range(len(seg_y)), seg_y, 'green', label="first_data")

    if maximum_value == []:
        maximum_value.append(np.mean(y))

    for i in range(len(min_value)):
        if y[min_value[i]] >= seg_crest:
            minimum_value.append(y[min_value[i]])

    return gain_second_data(seg_y, maximum_value, minimum_value)


# 第二次获取持续时间最长的波峰、最大最小值、中位数
def gain_second_data(y, maximum_value, minimum_value):
    α = 0.8
    mean_maximum_value1 = np.mean(maximum_value)
    # print(mean_maximum_value, "====")

    if minimum_value == []:
        mean_minimum_value = mean_maximum_value1
    else:
        mean_minimum_value = np.mean(minimum_value)
    mean_seg_crest = (mean_maximum_value1 + mean_minimum_value) / 2.0
    weight_seg_crest = mean_seg_crest * α

    seg_y = []
    maximum_value = []  # 记录极大值的值
    max_value = argrelextrema(np.array(y), np.greater)[0]  # 获取极大值的下标
    for i in range(len(y)):
        if y[i] >= weight_seg_crest:
            seg_y.append(y[i])
    for i in range(len(max_value)):
        if y[max_value[i]] >= weight_seg_crest:
            maximum_value.append(y[max_value[i]])

    # plt.plot(range(len(seg_y)), seg_y, 'red', label="second_data")
    # plt.legend(loc=4)
    # plt.xlabel("number")
    # plt.ylabel("weight")
    # plt.show()  # 显示预测值与测试值曲线

    mean_crest = np.mean(seg_y)  # 获取大于分界线值的平均值
    max_crest = np.max(seg_y)  # 获取最大值

    if maximum_value == []:
        maximum_value.append(mean_crest)

    # 获取众数
    vals, counts = np.unique(seg_y, return_counts=True)
    mode_crest = vals[np.argmax(counts)]

    # 获取中位数
    seg_y.sort()
    median_crest = seg_y[(len(seg_y) - 1) // 2]

    mean_maximum_value = np.mean(maximum_value)
    if max_crest * 0.9 > mean_crest:
        max_crest = max_crest * 0.4 + mean_crest * 0.6
    else:
        max_crest = max_crest * 0.6 + mean_crest * 0.4

    return max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest


def lame_dataloader(type='train'):
    if type == 'train':
        x, y, true_weight = gain_x_y_true_weight_data()
        print(true_weight)
        label = ['max_crest', 'median_crest', 'mean_maximum_value', 'mean_crest', 'mode_crest', 'true_weight']
    elif type == 'predict':
        x, y = gain_x_y_data()
        label = ['max_crest', 'median_crest', 'mean_maximum_value', 'mean_crest', 'mode_crest']

    crest_segment = []
    for i in range(len(y)):
        # print(i)
        temp = []
        # max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_first_data(x[i], y[i])
        max_crest, median_crest, mean_maximum_value, mean_crest, mode_crest = gain_crest_segment(x[i], y[i])
        temp.append(max_crest)
        temp.append(median_crest)
        temp.append(mean_maximum_value)
        temp.append(mean_crest)
        temp.append(mode_crest)
        if type == 'train':
            temp.append(true_weight[i][0])
        crest_segment.append(temp)

    crest_segment = pd.DataFrame(data=crest_segment, columns=label)
    # print(crest_segment)
    # print(true_weight)

    return crest_segment


def lame_dataloader_test(type='train', path=None):
    if type == 'train':
        x, y, true_weight = gain_x_y_true_weight_data()
        print(true_weight)
        label = ['max_crest', 'median_crest', 'mean_maximum_value', 'mean_crest', 'mode_crest', 'true_weight']
    elif type == 'predict':
        x, y, true_weight = DataLoader.gain_x_y_true_weight_data(path)
        label = ['max_crest', 'median_crest', 'mean_maximum_value', 'mean_crest', 'mode_crest']

    crest_segment = []
    for i in range(len(y)):
        # print(i)
        temp = []
        # max_crest, median_crest, mode_crest, mean_maximum_value, mean_crest = gain_first_data(x[i], y[i])
        max_crest, median_crest, mean_maximum_value, mean_crest, mode_crest = gain_crest_segment(x[i], y[i])
        temp.append(max_crest)
        temp.append(median_crest)
        temp.append(mean_maximum_value)
        temp.append(mean_crest)
        temp.append(mode_crest)
        if type == 'train':
            temp.append(true_weight[i][0])
        crest_segment.append(temp)

    crest_segment = pd.DataFrame(data=crest_segment, columns=label)
    # print(crest_segment)
    # print(true_weight)

    return crest_segment, true_weight