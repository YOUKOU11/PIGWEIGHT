import os

import numpy as np


def gain_x_y_data(path):
    x = []
    y = []

    # root = '../regression/data/temp_origin_data/'
    # for filepath, dirnames, filenames in os.walk(root):
    #     for filename in filenames:
    #         f = open(root + filename, encoding='utf-8')
    #         x_temp = []
    #         y_temp = []
    #         for line in f:
    #             y_temp.append(np.float64(line.strip().split(" ")[0]))
    #             # one = np.int64(line.strip().split(" ")[2].split(":")[0])
    #             two = np.int64(line.strip().split(" ")[2].split(":")[1])
    #             three = np.int64(line.strip().split(" ")[2].split(":")[2])
    #             four = np.int64(line.strip().split(" ")[2].split(":")[3])
    #             time = (((two * 60 + three) * 1000) + four) / 1000.0
    #             x_temp.append(time)
    #
    #         x_time_temp = []
    #         for i in range(len(x_temp)):
    #             if i == 0:
    #                 x_time_temp.append(0.0)
    #             else:
    #                 x_time_temp.append((x_temp[i] - x_temp[0]) / 1000.0)
    #         x.append(x_time_temp)
    #         y.append(y_temp)
    #         f.close()
    #         os.remove(root + '/' + filename)

    f = open(path, encoding='utf-8')
    x_temp = []
    y_temp = []
    for line in f:
        y_temp.append(np.float64(line.strip().split(" ")[0]))
        # one = np.int64(line.strip().split(" ")[2].split(":")[0])
        two = np.int64(line.strip().split(" ")[2].split(":")[1])
        three = np.int64(line.strip().split(" ")[2].split(":")[2])
        four = np.int64(line.strip().split(" ")[2].split(":")[3])
        time = (((two * 60 + three) * 1000) + four) / 1000.0
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

    return x, y


# 获取x,y, true_weight数据
def gain_x_y_true_weight_data(path):
    x = []
    y = []

    weight_root = "D:/Python项目/regression/data/weight/true_weight.txt"
    weight = []
    f = open(weight_root, encoding='GBK')
    for line in f:
        temp = []
        line = line.rstrip("\n")
        temp.append(line)
        weight.append(temp)

    f = open(path, encoding='utf-8')
    x_temp = []
    y_temp = []
    filename = path.split("/")[-1]
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
        file_name = weight[i][0].split(",")[0]
        weight_data = weight[i][0].split(",")[1]
        if file_name == filename:
            true_weight = np.float_(weight_data)
            break
        else:
            true_weight = 0
    f.close()
    # os.remove(path)

    return x, y, true_weight
