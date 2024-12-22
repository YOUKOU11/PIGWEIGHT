import numpy as np

from regression.predict.fast_predict import fast_predict
from regression.predict.lame_predict import lame_predict
from regression.predict.slow_predict import slow_predict
from regression.predict.normal_predict import normal_predict
from regression.predict.uncooperative_predict import uncooperative_predict
from regression.predict.human_predict import human_predict


def proxy_predict(type, path):
    if type == 'fast':
        predict = fast_predict(path)
    elif type == 'slow':
        predict = slow_predict(path)
    elif type == 'normal':
        predict = normal_predict(path)
    elif type == 'uncooperative':
        predict = uncooperative_predict(path)
    elif type == 'human_or_still-life':
        predict = human_predict(path)
    else:
        predict = human_predict(path)
    return predict


def statistics_predict(true_type, type, path, predict_confusion_matrix, type_sum, compare_confusion_matrix, sum):
    """
    Args:
        type: 预测类别
        path: 源文件路径
        predict_confusion_matrix: 混淆矩阵
        type_sum: 每一类的个数

    Returns:混淆矩阵predict_confusion_matrix和每一类的个数type_sum

    """
    classification = ['fast', 'slow', 'normal', 'uncooperative', 'lame']

    if true_type == 'fast':
        true_predict, true_weight = fast_predict(path)
    elif true_type == 'slow':
        true_predict, true_weight = slow_predict(path)
    elif true_type == 'normal':
        true_predict, true_weight = normal_predict(path)
    elif true_type == 'uncooperative':
        true_predict, true_weight = uncooperative_predict(path)
    # elif true_type == 'human_or_still-life':
    #     true_predict, true_weight = human_predict(path)
    else:
        true_predict, true_weight = lame_predict(path)

    if type == 'fast':
        predict, true_weight1 = fast_predict(path)
    elif type == 'slow':
        predict, true_weight1 = slow_predict(path)
    elif type == 'normal':
        predict, true_weight1 = normal_predict(path)
    elif type == 'uncooperative':
        predict, true_weight1 = uncooperative_predict(path)
    # elif type == 'human_or_still-life':
    #     predict, true_weight1 = human_predict(path)
    else:
        predict, true_weight1 = lame_predict(path)

    sum += round(np.fabs(np.float_(true_weight) - np.float_(true_predict[0])), 2)
    x_book = 0
    y_book = 0
    for i in range(len(classification)):
        if classification[i] == true_type:
            x_book = i
        if classification[i] == type:
            y_book = i
    if true_type != type:
        compare_confusion_matrix[x_book][y_book] += round(np.fabs(np.float_(true_weight) - np.float_(true_predict[0])), 2)
        predict_confusion_matrix[x_book][y_book] += round(np.fabs(np.float_(true_weight1) - np.float_(predict[0])), 2)
        type_sum[x_book][x_book] += 1
        type_sum[x_book][y_book] += 1

    # print(predict_confusion_matrix)
    # print(compare_confusion_matrix)
    # print(type_sum)
    return predict_confusion_matrix, type_sum, compare_confusion_matrix, sum
