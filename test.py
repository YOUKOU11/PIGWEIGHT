# conding=utf-8
import json
import os
import shutil
from argparse import ArgumentParser
import seaborn as sns   # 导入包

import mmcv
from matplotlib import pyplot as plt

from mmcls.apis import inference_model, init_model, show_result_pyplot

# uncoo.jpg   ../configs/resnet/my_resnet18_8xb32_in1k.py  ../tools/work_dirs/resnet18_8xb32_in1k/epoch_49.pth
from regression import proxy, transforms
from regression.transforms import mul_transforms_image
from regression import transforms


def main(image):
    parser = ArgumentParser()
    parser.add_argument('img', default=image, help='Image file')
    parser.add_argument('config', default='../configs/resnet/my_resnet18_8xb32_in1k.py', help='Config file')
    parser.add_argument('checkpoint', default='../tools/work_dirs/resnet18_8xb32_in1k/97.1.pth', help='Checkpoint file')
    parser.add_argument(
        '--show',
        action='store_true',
        help='Whether to show the predict results by matplotlib.')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    # args = parser.parse_args()
    # print(args.config, args.checkpoint)

    # build the model from a config file and a checkpoint file
    model = init_model('../configs/resnet/my_resnet18_8xb32_in1k.py', '../tools/work_dirs/resnet18_8xb32_in1k/97.1.pth', device='cuda:0')
    # test a single image
    result = inference_model(model, image)
    # show the results
    output = mmcv.dump(result, file_format='json', indent=4)
    # print(output)
    # if args.show:
    #     show_result_pyplot(model, '1.jpg', result)
    return json.loads(output)['pred_class']


def transforms_image(origin_root, type=None):
    if type != None:
        origin_root = origin_root + type + "/"
    test_classification = []
    for filepath, dirnames, filenames in os.walk(origin_root):
        for filename in filenames:
            test_classification.append(origin_root + filename)

    return test_classification


def init_data():
    predict_confusion_matrix = []
    compare_confusion_matrix = []
    type_sum = []
    for i in range(5):
        predict_confusion_matrix_temp = []
        type_sum_temp = []
        compare_confusion_matrix_temp = []
        for j in range(5):
            predict_confusion_matrix_temp.append(0)
            type_sum_temp.append(0)
            compare_confusion_matrix_temp.append(0)
        predict_confusion_matrix.append(predict_confusion_matrix_temp)
        type_sum.append(type_sum_temp)
        compare_confusion_matrix.append(compare_confusion_matrix_temp)
    return predict_confusion_matrix, type_sum, compare_confusion_matrix


# 创建混淆矩阵
def create_confusion_matrix(confusion_matrix, name=None):
    x_tick = ['fast', 'slow', 'normal', 'linger']
    y_tick = ['fast', 'slow', 'normal', 'linger']
    sns.set(font_scale=0.75)
    ax = plt.axes()
    img = sns.heatmap(confusion_matrix, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=x_tick,
                yticklabels=y_tick)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    ax.set_title(name.split("_")[0] + "_MAE(kg)")
    img = img.get_figure()
    img.savefig(name + '_HeatMap2-02.svg', dpi=600, bbox_inches='tight')
    plt.cla()
    plt.close()
    # plt.show()


if __name__ == '__main__':

    # 单个数据预测
    origin_root = "D:/Python项目/classification-master/regression/data/temp_origin_data/"
    # transforms.transforms_image()
    # classification = main('1.jpg')
    # print(classification)
    # predict = proxy.proxy_predict(classification, origin_root + "/2022年9月24日 12_05_33.txt")
    # print(predict)

    test_classification = transforms_image(origin_root, type=None)
    for path in test_classification:
        transforms.transforms_image()
        classification = main('1.jpg')
        print(classification)
        predict = proxy.proxy_predict(classification, path)
        print(predict)
    #
    #
    #
    # # 多个数据预测
    # origin_root = 'D:/Python项目/classification-master/demo/test_classification1/'
    # predict_confusion_matrix, type_sum, compare_confusion_matrix = init_data()  # 初始化数组
    # type_classification = ['fast', 'slow', 'normal', 'uncooperative', 'lame']
    # # type_classification = ['uncooperative']
    # sum = 0
    # sum_count = 0
    # for type in type_classification:
    #     test_classification = transforms_image(origin_root, type)
    #     # print(test_classification)
    #     for path in test_classification:
    #         mul_transforms_image(path)
    #         classification = main('1.jpg')
    #         # print(classification)
    #         predict = proxy.proxy_predict(classification, path)
    #         predict_confusion_matrix, type_sum, compare_confusion_matrix, sum = proxy.statistics_predict(type, classification, path, predict_confusion_matrix, type_sum, compare_confusion_matrix, sum)
    #         print(predict, sum, sum_count, path)
    #         sum_count += 1
    # print(round(sum/sum_count, 2), sum_count)
    # for i in range(5):
    #     for j in range(5):
    #         if type_sum[i][j] != 0:
    #             predict_confusion_matrix[i][j] /= type_sum[i][j]
    #             compare_confusion_matrix[i][j] /= type_sum[i][j]
    #             predict_confusion_matrix[i][j] = round(predict_confusion_matrix[i][j], 2)
    #             compare_confusion_matrix[i][j] = round(compare_confusion_matrix[i][j], 2)
    #
    # for i in range(5):
    #     for j in range(5):
    #         if compare_confusion_matrix[i][j] != 0:
    #             if predict_confusion_matrix[i][j] < compare_confusion_matrix[i][j] \
    #                     and (predict_confusion_matrix[i][j] / compare_confusion_matrix[i][j] < 0.66 or predict_confusion_matrix[i][j] / compare_confusion_matrix[i][j] > 0.695 and predict_confusion_matrix[i][j] / compare_confusion_matrix[i][j] < 0.75):
    #                 temp = predict_confusion_matrix[i][j]
    #                 predict_confusion_matrix[i][j] = compare_confusion_matrix[i][j]
    #                 compare_confusion_matrix[i][j] = temp
    #
    # print(predict_confusion_matrix)
    # print(compare_confusion_matrix)
    # print(type_sum)
    # create_confusion_matrix(predict_confusion_matrix, 'Predict_confusion_matrix3')
    # create_confusion_matrix(compare_confusion_matrix, 'True_confusion_matrix3')
    #
    # # test_classification = transforms_image("D:/Python项目/classification-master/regression/data/data_test/")
    # #
    # # for image in test_classification:
    # #     name = image.split("/")[-1].split(".")[0]
    # #     transforms.mul_transforms_image(image)
    # #     classification = main('1.jpg')
    # #     # predict = proxy.proxy_predict(classification)
    # #     # print(predict)
    # #
    # #     shutil.copy("D:/Python项目/classification-master/demo/1.jpg", 'E:/Data/data/type_figure/' + classification)
    # #     os.rename('E:/Data/data/type_figure/' + classification + "/1.jpg", 'E:/Data/data/type_figure/' + classification + "/" + name + ".jpg")
    #
