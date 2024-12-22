import torch
from regression.dataloader import fast_dataloader


def fast_predict(path):
    X, true_weight = fast_dataloader.fast_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/best_fast_regression.pth")
    # a = model.intercept_  # 截距
    # b = model.coef_  # 回归系数
    # print("拟合参数:截距", a, ",回归系数：", b)
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    # print(Y_pred)
    return Y_pred, true_weight
