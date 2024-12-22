import torch
from regression.dataloader import normal_dataloader


def normal_predict(path):
    X, true_weight = normal_dataloader.normal_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/best_normal_regression.pth")
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    return Y_pred, true_weight
