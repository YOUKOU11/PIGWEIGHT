import torch
from regression.dataloader import slow_dataloader


def slow_predict(path):
    X, true_weight = slow_dataloader.slow_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/best_slow_regression.pth")
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    return Y_pred, true_weight
