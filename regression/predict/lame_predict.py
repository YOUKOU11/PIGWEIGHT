import torch
from regression.dataloader import lame_dataloader


def lame_predict(path):
    X, true_weight = lame_dataloader.lame_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/best_lame_regression.pth")
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    return Y_pred, true_weight
