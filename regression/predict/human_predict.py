import torch
from regression.dataloader import human_dataloader


def human_predict(path):
    X, true_weight = human_dataloader.human_or_still_life_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/human_regression.pth")
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    return Y_pred, true_weight
