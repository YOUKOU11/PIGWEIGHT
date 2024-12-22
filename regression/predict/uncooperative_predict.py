import torch
from regression.dataloader import uncooperative_dataloader


def uncooperative_predict(path):
    X, true_weight = uncooperative_dataloader.uncooperative_dataloader_test(type='predict', path=path)
    model = torch.load("../regression/model/best_uncooperative_regression.pth")
    Y_pred = model.predict(X)  # 对测试集数据，用predict函数预测
    return Y_pred, true_weight
