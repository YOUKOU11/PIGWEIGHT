# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix    # 导入计算混淆矩阵的包
import seaborn as sns   # 导入包
import matplotlib.pyplot as plt

def create_confusion_matrix(predict, target):
    confusion_matrix = []
    # 创建混淆矩阵
    for i in range(5):
        temp = []
        for j in range(5):
            temp.append(0)
        confusion_matrix.append(temp)
    for i in range(len(predict)):
        confusion_matrix[target[i]][predict[i]] += 1

    pre = []
    recall = []
    f1 = []
    for i in range(5):
        sum1 = 0
        sum2 = 0
        for j in range(5):
            sum1 += confusion_matrix[i][j]
            sum2 += confusion_matrix[j][i]
        if sum2 != 0:
            pre.append(confusion_matrix[i][i] / sum2 * 100)
        else:
            pre.append(100)
        if sum1 != 0:
            recall.append(confusion_matrix[i][i] / sum1 * 100)
        else:
            recall.append(100)
        f1.append(2 * pre[i] * recall[i] / (pre[i] + recall[i]))
    print("\n")
    print(pre)
    print(recall)
    print(f1)

    x_tick = ['fast', 'lame', 'normal', 'slow', 'uncooperative']
    y_tick = ['fast', 'lame', 'normal', 'slow', 'uncooperative']
    sns.set(font_scale=0.75)
    ax = plt.axes()
    img = sns.heatmap(confusion_matrix, fmt='g', cmap='Blues', annot=True, cbar=False, xticklabels=x_tick,
                yticklabels=y_tick)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    ax.set_xlabel('predict')
    ax.set_ylabel('true')
    ax.set_title('ResNet18-SE')
    img = img.get_figure()
    # img.savefig('ResNet18-SE22222.svg', dpi=600, bbox_inches='tight')
    plt.show()


def accuracy_numpy(pred, target, topk=(1, ), thrs=0.):
    print(666)
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.shape[0]

    static_inds = np.indices((num, maxk))[0]
    pred_label = pred.argpartition(-maxk, axis=1)[:, -maxk:]
    pred_score = pred[static_inds, pred_label]

    sort_inds = np.argsort(pred_score, axis=1)[:, ::-1]
    pred_label = pred_label[static_inds, sort_inds]
    pred_score = pred_score[static_inds, sort_inds]

    for k in topk:
        correct_k = pred_label[:, :k] == target.reshape(-1, 1)
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct_k = correct_k & (pred_score[:, :k] > thr)
            _correct_k = np.logical_or.reduce(_correct_k, axis=1)
            res_thr.append((_correct_k.sum() * 100. / num))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy_torch(pred, target, topk=(1, ), thrs=0.):
    if isinstance(thrs, Number):
        thrs = (thrs, )
        res_single = True
    elif isinstance(thrs, tuple):
        res_single = False
    else:
        raise TypeError(
            f'thrs should be a number or tuple, but got {type(thrs)}.')

    res = []
    maxk = max(topk)
    num = pred.size(0)
    pred = pred.float()
    pred_score, pred_label = pred.topk(maxk, dim=1)
    pred_label = pred_label.t()
    # create_confusion_matrix(pred_label[0].tolist(), target.tolist())     # 制作混淆矩阵
    correct = pred_label.eq(target.view(1, -1).expand_as(pred_label))
    for k in topk:
        res_thr = []
        for thr in thrs:
            # Only prediction values larger than thr are counted as correct
            _correct = correct & (pred_score.t() > thr)
            correct_k = _correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res_thr.append((correct_k.mul_(100. / num)))
        if res_single:
            res.append(res_thr[0])
        else:
            res.append(res_thr)
    return res


def accuracy(pred, target, topk=1, thrs=0.):
    """Calculate accuracy according to the prediction and target.

    Args:
        pred (torch.Tensor | np.array): The model prediction.
        target (torch.Tensor | np.array): The target of each prediction
        topk (int | tuple[int]): If the predictions in ``topk``
            matches the target, the predictions will be regarded as
            correct ones. Defaults to 1.
        thrs (Number | tuple[Number], optional): Predictions with scores under
            the thresholds are considered negative. Default to 0.

    Returns:
        torch.Tensor | list[torch.Tensor] | list[list[torch.Tensor]]: Accuracy
            - torch.Tensor: If both ``topk`` and ``thrs`` is a single value.
            - list[torch.Tensor]: If one of ``topk`` or ``thrs`` is a tuple.
            - list[list[torch.Tensor]]: If both ``topk`` and ``thrs`` is a \
              tuple. And the first dim is ``topk``, the second dim is ``thrs``.
    """
    assert isinstance(topk, (int, tuple))
    if isinstance(topk, int):
        topk = (topk, )
        return_single = True
    else:
        return_single = False

    assert isinstance(pred, (torch.Tensor, np.ndarray)), \
        f'The pred should be torch.Tensor or np.ndarray ' \
        f'instead of {type(pred)}.'
    assert isinstance(target, (torch.Tensor, np.ndarray)), \
        f'The target should be torch.Tensor or np.ndarray ' \
        f'instead of {type(target)}.'

    # torch version is faster in most situations.
    to_tensor = (lambda x: torch.from_numpy(x)
                 if isinstance(x, np.ndarray) else x)
    pred = to_tensor(pred)
    target = to_tensor(target)

    res = accuracy_torch(pred, target, topk, thrs)

    return res[0] if return_single else res


class Accuracy(nn.Module):

    def __init__(self, topk=(1, )):
        """Module to calculate the accuracy.

        Args:
            topk (tuple): The criterion used to calculate the
                accuracy. Defaults to (1,).
        """
        super().__init__()
        self.topk = topk

    def forward(self, pred, target):
        """Forward function to calculate accuracy.

        Args:
            pred (torch.Tensor): Prediction of models.
            target (torch.Tensor): Target for each prediction.

        Returns:
            list[torch.Tensor]: The accuracies under different topk criterions.
        """
        return accuracy(pred, target, self.topk)
