import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from dataloader import fast_dataloader


if __name__ == '__main__':
    crest_segment = fast_dataloader.fast_dataloader(type='train')
    X = crest_segment.drop(["true_weight"], axis=1)  # 删除真实体重标签
    y = crest_segment["true_weight"]  # 真实体重标签
    print(X, y)

    X_train, X_valid, y_train, y_valid = train_test_split(X, y,
                                                          test_size=0.3,
                                                          train_size=0.7,
                                                          shuffle=True,
                                                          random_state=10)

    # 调用线性规划包
    model = LinearRegression()
    # model = SVR(kernel=kernel)
    # model = Ridge(alpha=1.0)
    # model = Lars()
    # model = ElasticNetCV(cv=2)
    model.fit(X_train, y_train)  # 线性回归训练
    print(model)
    torch.save(model, "../model/fast_regression.pth")

    a = model.intercept_  # 截距
    b = model.coef_  # 回归系数
    print("拟合参数:截距", a, ",回归系数：", b)

    # 显示线性方程，并限制参数的小数位为两位
    print("最佳拟合线: Y = ", round(a, 2), "+", round(b[0], 2), "* X1 + ", round(b[1], 2), "* X2", round(b[2], 2), "* X3 + ", round(b[3], 2), "* X4 + ", round(b[4], 2), "* X5")

    Y_pred = model.predict(X_valid)  # 对测试集数据，用predict函数预测
    # print(Y_pred)
    y_true = []
    sum = 0
    for i, data in enumerate(y_valid):
        y_true.append(data)
    for i in range(len(Y_pred)):
        sum = sum + np.fabs(Y_pred[i] - y_true[i])
        print(Y_pred[i] - y_true[i])
    print(sum / len(y_true))

    plt.plot(range(len(Y_pred)), Y_pred, 'red', linewidth=2.5, label="predict data")
    plt.plot(range(len(y_valid)), y_valid, 'green', label="test data")
    plt.legend(loc=2)
    plt.show()  # 显示预测值与测试值曲线
