"""课时 03-04：用矩阵一次计算多个样本的预测值。"""

import numpy as np


X = np.array(
    [
        [60, 2, 18],
        [75, 3, 12],
        [90, 3, 8],
        [105, 4, 15],
    ]
)

w = np.array([[3.0], [12.0], [-2.0]])
b = 20

y_hat = X @ w + b

print("X.shape =", X.shape)
print("y_hat.shape =", y_hat.shape)
print(y_hat)
