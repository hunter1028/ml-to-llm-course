"""课时 03-04：单个样本的向量与矩阵预测。"""

import numpy as np


x = np.array([[100, 3, 10]])
w = np.array([[3.0], [12.0], [-2.0]])
b = 20

y_hat = x @ w + b

print("x.shape =", x.shape)
print("w.shape =", w.shape)
print("y_hat.shape =", y_hat.shape)
print("y_hat =", y_hat)
