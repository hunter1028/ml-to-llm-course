"""课时 05-06：手写 MSE + L2 参数惩罚。"""

import numpy as np


y_true = np.array([3.0, 5.0, 7.0])
y_pred = np.array([2.5, 5.5, 8.0])
w = np.array([3.0, 4.0])

mse = np.mean((y_true - y_pred) ** 2)
lambda_ = 0.1
l2_penalty = np.sum(w**2)
loss = mse + lambda_ * l2_penalty

print("y_true =", y_true)
print("y_pred =", y_pred)
print("residual =", y_true - y_pred)
print("w =", w)
print("mse =", mse)
print("l2_penalty =", l2_penalty)
print("lambda =", lambda_)
print("loss =", loss)
