"""课时 07-08：演示正规方程、矩阵求逆和 lstsq 的差异。

目的：
1. 看到线性回归的解析解 w = (X^T X)^(-1) X^T y；
2. 看到特征强相关时，X^T X 的条件数会变大；
3. 理解工程中为什么更推荐 np.linalg.lstsq 或 sklearn，而不是手写矩阵求逆。
"""

import numpy as np
from sklearn.linear_model import LinearRegression, Ridge


# 两个特征几乎重复：x2 约等于 2 * x1。
X = np.array(
    [
        [1.0, 2.01],
        [2.0, 3.98],
        [3.0, 6.02],
        [4.0, 8.01],
        [5.0, 9.99],
    ]
)
y = np.array([3.0, 6.1, 8.9, 12.1, 14.9])

# 为了让公式更接近 w = (X^T X)^(-1) X^T y，这里手动加一列 1 表示截距。
X_with_bias = np.c_[np.ones(X.shape[0]), X]

xtx = X_with_bias.T @ X_with_bias
xty = X_with_bias.T @ y

print("X^T X 的条件数：", round(np.linalg.cond(xtx), 2))

# 写法 1：直接求逆。教学上能对应公式，但工程中不推荐。
w_by_inverse = np.linalg.inv(xtx) @ xty
print("直接求逆得到的 [b, w1, w2]：")
print(np.round(w_by_inverse, 4))

# 写法 2：lstsq。数值稳定性通常更好。
w_by_lstsq, *_ = np.linalg.lstsq(X_with_bias, y, rcond=None)
print("lstsq 得到的 [b, w1, w2]：")
print(np.round(w_by_lstsq, 4))

# 写法 3：sklearn 的普通线性回归。
linear_model = LinearRegression()
linear_model.fit(X, y)
print("sklearn LinearRegression：")
print("b =", round(linear_model.intercept_, 4))
print("w =", np.round(linear_model.coef_, 4))

# 写法 4：Ridge。加入 L2 正则后，相关特征下的极端权重会被压低。
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X, y)
print("Ridge(alpha=1.0)：")
print("b =", round(ridge_model.intercept_, 4))
print("w =", np.round(ridge_model.coef_, 4))
