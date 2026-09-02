"""课时 05-06：手动划分训练集、验证集和测试集。"""

import numpy as np
from sklearn.datasets import load_diabetes


raw = load_diabetes(as_frame=True)
X = raw.data
y = raw.target

rng = np.random.default_rng(42)
idx = rng.permutation(len(X))

n = len(X)
n_train = int(n * 0.7)
n_val = int(n * 0.15)

train_idx = idx[:n_train]
val_idx = idx[n_train : n_train + n_val]
test_idx = idx[n_train + n_val :]

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

print("数据总量：", n)
print("训练集：", X_train.shape, y_train.shape)
print("验证集：", X_val.shape, y_val.shape)
print("测试集：", X_test.shape, y_test.shape)

print("\n索引重叠检查：")
print("train ∩ val =", set(train_idx) & set(val_idx))
print("train ∩ test =", set(train_idx) & set(test_idx))
print("val ∩ test =", set(val_idx) & set(test_idx))
