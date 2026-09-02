"""课时 05-06：训练 baseline，并比较不同 L2 正则强度。"""

import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def split_data(random_state: int = 42):
    raw = load_diabetes(as_frame=True)
    X = raw.data
    y = raw.target

    rng = np.random.default_rng(random_state)
    idx = rng.permutation(len(X))

    n = len(X)
    n_train = int(n * 0.7)
    n_val = int(n * 0.15)

    train_idx = idx[:n_train]
    val_idx = idx[n_train : n_train + n_val]
    test_idx = idx[n_train + n_val :]

    return (
        X.iloc[train_idx],
        y.iloc[train_idx],
        X.iloc[val_idx],
        y.iloc[val_idx],
        X.iloc[test_idx],
        y.iloc[test_idx],
    )


def print_metrics(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"{name:5s} MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}")


X_train, y_train, X_val, y_val, X_test, y_test = split_data()

print("=== Baseline：普通线性回归 ===")
baseline = LinearRegression()
baseline.fit(X_train, y_train)

for name, X_part, y_part in [
    ("train", X_train, y_train),
    ("val", X_val, y_val),
    ("test", X_test, y_test),
]:
    pred = baseline.predict(X_part)
    print_metrics(name, y_part, pred)

print("\n=== Ridge：不同 alpha 对比 ===")
print("alpha    train_RMSE    val_RMSE")
for alpha in [0.0, 0.1, 1.0, 10.0, 100.0]:
    if alpha == 0.0:
        model = LinearRegression()
    else:
        model = Ridge(alpha=alpha)

    model.fit(X_train, y_train)
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    train_rmse = mean_squared_error(y_train, pred_train) ** 0.5
    val_rmse = mean_squared_error(y_val, pred_val) ** 0.5

    print(f"{alpha:5.1f}    {train_rmse:10.2f}    {val_rmse:8.2f}")
