"""课时 07-08：使用糖尿病数据演示线性回归与 Ridge 回归。"""

import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# 1. 读取数据。
# X 的每一行是一位样本，每一列是一个特征；y 是需要预测的疾病进展指标。
raw = load_diabetes(as_frame=True)
X = raw.data
y = raw.target

# 2. 留出 20% 的数据作为测试集。
# random_state 固定随机划分，便于重复运行时得到相同结果。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

# 3. 训练普通线性回归。
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
linear_pred = linear_model.predict(X_test)

# 4. 使用测试集评价模型。
linear_mae = mean_absolute_error(y_test, linear_pred)
linear_rmse = mean_squared_error(y_test, linear_pred) ** 0.5
linear_r2 = r2_score(y_test, linear_pred)

print("=== 普通线性回归 ===")
print("MAE:", round(linear_mae, 2))
print("RMSE:", round(linear_rmse, 2))
print("R2:", round(linear_r2, 3))

# 5. 按绝对值从大到小查看各个特征的权重。
# 糖尿病数据集中的特征已经做过尺度处理，因此权重更便于横向比较。
coefficients = pd.Series(
    linear_model.coef_,
    index=X.columns,
    name="coefficient",
).sort_values(key=abs, ascending=False)

print("\n线性回归权重：")
print(coefficients.round(3))

# 6. 对比不同正则强度下的 Ridge 回归。
# Ridge 的目标是：预测误差 + alpha * 权重平方和。
# alpha 越大，限制大权重的力量越强；过大时可能造成欠拟合。
print("\n=== Ridge 回归：不同 alpha 对比 ===")
for alpha in [0.1, 1.0, 10.0, 100.0]:
    ridge_model = Ridge(alpha=alpha)
    ridge_model.fit(X_train, y_train)
    ridge_pred = ridge_model.predict(X_test)

    ridge_mae = mean_absolute_error(y_test, ridge_pred)
    ridge_rmse = mean_squared_error(y_test, ridge_pred) ** 0.5
    ridge_r2 = r2_score(y_test, ridge_pred)
    weight_norm = (ridge_model.coef_**2).sum() ** 0.5

    print(
        f"alpha={alpha:>5}: "
        f"MAE={ridge_mae:.2f}, "
        f"RMSE={ridge_rmse:.2f}, "
        f"R2={ridge_r2:.3f}, "
        f"权重L2范数={weight_norm:.2f}"
    )
