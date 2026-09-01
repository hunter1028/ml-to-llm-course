"""比较 L1、L2 和 Huber 损失对异常值的敏感程度。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import HuberRegressor, LinearRegression, QuantileRegressor


# ---------- 可以直接调整的参数 ----------
USE_OUTLIER = True       # True：把 100㎡ 房价改成 340，制造一个异常值
HUBER_EPSILON = 1.35     # 越小越抗异常值；越大越接近 L2（必须 >= 1.0）
PREDICT_AREA = 95        # 要预测的房屋面积
# --------------------------------------

X = np.array([[50], [60], [70], [80], [90], [100], [110]], dtype=float)
y = np.array([158, 183, 207, 235, 268, 292, 325], dtype=float)

if USE_OUTLIER:
    y[X[:, 0] == 100] = 340

# L2：最小化残差平方和 sum((y - y_hat)^2)
l2_model = LinearRegression()

# L1：最小化绝对误差和 sum(abs(y - y_hat))
# quantile=0.5 就是中位数回归，也叫最小绝对偏差回归。
# alpha=0 表示不额外加入 L1 参数正则化，避免把“损失”和“正则化”混在一起。
l1_model = QuantileRegressor(quantile=0.5, alpha=0, solver="highs")

# Huber：小残差使用平方损失，大残差改用近似线性增长。
# epsilon 越小，越早把样本按异常值处理；越大，整体越接近 L2。
# alpha=0 表示不额外加入 L2 参数正则化。
huber_model = HuberRegressor(epsilon=HUBER_EPSILON, alpha=0, max_iter=1000)

models = {
    "L2 / squared error": l2_model,
    "L1 / absolute error": l1_model,
    f"Huber / epsilon={HUBER_EPSILON}": huber_model,
}

new_house = np.array([[PREDICT_AREA]], dtype=float)

print(f"数据中{'包含' if USE_OUTLIER else '不包含'}异常值")
print(f"预测面积：{PREDICT_AREA} 平方米\n")

for name, model in models.items():
    model.fit(X, y)
    prediction = model.predict(new_house)[0]
    print(name)
    print(f"  斜率 w：{model.coef_[0]:.4f}")
    print(f"  截距 b：{model.intercept_:.4f}")
    print(f"  预测价格：{prediction:.2f} 万元\n")

# 画出三种损失得到的回归直线。
x_line = np.linspace(45, 115, 300).reshape(-1, 1)
colors = ["tab:red", "tab:green", "tab:orange"]

plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], y, color="tab:blue", s=60, label="training data", zorder=3)

if USE_OUTLIER:
    plt.scatter(100, 340, color="black", s=140, marker="x", linewidths=3,
                label="outlier", zorder=4)

for (name, model), color in zip(models.items(), colors):
    plt.plot(x_line[:, 0], model.predict(x_line), color=color, linewidth=2,
             label=name)

plt.axvline(PREDICT_AREA, color="gray", linestyle="--", linewidth=1)
plt.xlabel("area (m^2)")
plt.ylabel("price (10k yuan)")
plt.title("L1 vs L2 vs Huber regression")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()

# 本地运行时弹出图像窗口；在无图形界面的环境中自动保存图片，避免 plt.show() 报警告。
if "agg" in plt.get_backend().lower():
    output_dir = Path(__file__).with_name("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "l1_l2_huber_regression.png"
    plt.savefig(output_path, dpi=160)
    print(f"图像已保存：{output_path}")
else:
    plt.show()
