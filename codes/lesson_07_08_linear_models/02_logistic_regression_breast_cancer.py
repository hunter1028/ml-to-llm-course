"""课时 07-08：使用乳腺癌数据演示逻辑回归、概率、阈值和标准化。"""

import warnings

from sklearn.datasets import load_breast_cancer
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# 1. 读取二分类数据。
# 在这个数据集中，y=0 表示 malignant（恶性），y=1 表示 benign（良性）。
raw = load_breast_cancer(as_frame=True)
X = raw.data
y = raw.target

# 2. 按标签比例划分训练集和测试集。
# stratify=y 保证两个数据集中 0、1 类别的比例大致一致。
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. Pipeline 保证 StandardScaler 只用训练集计算均值和标准差，
# 然后把同一套变换应用到测试集，避免预处理造成数据泄漏。
classifier = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, C=1.0)),
    ]
)
classifier.fit(X_train, y_train)

# 4. predict_proba 返回每个类别的概率。
# [:, 1] 取第二列，即模型估计的 P(y=1 | x)。
positive_probability = classifier.predict_proba(X_test)[:, 1]

# 5. 以 0.5 为阈值，把概率转换为最终类别。
prediction = (positive_probability >= 0.5).astype(int)

print("=== 标准化 + 逻辑回归 ===")
print("accuracy:", round(accuracy_score(y_test, prediction), 3))
print("precision:", round(precision_score(y_test, prediction), 3))
print("recall:", round(recall_score(y_test, prediction), 3))
print("f1:", round(f1_score(y_test, prediction), 3))
print("auc:", round(roc_auc_score(y_test, positive_probability), 3))
print("confusion matrix:")
print(confusion_matrix(y_test, prediction))

# 6. 比较不同阈值。
# 这里把 y=1（良性）作为正类，因此指标变化要结合这个标签含义理解。
print("\n=== 不同阈值的 Precision / Recall / F1 ===")
for threshold in [0.3, 0.5, 0.7]:
    threshold_prediction = (positive_probability >= threshold).astype(int)
    print(
        f"threshold={threshold:.1f}: "
        f"precision={precision_score(y_test, threshold_prediction):.3f}, "
        f"recall={recall_score(y_test, threshold_prediction):.3f}, "
        f"f1={f1_score(y_test, threshold_prediction):.3f}"
    )

# 7. 对比有无标准化。
# 逻辑回归包含正则化，特征量纲差异较大时，标准化通常更利于优化和解释。
model_without_scaling = LogisticRegression(max_iter=1000)
with warnings.catch_warnings(record=True) as captured_warnings:
    warnings.simplefilter("always", ConvergenceWarning)
    model_without_scaling.fit(X_train, y_train)
raw_score = model_without_scaling.score(X_test, y_test)
raw_iterations = model_without_scaling.n_iter_[0]
raw_reached_limit = any(
    issubclass(item.category, ConvergenceWarning) for item in captured_warnings
)

model_with_scaling = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ]
)
model_with_scaling.fit(X_train, y_train)
scaled_score = model_with_scaling.score(X_test, y_test)
scaled_iterations = model_with_scaling.named_steps["model"].n_iter_[0]

print("\n=== 有无标准化对比 ===")
print(f"未标准化：accuracy={raw_score:.3f}, 迭代次数={raw_iterations}")
if raw_reached_limit:
    print("未标准化模型已达到 max_iter：这说明量纲差异会让优化更难。")
print(f"已标准化：accuracy={scaled_score:.3f}, 迭代次数={scaled_iterations}")

# 8. Pandas 简单数据检查与异常截尾示例。
# 真正建模时，分位点也应只使用训练集计算，再应用到测试集。
dataframe = X.copy()
dataframe["target"] = y

print("\n=== 数据检查 ===")
print("缺失值最多的 5 列：")
print(dataframe.isna().sum().sort_values(ascending=False).head())

print("\n前 5 个特征的描述统计：")
print(dataframe.describe().T[["mean", "std", "min", "max"]].head().round(3))

column = X.columns[0]
lower = X_train[column].quantile(0.01)
upper = X_train[column].quantile(0.99)
clipped_test_column = X_test[column].clip(lower, upper)

print(f"\n特征 {column!r} 的训练集 1%~99% 截尾范围：[{lower:.3f}, {upper:.3f}]")
print("测试集中被截尾的样本数：", int((clipped_test_column != X_test[column]).sum()))
