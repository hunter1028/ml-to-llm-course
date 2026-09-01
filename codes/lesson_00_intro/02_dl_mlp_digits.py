"""使用多层感知机识别 scikit-learn 手写数字。"""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


digits = load_digits()
X = digits.data
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

first_image = X_test[0].copy().reshape(8, 8)
first_true_label = y_test[0]

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = MLPClassifier(
    hidden_layer_sizes=(64,32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("测试集准确率：", accuracy)
print("第一张图片的真实数字：", first_true_label)
print("第一张图片的预测数字：", y_pred[0])

plt.imshow(first_image, cmap="gray_r")
plt.title(f"true={first_true_label}, predict={y_pred[0]}")
plt.axis("off")
plt.show()
plt.close()

wrong_index = (y_pred != y_test).nonzero()[0][0]
print("真实数字：", y_test[wrong_index])
print("预测数字：", y_pred[wrong_index])
plt.imshow(X_test[wrong_index].reshape(8, 8), cmap="gray_r")
plt.title(f"true={y_test[wrong_index]}, predict={y_pred[wrong_index]}")
plt.axis("off")
plt.show()