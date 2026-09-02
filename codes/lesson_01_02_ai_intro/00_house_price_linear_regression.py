"""课时 01-02：用面积、房间数、房龄预测房价的线性回归最小示例。"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error


# X: 面积、房间数、房龄。
X = np.array(
    [
        [60, 2, 18],
        [75, 3, 12],
        [90, 3, 8],
        [105, 4, 15],
        [120, 4, 6],
    ]
)

# y: 房价 / 万元。
y = np.array([185, 248, 312, 340, 420])

model = LinearRegression()
model.fit(X, y)

print("w:", model.coef_)
print("b:", model.intercept_)

new_house = np.array([[100, 3, 10]])
prediction = model.predict(new_house)
print("预测房价:", prediction[0])

y_hat = model.predict(X)
print("MSE:", mean_squared_error(y, y_hat))
print("MAE:", mean_absolute_error(y, y_hat))
