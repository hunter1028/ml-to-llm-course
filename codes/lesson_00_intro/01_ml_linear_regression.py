"""使用线性回归，根据房屋面积预测房价。"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


X = np.array([[50], [60], [70], [80], [90], [100], [110]])
y = np.array([158, 183, 207, 235, 268, 292, 325])

model = LinearRegression()
model.fit(X, y)

new_house = np.array([[95]])
predicted_price = model.predict(new_house)

print("学到的斜率：", round(model.coef_[0], 2))
print("学到的截距：", round(model.intercept_, 2))
print("95 平方米预测价格：", round(predicted_price[0], 2), "万元")

plt.scatter(X[:, 0], y, color="blue", label="history data")
plt.plot(X[:, 0], model.predict(X), color="red", label="learned line")
plt.scatter(95, predicted_price[0], color="green", s=100, label="new prediction")
plt.xlabel("area (m^2)")
plt.ylabel("price (10k yuan)")
plt.legend()
plt.show()
