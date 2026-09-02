"""课时 03-04：一维函数上的梯度下降直觉。"""

w = 10.0
alpha = 0.1

for step in range(20):
    loss = (w - 3) ** 2
    grad = 2 * (w - 3)
    w = w - alpha * grad
    print(step, "w=", round(w, 4), "loss=", round(loss, 4))
