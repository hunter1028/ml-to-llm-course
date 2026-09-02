"""课时 03-04：画出不同标准差下的正态分布曲线。"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


x = np.linspace(-5, 5, 300)

for sigma in [0.5, 1.0, 2.0]:
    y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x**2) / (2 * sigma**2))
    plt.plot(x, y, label=f"sigma={sigma}")

plt.legend()
plt.title("Normal Distribution")
plt.xlabel("x")
plt.ylabel("density")
plt.grid(alpha=0.2)
plt.tight_layout()

# 本地运行时弹出图像窗口；在无图形界面的环境中自动保存图片。
if "agg" in plt.get_backend().lower():
    output_dir = Path(__file__).with_name("outputs")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "normal_distribution.png"
    plt.savefig(output_path, dpi=160)
    print(f"图像已保存：{output_path}")
else:
    plt.show()
