# 课时 07-08：线性模型理论与实战代码

本目录放课时 07-08 配套的可运行代码。讲义中的零散代码块保留在正文中用于解释，完整脚本统一放在这里。

## 文件说明

| 文件 | 对应内容 | 运行方式 |
|---|---|---|
| `00_normal_equation_demo.py` | 正规方程、矩阵求逆、`lstsq` 与 Ridge 稳定性 | `python lesson_07_08_linear_models/00_normal_equation_demo.py` |
| `01_linear_regression_diabetes.py` | 糖尿病数据：线性回归与 Ridge 回归 | `python lesson_07_08_linear_models/01_linear_regression_diabetes.py` |
| `02_logistic_regression_breast_cancer.py` | 乳腺癌数据：逻辑回归、概率、阈值、标准化 | `python lesson_07_08_linear_models/02_logistic_regression_breast_cancer.py` |
| `03_l1_l2_huber_regression.py` | L1、L2、Huber 对异常值的影响 | `python lesson_07_08_linear_models/03_l1_l2_huber_regression.py` |

## 运行前准备

在 `codes/` 目录下执行：

```bash
uv sync
```

或者安装依赖：

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```
