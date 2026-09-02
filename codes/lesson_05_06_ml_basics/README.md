# 课时 05-06：机器学习基础范式与通用原理代码

本目录放课时 05-06 配套的可运行代码。讲义正文保留关键代码片段，完整脚本统一放在这里。

## 文件说明

| 文件 | 对应内容 | 运行方式 |
|---|---|---|
| `00_environment_check.py` | 检查 numpy、pandas、scikit-learn 版本 | `python lesson_05_06_ml_basics/00_environment_check.py` |
| `01_manual_train_val_test_split.py` | 手动划分训练集、验证集、测试集，并检查索引是否重叠 | `python lesson_05_06_ml_basics/01_manual_train_val_test_split.py` |
| `02_baseline_and_ridge.py` | 训练普通线性回归 baseline，并比较不同 Ridge 正则强度 | `python lesson_05_06_ml_basics/02_baseline_and_ridge.py` |
| `03_manual_l2_loss.py` | 手写 `MSE + lambda * L2 penalty` | `python lesson_05_06_ml_basics/03_manual_l2_loss.py` |

## 运行前准备

在 `codes/` 目录下执行：

```bash
uv sync
```

或者安装依赖：

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```
