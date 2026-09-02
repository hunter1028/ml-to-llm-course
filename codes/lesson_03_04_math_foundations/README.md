# 课时 03-04：AI 核心数学基础代码

本目录放课时 03-04 配套的可运行代码。讲义正文保留关键代码片段，完整脚本统一放在这里。

## 文件说明

| 文件 | 对应内容 | 运行方式 |
|---|---|---|
| `00_vector_matrix_prediction.py` | 单个样本的向量与矩阵预测 | `python lesson_03_04_math_foundations/00_vector_matrix_prediction.py` |
| `01_batch_prediction.py` | 用矩阵一次计算多个样本预测值 | `python lesson_03_04_math_foundations/01_batch_prediction.py` |
| `02_cosine_similarity.py` | 用余弦相似度比较 query 和文档向量 | `python lesson_03_04_math_foundations/02_cosine_similarity.py` |
| `03_normal_distribution.py` | 画出不同标准差下的正态分布曲线 | `python lesson_03_04_math_foundations/03_normal_distribution.py` |
| `04_gradient_descent_demo.py` | 一维函数上的梯度下降直觉 | `python lesson_03_04_math_foundations/04_gradient_descent_demo.py` |

## 运行前准备

在 `codes/` 目录下执行：

```bash
uv sync
```

或者安装依赖：

```bash
python -m pip install numpy matplotlib
```

如果在没有图形界面的环境中运行，正态分布脚本会把图保存到本目录的 `outputs/` 子目录。
