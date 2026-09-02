# 课程示例代码

代码按课时分目录存放，避免后续脚本越来越多时堆在根目录。

## 目录结构

```text
codes/
├── lesson_00_intro/              两个小 AI 例子
├── lesson_05_06_ml_basics/       课时 05-06：机器学习基础范式与通用原理
├── lesson_07_08_linear_models/   课时 07-08：线性模型理论与实战
├── main.py                       代码目录说明入口
├── pyproject.toml                Python 依赖配置
└── uv.lock                       uv 锁文件
```

## 安装依赖

推荐在 `codes/` 目录下使用 uv：

```bash
cd /Users/liuzhicheng/Documents/01pgm/ml-course/codes
uv sync
```

也可以使用 pip：

```bash
python -m pip install numpy pandas matplotlib scikit-learn
```

## 两个小 AI 例子

### 线性回归预测房价

```bash
python lesson_00_intro/01_ml_linear_regression.py
```

### 多层感知机识别手写数字

```bash
python lesson_00_intro/02_dl_mlp_digits.py
```

## 课时 05-06：机器学习基础范式与通用原理

### 环境检查

```bash
python lesson_05_06_ml_basics/00_environment_check.py
```

### 手动划分训练集、验证集和测试集

```bash
python lesson_05_06_ml_basics/01_manual_train_val_test_split.py
```

### Baseline 与 Ridge 正则对比

```bash
python lesson_05_06_ml_basics/02_baseline_and_ridge.py
```

### 手写 L2 损失

```bash
python lesson_05_06_ml_basics/03_manual_l2_loss.py
```

## 课时 07-08：线性模型理论与实战

### 正规方程、矩阵求逆与 lstsq

```bash
python lesson_07_08_linear_models/00_normal_equation_demo.py
```

### 糖尿病数据：线性回归与 Ridge 回归

```bash
python lesson_07_08_linear_models/01_linear_regression_diabetes.py
```

### 乳腺癌数据：逻辑回归、概率与阈值

```bash
python lesson_07_08_linear_models/02_logistic_regression_breast_cancer.py
```

### L1、L2 与 Huber 回归对比

```bash
python lesson_07_08_linear_models/03_l1_l2_huber_regression.py
```

可在文件开头直接调整：

- `USE_OUTLIER`：是否加入异常值；
- `HUBER_EPSILON`：Huber 损失从平方区间切换到线性区间的阈值；
- `PREDICT_AREA`：需要预测的房屋面积。

如果在没有图形界面的环境中运行，脚本会把图保存到对应课时目录的 `outputs/` 子目录。
