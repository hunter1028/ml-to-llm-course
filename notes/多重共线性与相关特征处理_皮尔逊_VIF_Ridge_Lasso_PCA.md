# 多重共线性与相关特征处理：Pearson、VIF、Ridge、Lasso 与 PCA

## 1. 先理解问题：为什么预测正确，权重却可能很奇怪

多重共线性（Multicollinearity）是指：两个或多个输入特征携带了大量重复信息，一个特征可以被其他特征近似表示。

以房价预测为例：

- $x_1$：套内面积；
- $x_2$：建筑面积；
- 在当前数据中，$x_2\approx1.2x_1$；
- 房价的真实规律近似为 $y=3x_1$。

下面这个模型看起来很奇怪：

$$
\hat y=15x_1-10x_2
$$

但是把 $x_2\approx1.2x_1$ 代入：

$$
\hat y\approx15x_1-10\times1.2x_1=3x_1
$$

当 $x_1=80$、$x_2=96$ 时：

$$
\hat y=15\times80-10\times96=240
$$

预测结果可以完全正确，但不能据此解释为“建筑面积越大，房价越低”。这里的大正权重和大负权重主要是在互相补偿。

模型真正识别到的可能只是：

$$
w_1+1.2w_2\approx3
$$

满足这个关系的权重组合有很多，因此训练数据稍有变化，$w_1$ 和 $w_2$ 就可能大幅波动。

如果未来房屋的公摊比例发生变化，例如仍然是 $x_1=80$，但 $x_2=120$，上面的模型会预测：

$$
\hat y=15\times80-10\times120=0
$$

这说明共线性不仅会破坏参数解释，还可能使模型在特征关系发生变化时变得脆弱。

> 预测效果好，不等于每个权重都可以单独解释；权重有方向，也不等于存在因果关系。

---

## 2. Pearson 相关系数：先检查两两线性相关

Pearson 相关系数用于衡量两个变量之间的线性相关程度：

$$
r_{xy}=\frac{\operatorname{cov}(x,y)}{\sigma_x\sigma_y}
$$

其取值范围是 $[-1,1]$：

| 取值 | 含义 |
|---:|---|
| 接近 $1$ | 高度正线性相关 |
| 接近 $-1$ | 高度负线性相关 |
| 接近 $0$ | 没有明显线性相关，但仍可能存在非线性关系 |

排查共线性时看的是**特征与特征之间**的相关性，通常检查 $|r|$。工程中常用 $0.8$ 或 $0.9$ 作为提醒阈值，但这不是必须遵守的数学分界线。

```python
import pandas as pd

corr_matrix = X_train.corr(method="pearson")
print(corr_matrix.round(2))
```

注意事项：

1. 正相关和负相关都可能产生共线性，因此要看绝对值 $|r|$。
2. 高相关只是提醒，不代表必须机械删除其中一个特征。
3. Pearson 主要发现两两线性关系，发现不了所有非线性依赖。
4. 相关性分析应只使用训练集，不能根据测试集结果反过来选择特征。
5. 特征与目标 $y$ 的相关性只能作为辅助信息，不能简单地“谁和 $y$ 相关低就删除谁”，因为组合、交互和非线性关系可能仍然有价值。

### Pearson 检查不到的一类问题

假设：

$$
x_3=x_1+x_2
$$

此时 $x_3$ 可以被另外两个特征完全表示。即使某些两两相关系数没有超过人工阈值，三个特征放在一起仍然存在严重共线性。因此还需要 VIF、矩阵条件数或 SVD 等方法检查多个特征之间的联合关系。

---

## 3. VIF：检查一个特征能否被其他特征共同解释

VIF（Variance Inflation Factor，方差膨胀因子）的思路是：

> 选择第 $j$ 个特征作为被预测对象，用其余所有特征预测它。

如果预测得到的决定系数为 $R_j^2$，那么：

$$
VIF_j=\frac{1}{1-R_j^2}
$$

当 $R_j^2$ 很高时，说明第 $j$ 个特征可以被其他特征共同推算出来，它提供的独立信息较少。

| VIF | 常见经验解释 |
|---:|---|
| 接近 1 | 基本没有明显共线性 |
| 大于 5 | 值得进一步检查 |
| 大于 10 | 通常认为共线性比较严重 |

这些阈值同样是经验规则，不应代替业务判断。

```python
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

X_vif = X_train.select_dtypes(include="number").dropna().copy()
X_with_const = add_constant(X_vif)

vif_table = pd.DataFrame({
    "feature": X_vif.columns,
    "VIF": [
        variance_inflation_factor(X_with_const.values, i)
        for i in range(1, X_with_const.shape[1])
    ],
})

print(vif_table.sort_values("VIF", ascending=False))
```

使用 VIF 时还应注意：

- 类别特征经过独热编码后，应避免保留全部类别和截距形成“虚拟变量陷阱”；
- 极高的 VIF 提示要调查，不表示一定删除；
- 如果业务必须同时控制多个相关变量，可以保留变量，但需要承认单个系数的估计不稳定；
- VIF 更常用于关注统计推断和权重解释的线性模型。

---

## 4. 发现高度相关后，应该如何处理

### 4.1 删除冗余特征

如果两个特征表达几乎相同的信息，可以保留一个。例如“月收入”和由它乘以 12 得到的“年收入”没有必要同时输入模型。

选择保留哪个特征时，优先考虑：

1. 业务含义是否更直接；
2. 测量是否更准确；
3. 缺失值是否更少；
4. 线上预测时是否稳定可得；
5. 是否存在数据泄漏；
6. 在交叉验证中是否表现更稳定。

不要只因为某个特征与目标变量的 Pearson 相关系数稍高，就武断地保留它。

### 4.2 合并或重新构造特征

高度相关的原始特征有时可以转换成更有业务意义的指标。例如：

$$
得房率=\frac{套内面积}{建筑面积}
$$

可以根据问题改为使用“建筑面积 + 得房率”，而不是直接放入两个高度重复的面积特征。

特征构造后仍然要重新检查相关性和验证集表现，不能假设新特征一定更好。

---

## 5. Ridge：保留特征，同时限制权重过度补偿

Ridge 回归在线性回归损失后加入 L2 惩罚：

$$
L=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2+\lambda\sum_{j=1}^{n}w_j^2
$$

当权重变得很大时，$w_j^2$ 会明显增大损失。因此 Ridge 会抑制“一个权重特别大、另一个权重反方向特别大”的补偿现象。

Ridge 的特点：

- 通常不会把权重精确压成 0；
- 可以保留全部特征；
- 经常能改善相关特征下的参数稳定性和泛化能力；
- 适合更重视预测、又不希望轻易删除特征的情况。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

ridge_model = make_pipeline(
    StandardScaler(),
    Ridge(alpha=1.0),
)

ridge_model.fit(X_train, y_train)
y_pred = ridge_model.predict(X_test)
```

由于 L1/L2 惩罚直接作用于权重，使用 Ridge、Lasso 前通常应先标准化特征，否则单位较大的特征和单位较小的特征会受到不公平的约束。

在 sklearn 的 `Ridge` 中，`alpha` 越大，正则越强。最终取值应通过交叉验证选择，而不是固定照抄 `1.0`。

---

## 6. Lasso：在约束权重的同时做稀疏选择

Lasso 回归加入 L1 惩罚：

$$
L=\frac{1}{m}\sum_{i=1}^{m}(y_i-\hat y_i)^2+\lambda\sum_{j=1}^{n}|w_j|
$$

L1 惩罚可能把一部分权重直接压缩为 0，因此具有自动特征选择效果。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

lasso_model = make_pipeline(
    StandardScaler(),
    Lasso(alpha=0.01),
)

lasso_model.fit(X_train, y_train)
```

但是，当多个特征高度相关时，Lasso 可能留下其中一个、删除其他特征。训练样本稍有变化，它选择的特征也可能变化。因此：

- 权重为 0 不代表该变量在业务上毫无作用；
- Lasso 做出的特征选择未必稳定；
- 如果既希望稀疏，又希望相关特征的选择更平稳，可以进一步考虑 Elastic Net（L1 + L2）。

---

## 7. PCA：把相关特征压缩成新的综合方向

PCA（Principal Component Analysis，主成分分析）通过旋转坐标系，把原始特征转换成互相正交的主成分：

$$
X\rightarrow Z
$$

其中新的主成分 $Z_1,Z_2,\ldots$ 彼此不相关，并按照能够保留的数据方差从高到低排列。

```python
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

pca_model = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    LinearRegression(),
)

pca_model.fit(X_train, y_train)
```

`n_components=0.95` 表示保留能够解释约 95% 数据方差的主成分数量。

PCA 的优缺点：

| 优点 | 缺点 |
|---|---|
| 消除主成分之间的线性相关 | 新特征是原始特征的组合，业务解释变差 |
| 可以压缩大量重复特征 | 保留数据方差不等于一定保留最有预测价值的信息 |
| 有时能改善数值稳定性和训练速度 | 必须把标准化和 PCA 放进训练管道，避免数据泄漏 |

因此 PCA 更适合重视降维和预测效率、能够接受可解释性下降的场景。

---

## 8. 几种方法如何选择

| 方法 | 主要作用 | 是否删除原始特征 | 可解释性 | 典型场景 |
|---|---|---:|---:|---|
| Pearson | 两两线性相关的初步筛查 | 否 | 高 | 快速发现明显重复特征 |
| VIF | 检查一个特征能否被其余特征共同解释 | 否 | 高 | 线性模型参数解释、统计推断 |
| 业务删除或合并 | 去掉明确重复信息 | 是 | 通常较高 | 月收入/年收入、同义指标 |
| Ridge | 缩小权重，缓解不稳定 | 通常否 | 中 | 重视预测，希望保留全部变量 |
| Lasso | 缩小权重并产生稀疏选择 | 可能 | 中 | 特征较多，希望得到稀疏模型 |
| PCA | 转换为少量互不相关的综合特征 | 替换 | 较低 | 高维压缩、预测优先 |

一个实用流程是：

```text
业务检查与数据质量检查
        ↓
只在训练集上计算 Pearson 相关矩阵
        ↓
用 VIF 等方法检查多个特征的联合共线性
        ↓
明确重复：删除或合并
        ↓
希望保留变量：Ridge
希望稀疏选择：Lasso / Elastic Net
高维压缩且可接受解释下降：PCA
        ↓
用交叉验证比较预测效果、权重稳定性和业务可解释性
```

---

## 9. 最容易产生的几个误解

### 误解一：两个特征相关，就必须删除一个

不一定。是否删除取决于建模目标、业务含义、数据质量和验证结果。Ridge 也可以在保留变量的同时缓解问题。

### 误解二：特征之间不相关，就一定没有共线性

不一定。Pearson 是两两线性检查，多个特征组合起来仍可能近似表示另一个特征。

### 误解三：预测准确，所以权重解释一定正确

不一定。高度相关的特征可以用多组权重得到相近预测，单个权重可能只是数学补偿。

### 误解四：Lasso 删除了某个特征，说明它没有业务价值

不一定。它可能只是与另一个被保留的特征提供了相近信息。

### 误解五：树模型完全不受相关特征影响

树模型的预测通常不像线性回归系数那样直接受到共线性影响，但相关特征仍可能互相替代，使特征重要性分散或变得不稳定。因此解释模型结果时仍应检查相关特征。

---

## 10. 结论

多重共线性的核心问题不是“模型一定无法预测”，而是：

1. 单个权重可能大幅波动；
2. 权重方向可能不符合业务直觉；
3. 参数解释和显著性判断可能失真；
4. 当特征之间的关系发生变化时，模型可能变得脆弱。

Pearson 适合初步发现两两线性相关，VIF 适合检查多个特征之间的联合关系；发现问题后，可以根据目标选择删除、合并、Ridge、Lasso 或 PCA。最终不能只看一个阈值，而要同时比较验证集表现、参数稳定性和业务解释是否合理。
