"""课时 09-10：DBSCAN 识别异常群体示例。"""

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


def make_user_data(n: int = 600, random_state: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    high_value = np.column_stack([
        rng.normal(24, 4, n // 3),
        rng.lognormal(7.7, 0.35, n // 3),
        rng.normal(18, 4, n // 3),
        rng.normal(0.10, 0.04, n // 3),
    ])
    price_sensitive = np.column_stack([
        rng.normal(12, 4, n // 3),
        rng.lognormal(6.0, 0.45, n // 3),
        rng.normal(5, 2, n // 3),
        rng.normal(0.72, 0.10, n // 3),
    ])
    silent = np.column_stack([
        rng.normal(3, 2, n // 3),
        rng.lognormal(4.8, 0.5, n // 3),
        rng.normal(1, 1, n // 3),
        rng.normal(0.30, 0.15, n // 3),
    ])
    X = np.vstack([high_value, price_sensitive, silent])
    df = pd.DataFrame(X, columns=[
        "login_days_30d", "pay_amount_30d", "order_cnt_30d", "coupon_use_rate"
    ])
    df["login_days_30d"] = df["login_days_30d"].clip(0, 30)
    df["order_cnt_30d"] = df["order_cnt_30d"].clip(0)
    df["coupon_use_rate"] = df["coupon_use_rate"].clip(0, 1)
    return df


def prepare_features(df: pd.DataFrame):
    df_model = df.copy()
    df_model["pay_amount_30d"] = np.log1p(df_model["pay_amount_30d"])
    return StandardScaler().fit_transform(df_model), None


def print_dbscan_result(df: pd.DataFrame, X_scaled: np.ndarray, eps: float, min_samples: int) -> None:
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)
    result = df.assign(dbscan_label=labels)

    print(f"\neps={eps}, min_samples={min_samples}")
    print("label counts:")
    print(result["dbscan_label"].value_counts().sort_index())
    print("cluster profile:")
    print(result.groupby("dbscan_label").mean().round(2))


def main() -> None:
    df = make_user_data()
    X_scaled, _ = prepare_features(df)

    # min_samples 可先设为 特征数 * 2。
    min_samples = X_scaled.shape[1] * 2

    # 第一步：先用 NearestNeighbors 计算近邻距离。
    # 这一步不做聚类，只是为了给 DBSCAN 的 eps 找一个参考范围。
    nn = NearestNeighbors(n_neighbors=min_samples)
    nn.fit(X_scaled)
    distances, _ = nn.kneighbors(X_scaled)

    # 第二步：取每个样本到“第 min_samples 个近邻”的距离。
    # 如果这个距离很大，说明这个样本周围比较稀疏。
    k_distance = np.sort(distances[:, -1])
    print("Step 1: use NearestNeighbors to estimate a useful eps range")
    print(f"min_samples: {min_samples}")
    print("largest k-distances:", k_distance[-10:].round(3))
    print("k-distance quantiles 50%/90%/95%/99%:", np.quantile(k_distance, [0.5, 0.9, 0.95, 0.99]).round(3))

    print("\nStep 2: run DBSCAN with candidate eps values")
    # 根据上面的 k-distance 分布，先试一个偏小的 eps。
    print_dbscan_result(df, X_scaled, eps=0.55, min_samples=min_samples)

    # 再试一个偏大的 eps，观察相邻密度区域被合并的情况。
    print_dbscan_result(df, X_scaled, eps=0.90, min_samples=min_samples)


if __name__ == "__main__":
    main()
