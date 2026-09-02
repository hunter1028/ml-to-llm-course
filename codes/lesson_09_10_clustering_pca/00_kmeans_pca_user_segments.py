"""课时 09-10：K-Means + PCA 用户分群示例。"""

import os

# 避免部分 macOS 环境下 joblib 探测 CPU 核数时输出无关警告。
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def make_user_data(n: int = 600, random_state: int = 42) -> pd.DataFrame:
    """构造三类行为差异明显的用户数据。"""
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
    df = pd.DataFrame(
        X,
        columns=[
            "login_days_30d",
            "pay_amount_30d",
            "order_cnt_30d",
            "coupon_use_rate",
        ],
    )
    df["login_days_30d"] = df["login_days_30d"].clip(0, 30)
    df["order_cnt_30d"] = df["order_cnt_30d"].clip(0)
    df["coupon_use_rate"] = df["coupon_use_rate"].clip(0, 1)
    return df


def prepare_features(df: pd.DataFrame):
    df_model = df.copy()
    df_model["pay_amount_30d"] = np.log1p(df_model["pay_amount_30d"])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_model)
    return X_scaled, scaler


def main() -> None:
    df = make_user_data()
    X_scaled, _ = prepare_features(df)

    print("k  silhouette  inertia")
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        score = silhouette_score(X_scaled, labels)
        print(f"{k:<2d} {score:10.3f} {km.inertia_:8.1f}")

    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    df["cluster"] = km.fit_predict(X_scaled)

    profile = df.groupby("cluster").mean().round(2)
    print("\ncluster profile:")
    print(profile)

    pca = PCA(n_components=2)
    pca.fit_transform(X_scaled)
    explained_ratio = pca.explained_variance_ratio_
    print("\nexplained variance by component:", explained_ratio.round(3))
    print("total explained variance:", round(explained_ratio.sum(), 3))


if __name__ == "__main__":
    main()
