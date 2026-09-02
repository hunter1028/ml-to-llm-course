"""课时 37-38：文本、类别、数值特征联合建模 Pipeline。"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 xgboost：python -m pip install xgboost") from exc


text_col = "text"
cat_cols = ["channel", "product_type", "customer_level"]
num_cols = ["complaint_count_30d", "order_amount"]


def make_demo_data() -> pd.DataFrame:
    rows = [
        ["充电器发热并有焦味", "app", "electronics", "gold", 2, 299, 1],
        ["想咨询优惠券使用规则", "web", "coupon", "silver", 0, 99, 0],
        ["包裹显示签收但没有收到", "app", "logistics", "gold", 3, 520, 1],
        ["商品很好，想再次购买", "web", "clothes", "bronze", 0, 120, 0],
        ["重复扣款要求退款", "phone", "payment", "gold", 4, 899, 1],
        ["修改收货地址", "app", "logistics", "silver", 1, 80, 0],
        ["孩子使用后皮肤过敏", "phone", "baby", "gold", 2, 180, 1],
        ["普通发票怎么开", "web", "invoice", "bronze", 0, 60, 0],
    ] * 12
    return pd.DataFrame(
        rows,
        columns=[text_col] + cat_cols + num_cols + ["need_escalation"],
    )


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("text", TfidfVectorizer(max_features=20000, ngram_range=(1, 2), min_df=2), text_col),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), num_cols),
        ]
    )

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
    )

    return Pipeline([
        ("features", preprocess),
        ("model", model),
    ])


def main() -> None:
    df = make_demo_data()
    train_df, valid_df = train_test_split(
        df, test_size=0.25, random_state=42, stratify=df["need_escalation"]
    )

    pipeline = build_pipeline()
    pipeline.fit(train_df, train_df["need_escalation"])
    proba = pipeline.predict_proba(valid_df)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print(classification_report(valid_df["need_escalation"], pred))
    print("proba sample:", proba[:5].round(3))


if __name__ == "__main__":
    main()
