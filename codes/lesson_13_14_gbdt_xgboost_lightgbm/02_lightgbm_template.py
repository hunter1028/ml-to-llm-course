"""课时 13-14：LightGBM 训练模板。"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from lightgbm import LGBMClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 lightgbm：python -m pip install lightgbm") from exc


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    lgb = LGBMClassifier(
        n_estimators=600,
        learning_rate=0.03,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        objective="binary",
        random_state=42,
        n_jobs=1,
        verbose=-1,
    )

    lgb.fit(X_train, y_train)
    proba = lgb.predict_proba(X_test)[:, 1]
    print("LightGBM AUC:", round(roc_auc_score(y_test, proba), 4))


if __name__ == "__main__":
    main()
