"""课时 13-14：XGBoost 训练模板。"""

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise SystemExit("请先安装 xgboost：python -m pip install xgboost") from exc


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=42,
        n_jobs=1,
    )

    xgb.fit(X_train, y_train)
    proba = xgb.predict_proba(X_test)[:, 1]
    print("XGBoost AUC:", round(roc_auc_score(y_test, proba), 4))


if __name__ == "__main__":
    main()
