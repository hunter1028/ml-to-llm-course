"""课时 15-16：用收益函数选择分类阈值。"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split


def profit_at_threshold(y_true, score, threshold: float) -> float:
    pred = score >= threshold
    tp = np.sum((pred == 1) & (y_true == 1))
    fp = np.sum((pred == 1) & (y_true == 0))
    fn = np.sum((pred == 0) & (y_true == 1))

    gain_tp = 100
    cost_fp = 8
    cost_fn = 120
    action_cost = 2 * np.sum(pred)
    return tp * gain_tp - fp * cost_fp - fn * cost_fn - action_cost


def main() -> None:
    X, y = make_classification(
        n_samples=5000,
        n_features=20,
        n_informative=8,
        weights=[0.92, 0.08],
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    print("AUC:", round(roc_auc_score(y_test, proba), 4))

    rows = []
    for t in np.linspace(0.05, 0.95, 19):
        pred = proba >= t
        rows.append({
            "threshold": round(float(t), 2),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "profit": profit_at_threshold(y_test, proba, t),
            "actions": int(pred.sum()),
        })

    table = pd.DataFrame(rows)
    print(table.round(3))
    print("\nbest:")
    print(max(rows, key=lambda x: x["profit"]))


if __name__ == "__main__":
    main()
