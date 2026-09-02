"""课时 37-38：按最低召回约束选择阈值。"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def choose_threshold(y_true, proba, min_recall=0.95):
    candidates = []
    for threshold in np.linspace(0.01, 0.99, 99):
        pred = (proba >= threshold).astype(int)
        recall = recall_score(y_true, pred)
        precision = precision_score(y_true, pred, zero_division=0)
        if recall >= min_recall:
            candidates.append((f1_score(y_true, pred), threshold, precision, recall))
    if not candidates:
        raise ValueError("没有阈值满足最低召回要求")
    return max(candidates)


def main() -> None:
    y_true = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 0])
    proba = np.array([0.92, 0.44, 0.81, 0.63, 0.20, 0.11, 0.71, 0.35, 0.58, 0.08])
    f1, threshold, precision, recall = choose_threshold(y_true, proba, min_recall=0.8)
    print({
        "threshold": round(float(threshold), 2),
        "precision": round(float(precision), 3),
        "recall": round(float(recall), 3),
        "f1": round(float(f1), 3),
    })


if __name__ == "__main__":
    main()
