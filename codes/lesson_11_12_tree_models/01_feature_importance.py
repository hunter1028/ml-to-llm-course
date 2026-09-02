"""课时 11-12：Impurity Importance 与 Permutation Importance 对比。"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split


def main() -> None:
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=1,
    )
    rf.fit(X_train, y_train)

    importance = pd.DataFrame({
        "feature": feature_names,
        "impurity_importance": rf.feature_importances_,
    }).sort_values("impurity_importance", ascending=False)
    print("impurity importance:")
    print(importance.head(10))

    perm = permutation_importance(
        rf, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=1
    )
    perm_importance = pd.DataFrame({
        "feature": feature_names,
        "perm_importance": perm.importances_mean,
    }).sort_values("perm_importance", ascending=False)
    print("\npermutation importance:")
    print(perm_importance.head(10))


if __name__ == "__main__":
    main()
