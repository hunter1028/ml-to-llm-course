"""课时 13-14：sklearn GBDT baseline。"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split


def load_data(random_state: int = 42):
    X, y = load_breast_cancer(return_X_y=True)
    return train_test_split(X, y, test_size=0.25, random_state=random_state, stratify=y)


def main() -> None:
    X_train, X_test, y_train, y_test = load_data()

    model = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    print("AUC:", round(roc_auc_score(y_test, proba), 4))
    print(classification_report(y_test, pred))


if __name__ == "__main__":
    main()
