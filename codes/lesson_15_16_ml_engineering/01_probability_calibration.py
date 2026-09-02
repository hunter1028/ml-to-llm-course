"""课时 15-16：概率校准示例。"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split


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

    base_model = GradientBoostingClassifier(random_state=42)
    calibrated = CalibratedClassifierCV(base_model, method="isotonic", cv=3)
    calibrated.fit(X_train, y_train)

    raw = model.predict_proba(X_test)[:, 1]
    cal = calibrated.predict_proba(X_test)[:, 1]

    print("raw auc:", round(roc_auc_score(y_test, raw), 4))
    print("calibrated auc:", round(roc_auc_score(y_test, cal), 4))
    print("raw brier:", round(brier_score_loss(y_test, raw), 4))
    print("calibrated brier:", round(brier_score_loss(y_test, cal), 4))


if __name__ == "__main__":
    main()
