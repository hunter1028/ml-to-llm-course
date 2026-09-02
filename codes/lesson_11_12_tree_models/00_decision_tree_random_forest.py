"""课时 11-12：决策树、随机森林和 Extra Trees 对比。"""

from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text


def train_models(random_state: int = 42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=random_state, stratify=y
    )

    models = {
        "tree_deep": DecisionTreeClassifier(random_state=random_state),
        "tree_pruned": DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=10, random_state=random_state
        ),
        "forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
            oob_score=True,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=200,
            min_samples_leaf=3,
            random_state=random_state,
            n_jobs=1,
        ),
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models, X_test, y_test, feature_names


def main() -> None:
    models, X_test, y_test, feature_names = train_models()

    for name, model in models.items():
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        print(name)
        print("accuracy:", round(accuracy_score(y_test, pred), 4))
        print("auc:", round(roc_auc_score(y_test, proba), 4))
        if hasattr(model, "oob_score_"):
            print("oob:", round(model.oob_score_, 4))
        print()

    small_tree = models["tree_pruned"]
    print(export_text(small_tree, feature_names=list(feature_names), max_depth=3))


if __name__ == "__main__":
    main()
