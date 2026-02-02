from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple, List

import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass(frozen=True)
class TrainConfig:
    target_col: str = "target"
    test_size: float = 0.2
    random_state: int = 42
    n_jobs: int = -1
    cv_folds: int = 5
    scoring: str = "f1_weighted"  # better default for multiclass


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
    )


def build_model(preprocessor: ColumnTransformer) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", KNeighborsClassifier()),
        ]
    )


def hyperparameter_search(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series, cfg: TrainConfig) -> GridSearchCV:
    # Cap K so it never exceeds training size (general safety)
    n_train = len(X_train)
    base_ks: List[int] = [3, 5, 7, 9, 11, 15, 21, 31]
    ks = [k for k in base_ks if k < n_train]  # must be < n_train

    param_grid: Dict[str, Any] = {
        "model__n_neighbors": ks,
        "model__weights": ["uniform", "distance"],
        "model__metric": ["minkowski"],
        "model__p": [1, 2],  # 1=manhattan, 2=euclidean
        "model__algorithm": ["auto"],
    }

    cv = StratifiedKFold(n_splits=cfg.cv_folds, shuffle=True, random_state=cfg.random_state)

    search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=cfg.scoring,
        cv=cv,
        n_jobs=cfg.n_jobs,
        refit=True,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)
    return search


def evaluate_classifier(model: BaseEstimator, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    y_pred = model.predict(X_test)

    metrics: Dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "F1": float(
            f1_score(
                y_test,
                y_pred,
                average="binary" if y_test.nunique() == 2 else "weighted",
            )
        ),
        "Confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification_report": classification_report(y_test, y_pred, digits=4),
    }

    # AUC: only meaningful for binary unless you configure multiclass AUC explicitly
    if y_test.nunique() == 2 and hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = float(roc_auc_score(y_test, proba))

    return metrics


def train_knn_classifier(df: pd.DataFrame, cfg: TrainConfig) -> Tuple[BaseEstimator, Dict[str, Any]]:
    if cfg.target_col not in df.columns:
        raise ValueError(f"target_col='{cfg.target_col}' not found in df columns.")

    y = df[cfg.target_col]
    X = df.drop(columns=[cfg.target_col])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.test_size,
        random_state=cfg.random_state,
        stratify=y if y.nunique() > 1 else None,
    )

    preprocessor = build_preprocessor(X_train)
    pipeline = build_model(preprocessor)

    search = hyperparameter_search(pipeline, X_train, y_train, cfg)
    best_model = search.best_estimator_

    results = {
        "best_params": search.best_params_,
        "best_cv_score": float(search.best_score_),
        "test_metrics": evaluate_classifier(best_model, X_test, y_test),
    }

    return best_model, results


if __name__ == "__main__":

    df = pd.read_csv("iris_knn.csv")

    cfg = TrainConfig(target_col="species", scoring="f1_weighted")
    model, results = train_knn_classifier(df, cfg)

    print("\nBest CV Score:")
    print(results["best_cv_score"])

    print("\nTest Metrics:")
    for k, v in results["test_metrics"].items():
        if k == "Classification_report":
            print("\nClassification Report:\n", v)
        else:
            print(f"{k}: {v}")