"""
Model training, cross-validation, benchmarking, and export script.
Trains multiple classification models, evaluates them using Stratified 5-Fold CV,
and serializes the champion model pipeline to disk.
"""

import json
import os
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
import joblib

from src.data_loader import load_raw_dataset, prepare_features_and_target, FEATURE_COLUMNS


def evaluate_models_cv(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
    """
    Evaluates candidate models using 5-Fold Stratified Cross-Validation.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    candidate_models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    }

    results: Dict[str, Dict[str, float]] = {}

    for name, clf in candidate_models.items():
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", clf),
        ])

        acc_scores = []
        prec_scores = []
        rec_scores = []
        f1_scores = []
        auc_scores = []

        for train_idx, val_idx in skf.split(X, y):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

            pipe.fit(X_tr, y_tr)
            y_pred = pipe.predict(X_val)
            y_prob = pipe.predict_proba(X_val)[:, 1]

            acc_scores.append(accuracy_score(y_val, y_pred))
            prec_scores.append(precision_score(y_val, y_pred, zero_division=0))
            rec_scores.append(recall_score(y_val, y_pred, zero_division=0))
            f1_scores.append(f1_score(y_val, y_pred, zero_division=0))
            auc_scores.append(roc_auc_score(y_val, y_prob))

        results[name] = {
            "Accuracy": float(np.mean(acc_scores)),
            "Precision": float(np.mean(prec_scores)),
            "Recall": float(np.mean(rec_scores)),
            "F1_Score": float(np.mean(f1_scores)),
            "ROC_AUC": float(np.mean(auc_scores)),
        }
        print(f"[{name}] Acc: {results[name]['Accuracy']:.3f} | F1: {results[name]['F1_Score']:.3f} | AUC: {results[name]['ROC_AUC']:.3f}")

    return results


def train_and_export_champion(csv_path: str, models_dir: str = "models") -> None:
    """
    Executes benchmark, selects champion model, fits on entire dataset,
    and exports artifacts.
    """
    os.makedirs(models_dir, exist_ok=True)

    df_raw = load_raw_dataset(csv_path)
    X, y = prepare_features_and_target(df_raw)

    if y is None:
        raise ValueError("Target variable 'rainfall' not found in dataset.")

    print(f"Dataset loaded: {len(X)} records across {len(FEATURE_COLUMNS)} features.")
    print("Evaluating models with Stratified 5-Fold Cross Validation...")
    benchmark_results = evaluate_models_cv(X, y)

    # Pick champion based on ROC_AUC and F1
    champion_name = max(benchmark_results.keys(), key=lambda k: benchmark_results[k]["ROC_AUC"])
    print(f"\nChampion model selected: {champion_name}")

    if champion_name == "Random Forest":
        champion_clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    elif champion_name == "Gradient Boosting":
        champion_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.08, max_depth=3, random_state=42)
    else:
        champion_clf = LogisticRegression(max_iter=1000, random_state=42)

    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", champion_clf),
    ])

    final_pipeline.fit(X, y)

    # Extract feature importance if available
    feature_importances = {}
    if hasattr(champion_clf, "feature_importances_"):
        for col, imp in zip(FEATURE_COLUMNS, champion_clf.feature_importances_):
            feature_importances[col] = float(imp)
    elif hasattr(champion_clf, "coef_"):
        for col, coef in zip(FEATURE_COLUMNS, champion_clf.coef_[0]):
            feature_importances[col] = float(abs(coef))

    # Save pipeline
    model_path = os.path.join(models_dir, "rainfall_pipeline.joblib")
    joblib.dump(final_pipeline, model_path)
    print(f"Pipeline saved to: {model_path}")

    # Save benchmark metadata
    metadata = {
        "champion_model": champion_name,
        "features": FEATURE_COLUMNS,
        "benchmark_results": benchmark_results,
        "feature_importances": feature_importances,
        "training_samples": len(X),
        "target_distribution": {
            "Rain (1)": int((y == 1).sum()),
            "No Rain (0)": int((y == 0).sum()),
        },
    }
    meta_path = os.path.join(models_dir, "benchmark_results.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Benchmark metadata saved to: {meta_path}")


if __name__ == "__main__":
    train_and_export_champion("dataset1.csv")
