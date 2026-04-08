from __future__ import annotations

"""Entrainement du modele supervise (regression sur Amount)."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from data_utils import load_dataset
from preprocess import DATE_COL_DEFAULT, TARGET_COL_DEFAULT, build_preprocessor, split_features_target



MODEL_OUTPUT_PATH = "artifacts/regression_model.pkl"
METRICS_OUTPUT_PATH = "reports/regression_metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def save_pipeline_visuals(pipeline, txt_path: str) -> None:
    """Sauvegarde une representation texte du pipeline."""
    path_txt = Path(txt_path)
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    path_txt.write_text(str(pipeline))


def time_based_split(df, date_col: str, test_size: float):
    """Split temporel pour eviter la fuite d'information."""
    df = df.sort_values(date_col)
    split_idx = int(len(df) * (1 - test_size))
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    return train_df, test_df


def main() -> None:
    """Pipeline complet: load -> split -> train -> eval -> save."""
    # Chargement du dataset local.
    df = load_dataset()

    if TARGET_COL_DEFAULT not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COL_DEFAULT}' not found in dataset."
        )

    # Securise la cible: conversion numerique + suppression des NaN.
    df[TARGET_COL_DEFAULT] = pd.to_numeric(df[TARGET_COL_DEFAULT], errors="coerce")
    before_rows = len(df)
    df = df.dropna(subset=[TARGET_COL_DEFAULT])
    dropped_rows = before_rows - len(df)
    if dropped_rows:
        print(f"Dropped {dropped_rows} rows with missing target.")
    if len(df) == 0:
        raise ValueError("All rows have missing target values after cleaning.")

    # Split temporel si la colonne date est presente (evite la fuite temporelle).
    if DATE_COL_DEFAULT in df.columns:
        train_df, test_df = time_based_split(df, DATE_COL_DEFAULT, TEST_SIZE)
    else:
        # Sinon split aleatoire classique.
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )

    X_train, y_train, spec = split_features_target(train_df)
    X_test, y_test, _ = split_features_target(test_df)

    # Preprocessing: imputation + encodage.
    preprocessor = build_preprocessor(spec)

    # Baseline simple pour comparer les performances.
    baseline = DummyRegressor(strategy="mean")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    # Modele principal: regression lineaire regularisee.
    model = Ridge(alpha=1.0)

    # Pipeline complet pour entrainement et inference.
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)

    # Sauvegarde d'une vue texte du pipeline.
    save_pipeline_visuals(pipeline, txt_path="reports/regression_supervised.txt")

    metrics = {
        "baseline_mae": float(mean_absolute_error(y_test, baseline_pred)),
        "baseline_rmse": float(np.sqrt(mean_squared_error(y_test, baseline_pred))),
        "baseline_r2": float(r2_score(y_test, baseline_pred)),
        "model_mae": float(mean_absolute_error(y_test, preds)),
        "model_rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "model_r2": float(r2_score(y_test, preds)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
    }

    metrics_path = Path(METRICS_OUTPUT_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    # Bundle pour reutilisation par l'API.
    bundle = {
        "model": pipeline,
        "feature_columns": spec.feature_columns,
        "numeric_features": spec.numeric_features,
        "categorical_features": spec.categorical_features,
        "target_col": TARGET_COL_DEFAULT,
        "date_col": DATE_COL_DEFAULT,
    }

    # Sauvegarde et verification de reload.
    Path(MODEL_OUTPUT_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_OUTPUT_PATH)
    reloaded = joblib.load(MODEL_OUTPUT_PATH)
    reloaded_model = reloaded.get("model")
    if reloaded_model is None:
        raise ValueError("Reload failed: 'model' not found in saved bundle.")
    _ = reloaded_model.predict(X_test.head(5))
    print("=== Supervised Training Results ===")
    print(f"Rows train: {metrics['rows_train']}")
    print(f"Rows test : {metrics['rows_test']}")
    print("")
    print("Baseline (DummyRegressor):")
    print(f"  MAE : {metrics['baseline_mae']:.4f}")
    print(f"  RMSE: {metrics['baseline_rmse']:.4f}")
    print(f"  R2  : {metrics['baseline_r2']:.4f}")
    print("")
    print("Model (Ridge Regression):")
    print(f"  MAE : {metrics['model_mae']:.4f}")
    print(f"  RMSE: {metrics['model_rmse']:.4f}")
    print(f"  R2  : {metrics['model_r2']:.4f}")
    print("")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved model bundle to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
