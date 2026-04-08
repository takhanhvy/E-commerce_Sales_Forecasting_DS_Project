from __future__ import annotations

"""API Flask: regression + clustering."""

import os
import pickle
import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    import joblib
except ImportError:  # pragma: no cover
    joblib = None

import pandas as pd
from flask import Flask, jsonify, request

# Racine du projet (pour importer src/ et trouver les modeles).
APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
# Chemins des modeles (modifiables via variables d'environnement).
SUPERVISED_MODEL_PATH = os.environ.get(
    "MODEL_PATH", str(APP_ROOT / "artifacts" / "regression_model.pkl")
)
UNSUPERVISED_MODEL_PATH = os.environ.get(
    "CLUSTER_MODEL_PATH", str(APP_ROOT / "artifacts" / "cluster_model.pkl")
)

app = Flask(__name__)


def load_bundle(path: str):
    """Charge un pickle via joblib si dispo, sinon pickle standard."""
    if joblib is not None:
        return joblib.load(path)
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_bundle(path: str):
    """Retourne (modele, bundle) pour simplifier la logique."""
    bundle = load_bundle(path)
    return bundle.get("model"), bundle


try:
    # Chargement du modele supervise.
    supervised_model, supervised_bundle = load_model_bundle(SUPERVISED_MODEL_PATH)
    supervised_feature_columns = supervised_bundle.get("feature_columns")
except Exception as exc:  # pylint: disable=broad-except
    supervised_model = None
    supervised_feature_columns = None
    supervised_error = str(exc)
else:
    supervised_error = None

try:
    # Chargement du modele de clustering.
    cluster_model, cluster_bundle = load_model_bundle(UNSUPERVISED_MODEL_PATH)
    cluster_feature_columns = cluster_bundle.get("feature_columns")
except Exception as exc:  # pylint: disable=broad-except
    cluster_model = None
    cluster_feature_columns = None
    cluster_error = str(exc)
else:
    cluster_error = None


def normalize_payload(payload: Any) -> List[Dict[str, Any]]:
    """Normalise le payload en liste de dicts."""
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        if not payload:
            raise ValueError("Payload list is empty")
        if not all(isinstance(item, dict) for item in payload):
            raise ValueError("Each item in the payload list must be an object")
        return payload
    raise ValueError("Payload must be an object or a list of objects")


def build_dataframe(
    records: List[Dict[str, Any]], feature_columns: List[str] | None
) -> pd.DataFrame:
    """Construit un DataFrame en alignant les colonnes du modele."""
    df = pd.DataFrame(records)
    if feature_columns:
        for col in feature_columns:
            if col not in df.columns:
                df[col] = None
        df = df[feature_columns]
    return df


@app.route("/health", methods=["GET"])
def health() -> Any:
    """Etat des modeles charges."""
    status = {
        "status": "ok"
        if supervised_model is not None and cluster_model is not None
        else "error",
        "regression_loaded": supervised_model is not None,
        "regression_error": supervised_error,
        "cluster_loaded": cluster_model is not None,
        "cluster_error": cluster_error,
    }
    return jsonify(status)


@app.route("/predict_regression", methods=["POST"])
def predict_regression() -> Any:
    """Endpoint regression: retourne une liste de predictions."""
    if supervised_model is None:
        return (
            jsonify(
                {"error": "Supervised model not loaded", "details": supervised_error}
            ),
            500,
        )

    try:
        # Le payload peut etre un dict ou une liste de dicts.
        payload = request.get_json(force=True)
        records = normalize_payload(payload)
        # Aligne les colonnes avec celles du modele.
        df = build_dataframe(records, supervised_feature_columns)
        preds = supervised_model.predict(df)
        return jsonify({"predictions": [float(p) for p in preds]})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_cluster", methods=["POST"])
def predict_cluster() -> Any:
    """Endpoint clustering: retourne les labels de clusters."""
    if cluster_model is None:
        return (
            jsonify({"error": "Cluster model not loaded", "details": cluster_error}),
            500,
        )

    try:
        # Le payload peut etre un dict ou une liste de dicts.
        payload = request.get_json(force=True)
        records = normalize_payload(payload)
        # Aligne les colonnes avec celles du modele.
        df = build_dataframe(records, cluster_feature_columns)
        labels = cluster_model.predict(df)
        return jsonify({"clusters": [int(l) for l in labels]})
    except Exception as exc:  # pylint: disable=broad-except
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
