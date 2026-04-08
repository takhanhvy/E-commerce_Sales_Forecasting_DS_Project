from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import joblib
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from data_utils import load_dataset
from preprocess import DATE_COL_DEFAULT, TARGET_COL_DEFAULT, build_preprocessor, split_features_target


MODEL_OUTPUT_PATH = "artifacts/cluster_model.pkl"
METRICS_OUTPUT_PATH = "reports/cluster_metrics.json"
RANDOM_STATE = 42
MAX_SAMPLE = 5000


def save_pipeline_visuals(pipeline, txt_path: str) -> None:
    path_txt = Path(txt_path)
    path_txt.parent.mkdir(parents=True, exist_ok=True)
    path_txt.write_text(str(pipeline))


def choose_k(X_transformed, random_state: int) -> tuple[int, float]:
    best_k = 3
    best_score = -1.0
    for k in range(2, 7):
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=random_state, n_init=10, batch_size=1024)
        labels = kmeans.fit_predict(X_transformed)
        score = silhouette_score(X_transformed, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, best_score


def main() -> None:
    df = load_dataset()

    if TARGET_COL_DEFAULT in df.columns:
        df = df.drop(columns=[TARGET_COL_DEFAULT])

    X, _, spec = split_features_target(df, target_col=TARGET_COL_DEFAULT)
    preprocessor = build_preprocessor(spec)

    if len(X) > MAX_SAMPLE:
        X_sample = X.sample(MAX_SAMPLE, random_state=RANDOM_STATE)
    else:
        X_sample = X.copy()

    X_transformed = preprocessor.fit_transform(X_sample)
    best_k, best_score = choose_k(X_transformed, RANDOM_STATE)

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("cluster", MiniBatchKMeans(n_clusters=best_k, random_state=RANDOM_STATE, n_init=10, batch_size=1024)),
        ]
    )
    pipeline.fit(X)
    sample_labels = pipeline.predict(X_sample)
    cluster_counts = Counter(sample_labels)

    save_pipeline_visuals(pipeline, txt_path="reports/pipeline_cluster.txt")

    output_path = Path(MODEL_OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": pipeline,
            "feature_columns": spec.feature_columns,
            "best_k": best_k,
            "date_col": DATE_COL_DEFAULT,
        },
        output_path,
    )
    reloaded = joblib.load(output_path)
    reloaded_model = reloaded.get("model")
    if reloaded_model is None:
        raise ValueError("Reload failed: 'model' not found in saved bundle.")
    _ = reloaded_model.predict(X_sample.head(5))
    metrics = {
        "rows_used_sample": int(len(X_sample)),
        "best_k": int(best_k),
        "silhouette_score": float(best_score),
        "cluster_distribution": {int(k): int(v) for k, v in cluster_counts.items()},
    }

    metrics_path = Path(METRICS_OUTPUT_PATH)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2))

    print("=== Unsupervised Clustering Results ===")
    print(f"Rows used (sample): {len(X_sample)}")
    print(f"Best k: {best_k}")
    print(f"Silhouette score (sample): {best_score:.4f}")
    print("Cluster distribution (sample):")
    for key in sorted(cluster_counts):
        print(f"  Cluster {key}: {cluster_counts[key]}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Saved cluster model to {output_path}")


if __name__ == "__main__":
    main()
