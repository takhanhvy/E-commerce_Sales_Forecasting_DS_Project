from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATE_COL_DEFAULT = "Date"
TARGET_COL_DEFAULT = "Amount"


@dataclass
class FeatureSpec:
    feature_columns: List[str]
    numeric_features: List[str]
    categorical_features: List[str]
    date_features: List[str]


def add_date_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        return df
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df["date_year"] = df[date_col].dt.year
    df["date_month"] = df[date_col].dt.month
    df["date_day"] = df[date_col].dt.day
    df["date_dayofweek"] = df[date_col].dt.dayofweek
    df = df.drop(columns=[date_col])
    return df


def split_features_target(
    df: pd.DataFrame,
    target_col: str = TARGET_COL_DEFAULT,
    date_col: str = DATE_COL_DEFAULT,
) -> Tuple[pd.DataFrame, Optional[pd.Series], FeatureSpec]:
    df = add_date_features(df, date_col)
    if target_col in df.columns:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    date_features = [
        c
        for c in ["date_year", "date_month", "date_day", "date_dayofweek"]
        if c in X.columns
    ]
    spec = FeatureSpec(
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        date_features=date_features,
    )
    return X, y, spec


def _build_onehot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True,
            min_frequency=0.01,
        )
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse=True, min_frequency=0.01)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(spec: FeatureSpec) -> ColumnTransformer:
    transformers = []

    if spec.numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler(with_mean=False)),
            ]
        )
        transformers.append(("num", numeric_pipeline, spec.numeric_features))

    if spec.categorical_features:
        encoder = _build_onehot_encoder()
        categorical_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", encoder),
            ]
        )
        transformers.append(("cat", categorical_pipeline, spec.categorical_features))

    if not transformers:
        raise ValueError("No usable features found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=0.0)
