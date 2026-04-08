from __future__ import annotations

"""Utilitaires pour charger le dataset local."""

from pathlib import Path
from typing import Optional

import pandas as pd

DEFAULT_DATA_PATH = Path("data/raw/Amazon Sale Report.csv")


def load_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Charge le CSV local depuis data/raw/."""
    data_path = Path(path) if path else DEFAULT_DATA_PATH
    if data_path.exists():
        return pd.read_csv(data_path)
    raise FileNotFoundError(
        f"Dataset not found at {data_path}. "
        "Place the CSV file at this path."
    )
