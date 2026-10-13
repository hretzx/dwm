"""
utils/preprocessing.py
----------------------
Shared data helpers: dataset loading (with a synthetic fallback),
feature names and the default number of clusters.
Later steps will add binning, scaling and validation here.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Where the real Kaggle CSV should be placed.
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "Crop_recommendation.csv"

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
TARGET = "label"
DEFAULT_K = 4  # default number of K-Means clusters

# Approximate average profile of each crop: N, P, K, temp, humidity, ph, rainfall.
# Used ONLY to build a synthetic dataset when the real CSV is missing.
_CROP_PROFILES = {
    "rice": (80, 48, 40, 23.7, 82, 6.4, 236),
    "maize": (78, 48, 20, 22.4, 65, 6.25, 85),
    "chickpea": (40, 68, 80, 18.9, 17, 7.3, 80),
    "kidneybeans": (21, 68, 20, 20.0, 22, 5.75, 106),
    "pigeonpeas": (21, 68, 20, 27.7, 48, 5.8, 149),
    "mothbeans": (21, 48, 20, 28.2, 53, 6.8, 51),
    "mungbean": (21, 48, 20, 28.5, 85, 6.7, 48),
    "blackgram": (40, 67, 19, 29.0, 65, 7.1, 68),
    "lentil": (19, 68, 19, 24.5, 65, 6.9, 46),
    "pomegranate": (18, 19, 40, 21.8, 90, 6.4, 108),
    "banana": (100, 82, 50, 27.4, 80, 5.98, 104),
    "mango": (20, 27, 30, 31.2, 50, 5.76, 95),
    "grapes": (23, 132, 200, 23.8, 82, 6.0, 69),
    "watermelon": (99, 17, 50, 25.6, 85, 6.5, 50),
    "muskmelon": (100, 17, 50, 28.7, 92, 6.36, 25),
    "apple": (21, 134, 200, 22.6, 92, 5.93, 113),
    "orange": (20, 10, 10, 22.8, 92, 7.0, 110),
    "papaya": (50, 59, 50, 33.7, 92, 6.74, 143),
    "coconut": (22, 17, 31, 27.4, 95, 5.98, 176),
    "cotton": (118, 46, 20, 24.0, 80, 6.9, 80),
    "jute": (78, 46, 40, 25.0, 80, 6.7, 175),
    "coffee": (101, 29, 30, 25.5, 59, 6.8, 158),
}


def generate_synthetic_dataset(per_crop: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create 22 crops x per_crop rows by adding random noise around each crop profile."""
    rng = np.random.default_rng(seed)
    rows = []
    for crop, (n, p, k, t, h, ph, r) in _CROP_PROFILES.items():
        block = pd.DataFrame({
            "N": rng.normal(n, max(3, 0.08 * n), per_crop).clip(0, 140).round(),
            "P": rng.normal(p, max(3, 0.08 * p), per_crop).clip(5, 145).round(),
            "K": rng.normal(k, max(3, 0.08 * k), per_crop).clip(5, 205).round(),
            "temperature": rng.normal(t, 1.5, per_crop).clip(8, 44),
            "humidity": rng.normal(h, 3, per_crop).clip(14, 100),
            "ph": rng.normal(ph, 0.4, per_crop).clip(3.5, 9.9),
            "rainfall": rng.normal(r, 0.12 * r + 5, per_crop).clip(20, 300),
            "label": crop,
        })
        rows.append(block)
    df = pd.concat(rows, ignore_index=True)
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def load_dataset() -> tuple[pd.DataFrame, bool]:
    """
    Return (dataframe, is_synthetic).
    Uses the real CSV if present; otherwise generates and saves a synthetic one
    so the app still runs out of the box.
    """
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
        missing = set(FEATURES + [TARGET]) - set(df.columns)
        if missing:
            raise ValueError(f"CSV is missing columns: {sorted(missing)}")
        return df.dropna().reset_index(drop=True), False

    df = generate_synthetic_dataset()
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    return df, True


# ---------------------------------------------------------------- Validation
# Allowed input range per feature: (min, max, unit). These cover the values
# seen in the Crop Recommendation dataset, with a little headroom.
VALID_RANGES = {
    "N": (0, 140, "kg/ha"),
    "P": (5, 145, "kg/ha"),
    "K": (5, 205, "kg/ha"),
    "ph": (3.5, 10.0, ""),
    "temperature": (8.0, 44.0, "°C"),
    "humidity": (14.0, 100.0, "%"),
    "rainfall": (20.0, 300.0, "mm"),
}


def validate_inputs(values: dict) -> list[str]:
    """Return a list of readable error messages (empty list = all inputs valid)."""
    errors = []
    for name, (lo, hi, unit) in VALID_RANGES.items():
        v = values.get(name)
        if v is None or not np.isfinite(v):
            errors.append(f"{name}: a number is required.")
        elif not lo <= v <= hi:
            errors.append(f"{name}: {v} is outside the allowed range {lo} to {hi} {unit}.")
    return errors


# ---------------------------------------------------------------- Categories
def category_cutoffs(df: pd.DataFrame) -> dict:
    """
    Low/medium/high cut-offs for each feature = 33rd and 67th percentile of the
    dataset. So 'high N' means 'higher than about two thirds of the records'.
    """
    return {
        f: (float(df[f].quantile(1 / 3)), float(df[f].quantile(2 / 3)))
        for f in FEATURES
    }


def categorize(value: float, cutoff: tuple[float, float]) -> str:
    """Return 'low', 'medium' or 'high' for one value given its (low, high) cut-offs."""
    low_cut, high_cut = cutoff
    if value < low_cut:
        return "low"
    if value < high_cut:
        return "medium"
    return "high"
