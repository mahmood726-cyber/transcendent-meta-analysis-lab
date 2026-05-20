import os
import warnings

import numpy as np
import pandas as pd

from core.types import MAData


def _get_data_dir():
    return os.environ.get(
        "TMAL_DATA_DIR",
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "data"),
    )


def load_integrated_df(domain=None, source=None, design=None, require_se=True):
    """Load all_real_data_integrated.csv as a cleaned DataFrame.

    Coerces ``log_or`` (and ``se`` when ``require_se``) to numeric, drops rows
    with NA in those columns, keeps only ``se > 0`` when required, then applies
    the optional ``domain``/``source``/``design`` filters. Also coerces ``year``
    to numeric when the column exists, leaving NaNs for unparseable values.
    """
    path = os.path.join(_get_data_dir(), "all_real_data_integrated.csv")
    df = pd.read_csv(path)

    numeric_cols = ["log_or"] + (["se"] if require_se else [])
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=numeric_cols)
    if require_se:
        df = df[df["se"] > 0]
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if domain is not None:
        df = df[df["domain"] == domain]
    if source is not None:
        df = df[df["source"] == source]
    if design is not None:
        df = df[df["design"] == design]
    if len(df) == 0:
        raise ValueError(
            f"0 studies after filtering (domain={domain}, source={source}, design={design})"
        )
    return df.reset_index(drop=True)


def load_integrated(domain=None, source=None, design=None):
    """Load all_real_data_integrated.csv and return MAData."""
    df = load_integrated_df(domain=domain, source=source, design=design)
    if len(df) < 3:
        warnings.warn(
            f"Only {len(df)} studies after filtering — results may be unreliable"
        )
    year = df["year"].values if "year" in df.columns else None
    design_arr = df["design"].values if "design" in df.columns else None
    group = (df["source"].astype(str) + ":" + df["domain"].astype(str)).values
    return MAData(
        effect=df["log_or"].values.astype(float),
        se=df["se"].values.astype(float),
        group=group,
        year=year,
        design=design_arr,
    )


def load_pairwise70(sample_n=None, seed=42):
    """Load ma4_results_pairwise70.csv and return MAData."""
    data_dir = os.environ.get("TMAL_R_DATA_DIR", _get_data_dir())
    path = os.path.join(data_dir, "ma4_results_pairwise70.csv")
    df = pd.read_csv(path)
    for col in ["theta", "sigma"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["theta", "sigma"])
    df = df[df["sigma"] > 0]
    if sample_n is not None and sample_n < len(df["review_id"].unique()):
        rng = np.random.RandomState(seed)
        reviews = rng.choice(df["review_id"].unique(), sample_n, replace=False)
        df = df[df["review_id"].isin(reviews)]
    if len(df) == 0:
        raise ValueError("0 studies after filtering")
    return MAData(
        effect=df["theta"].values.astype(float),
        se=df["sigma"].values.astype(float),
        group=df["review_id"].values,
    )
