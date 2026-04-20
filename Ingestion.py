from __future__ import annotations
import io
import re
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS_AVAILABLE = True
except ImportError:
    POLARS_AVAILABLE = False

# ── constants 
DATASET_TYPE_KEYWORDS = {
    "retail/sales":      ["sales","revenue","discount","product","category","store","region","units","price","order"],
    "finance":           ["profit","loss","income","expense","balance","asset","liability","cash","budget","roi","stock"],
    "healthcare":        ["patient","diagnosis","treatment","hospital","doctor","medication","age","bmi","blood","disease"],
    "hr/workforce":      ["employee","salary","tenure","department","attrition","churn","hire","performance","manager"],
    "marketing":         ["campaign","click","impression","conversion","lead","ctr","cpa","channel","spend","audience"],
    "logistics":         ["shipment","delivery","route","warehouse","inventory","freight","carrier","delay","transit"],
    "ecommerce":         ["cart","session","visit","bounce","page","product","checkout","basket","refund","return"],
}

TARGET_KEYWORDS = [
    "target","label","churn","fraud","default","sales","revenue","price","salary","outcome",
    "result","class","category","status","score","rating","y","output","predict"
]


# ── loaders 
def load_file(uploaded_file) -> pd.DataFrame:
    """
    Load CSV or Excel into a Pandas DataFrame.
    Uses Polars for CSV (much faster on large files) then converts.
    """
    name = uploaded_file.name.lower()
    raw = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        if POLARS_AVAILABLE:
            try:
                buf = io.BytesIO(raw)
                lf = pl.read_csv(
                    buf,
                    infer_schema_length=5000,
                    null_values=["", "NA", "N/A", "null", "NULL", "None"],
                    ignore_errors=True
                )
                return lf.to_pandas()
            except Exception:
                pass

        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(
                    io.BytesIO(raw),
                    encoding=enc,
                    low_memory=False,
                    na_values=["", "NA", "N/A", "null"]
                )
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV — try saving as UTF-8.")

    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")

    raise ValueError(f"Unsupported format: {name}. Please upload CSV or Excel.")


def load_large_csv_chunked(path: str, chunksize: int = 100_000) -> pd.DataFrame:
    """Load very large CSVs in chunks (fallback path for memory-constrained envs)."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        chunks.append(chunk)
    return pd.concat(chunks, ignore_index=True)


# ── type detection 
def classify_columns(df: pd.DataFrame) -> dict:
    """Classify columns into numerical, categorical, datetime, id-like, text."""
    result = {
        "numerical":   [],
        "categorical": [],
        "datetime":    [],
        "id_like":     [],
        "text":        []
    }

    dt_pattern = re.compile(
        r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}"
    )

    for col in df.columns:
        series  = df[col]
        n_unique = series.nunique()
        n_total  = len(series.dropna())

        if n_total == 0:
            continue

        # Already datetime dtype
        if pd.api.types.is_datetime64_any_dtype(series):
            result["datetime"].append(col)
            continue

        # Try coercing object columns to datetime
        if series.dtype == object:
            sample = series.dropna().head(50).astype(str)
            if sample.str.match(dt_pattern).mean() > 0.7:
                result["datetime"].append(col)
                continue

        # Numerical
        if pd.api.types.is_numeric_dtype(series):
            unique_ratio = n_unique / n_total
            if unique_ratio < 0.02 and n_unique <= 20:
                result["categorical"].append(col)
            else:
                result["numerical"].append(col)
            continue

        # ID-like: high cardinality strings
        unique_ratio = n_unique / n_total
        if unique_ratio > 0.7 and n_unique > 50:
            result["id_like"].append(col)
            continue

        # Long free text
        avg_len = series.dropna().astype(str).str.len().mean()
        if avg_len > 60:
            result["text"].append(col)
            continue

        result["categorical"].append(col)

    return result


# ── dataset type inference 
def infer_dataset_type(df: pd.DataFrame) -> tuple:
    """
    Returns (domain, summary_sentence).
    Scores column names against domain keyword lists.
    """
    all_cols = " ".join(df.columns.str.lower().tolist())
    scores = {domain: 0 for domain in DATASET_TYPE_KEYWORDS}

    for domain, keywords in DATASET_TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in all_cols:
                scores[domain] += 1

    best_domain = max(scores, key=scores.get)
    best_score  = scores[best_domain]

    col_types = classify_columns(df)
    n_num  = len(col_types["numerical"])
    n_cat  = len(col_types["categorical"])
    n_rows = len(df)

    if best_score == 0:
        best_domain = "general analytics"

    summaries = {
        "retail/sales":      "retail or sales transactions across regions and product categories",
        "finance":           "financial records covering income, expenses, and profitability",
        "healthcare":        "healthcare or medical data with patient and clinical attributes",
        "hr/workforce":      "human resources data covering employees, tenure, and performance",
        "marketing":         "marketing or campaign performance data with channel metrics",
        "logistics":         "logistics or supply-chain data covering deliveries and routes",
        "ecommerce":         "e-commerce data with user sessions, products, and conversions",
        "general analytics": "structured tabular data suitable for business analytics",
    }

    summary = (
        f"This dataset represents {summaries.get(best_domain, 'business data')}. "
        f"It contains {n_rows:,} records with {n_num} numerical and "
        f"{n_cat} categorical features."
    )
    return best_domain, summary


# ── target detection 
def suggest_targets(df: pd.DataFrame) -> list:
    """Return columns sorted by likelihood of being the target variable."""
    scored = []
    for col in df.columns:
        col_lower = col.lower().replace("_", " ").replace("-", " ")
        score = sum(1 for kw in TARGET_KEYWORDS if kw in col_lower)
        scored.append((col, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return [c for c, _ in scored]


# ── quick column summary 
def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Return a summary DataFrame with role, dtype, missing, unique, and sample."""
    rows = []
    col_types = classify_columns(df)
    type_map = {}
    for t, cols in col_types.items():
        for c in cols:
            type_map[c] = t.replace("_", " ").title()

    for col in df.columns:
        s      = df[col]
        n_miss = int(s.isnull().sum())
        rows.append({
            "Column":  col,
            "Role":    type_map.get(col, "Unknown"),
            "Dtype":   str(s.dtype),
            "Missing": f"{n_miss:,} ({n_miss / len(df) * 100:.1f}%)",
            "Unique":  f"{s.nunique():,}",
            "Sample":  ", ".join(str(v) for v in s.dropna().head(3).tolist()),
        })
    return pd.DataFrame(rows)


# ── schema fingerprint for caching 
def schema_fingerprint(df: pd.DataFrame) -> str:
    """Stable hash of shape + column names — used for cache invalidation."""
    sig         = f"{df.shape}|{'|'.join(df.columns.tolist())}"
    fingerprint = hashlib.md5(sig.encode()).hexdigest()[:8]
    print(fingerprint)
    return fingerprint