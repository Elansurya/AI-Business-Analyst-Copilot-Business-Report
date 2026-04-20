from __future__ import annotations
import re
from typing import Optional

import numpy as np
import pandas as pd

try:
    import polars as pl
    POLARS = True
except ImportError:
    POLARS = False


# ── helpers 
def _to_snake(name: str) -> str:
    return re.sub(r"\W+", "_", name.strip().lower()).strip("_")


def _detect_time_columns(df: pd.DataFrame) -> list[str]:
    """Find date/time columns heuristically."""
    found = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col)
            continue
        col_lower = col.lower()
        if any(k in col_lower for k in ["date","month","year","time","period","week","quarter"]):
            found.append(col)
    return found


def _safe_cast_dates(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    for col in date_cols:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            try:
                df[col] = pd.to_datetime(df[col], infer_format=True, errors="coerce")
            except Exception:
                pass
    return df


# ── core cleaning 
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Memory-efficient cleaning:
    • Drop >80% missing columns
    • Drop duplicate rows
    • Downcast numerics (int64→int32, float64→float32)
    • Strip whitespace from object columns
    """
    # Drop extremely sparse columns
    thresh = len(df) * 0.20        # must have ≥20% non-null
    df = df.dropna(thresh=thresh, axis=1)

    # Drop duplicates
    df = df.drop_duplicates()

    # Strip whitespace
    obj_cols = df.select_dtypes(include="object").columns
    for col in obj_cols:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # Downcast numerics to save memory
    for col in df.select_dtypes(include=["int64","int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df.reset_index(drop=True)


# ── aggregation pipeline 
def aggregate_dataset(
    df: pd.DataFrame,
    target_col: Optional[str] = None,
    max_output_rows: int = 2000,
) -> tuple[pd.DataFrame, dict]:
    """
    Core aggregation: raw (up to 500k rows) → summarized (≤2k rows).

    Strategy:
    1. Identify categorical group-by columns (≤30 unique values each).
    2. Identify numerical value columns.
    3. Detect time columns → extract year/month/quarter.
    4. Perform GroupBy with sum, mean, count, std.
    5. If result > max_output_rows, drop lowest-cardinality group key.

    Returns (aggregated_df, aggregation_metadata).
    """
    meta = {
        "original_rows":  len(df),
        "original_cols":  len(df.columns),
        "group_keys":     [],
        "value_cols":     [],
        "time_col":       None,
        "agg_rows":       0,
        "reduction_ratio": 0.0,
    }

    df = clean_dataframe(df)

    # ── 1. Detect time col 
    time_cols = _detect_time_columns(df)
    time_col  = None
    if time_cols:
        time_col = time_cols[0]
        df = _safe_cast_dates(df, [time_col])
        if pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df["_year"]    = df[time_col].dt.year
            df["_month"]   = df[time_col].dt.month
            df["_quarter"] = df[time_col].dt.quarter
            meta["time_col"] = time_col

    # ── 2. Identify group-by candidates 
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # exclude target, time-derived cols, id-like cols
    id_pattern = re.compile(r"\bid\b|\bindex\b|\bkey\b|\bcode\b|\bserial\b", re.I)

    cat_candidates = []
    for col in df.select_dtypes(include=["object","category"]).columns:
        n_unique = df[col].nunique()
        if n_unique < 2 or id_pattern.search(col):
            continue
        if n_unique <= 40:
            cat_candidates.append((col, n_unique))

    # Also check low-cardinality numerics (e.g. month, region-encoded)
    for col in num_cols:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 15 and col not in [target_col]:
            col_lower = col.lower()
            if any(k in col_lower for k in ["region","zone","area","segment","type","class","group","tier","flag"]):
                cat_candidates.append((col, n_unique))

    # Add time-derived cols
    time_derived = [c for c in ["_year","_quarter","_month"] if c in df.columns]
    group_keys_ordered = time_derived + [c for c, _ in sorted(cat_candidates, key=lambda x: x[1])]

    # Remove duplicates preserving order
    seen = set()
    group_keys = []
    for k in group_keys_ordered:
        if k not in seen:
            group_keys.append(k)
            seen.add(k)

    # ── 3. Value columns 
    exclude = set(group_keys) | {time_col} | {c for c, _ in cat_candidates}
    if target_col:
        # keep target in value cols for aggregation
        pass
    value_cols = [c for c in num_cols if c not in exclude and c not in ("_year","_quarter","_month")]

    if not value_cols:
        # If nothing to aggregate, return describe-level summary
        desc = df.describe().reset_index()
        meta["agg_rows"] = len(desc)
        meta["reduction_ratio"] = round(meta["agg_rows"] / meta["original_rows"], 6)
        return desc, meta

    # ── 4. Progressive groupby to stay under max_output_rows 
    agg_spec = {col: ["sum","mean","count","std"] for col in value_cols}

    def _run_groupby(keys: list[str]) -> pd.DataFrame:
        if not keys:
            agg_row = {}
            for col in value_cols:
                agg_row[f"{col}_sum"]   = df[col].sum()
                agg_row[f"{col}_mean"]  = df[col].mean()
                agg_row[f"{col}_count"] = df[col].count()
                agg_row[f"{col}_std"]   = df[col].std()
            return pd.DataFrame([agg_row])

        grouped = df.groupby(keys, observed=True)[value_cols].agg(["sum","mean","count","std"])
        grouped.columns = [f"{c}_{a}" for c, a in grouped.columns]
        return grouped.reset_index()

    # Try progressively fewer group keys until result ≤ max_output_rows
    used_keys = group_keys.copy()
    while True:
        try:
            result = _run_groupby(used_keys)
            if len(result) <= max_output_rows or len(used_keys) == 0:
                break
            used_keys = used_keys[:-1]          # drop the least-important key
        except Exception:
            used_keys = used_keys[:-1]
            if not used_keys:
                result = _run_groupby([])
                break

    meta["group_keys"]     = used_keys
    meta["value_cols"]     = value_cols
    meta["agg_rows"]       = len(result)
    meta["reduction_ratio"] = round(meta["agg_rows"] / meta["original_rows"], 6)

    return result, meta


# ── KPI extraction 
def extract_kpis(df: pd.DataFrame, agg_df: pd.DataFrame,
                 meta: dict, target_col: Optional[str] = None) -> dict:
    """
    Extract top-level KPI numbers for the Overview dashboard.
    Works from aggregated data only.
    """
    kpis = {
        "total_rows":      meta["original_rows"],
        "total_cols":      meta["original_cols"],
        "agg_rows":        meta["agg_rows"],
        "reduction_pct":   round((1 - meta["reduction_ratio"]) * 100, 1),
        "group_keys":      meta["group_keys"],
        "value_cols":      meta["value_cols"],
    }

    value_cols = meta.get("value_cols", [])
    if not value_cols:
        return kpis

    # Primary metric — prefer target col or first value col
    primary = target_col if (target_col and f"{target_col}_sum" in agg_df.columns) else value_cols[0]
    sum_col  = f"{primary}_sum"
    mean_col = f"{primary}_mean"

    if sum_col in agg_df.columns:
        kpis["primary_col"]   = primary
        kpis["primary_total"] = float(agg_df[sum_col].sum())
        kpis["primary_mean"]  = float(agg_df[mean_col].mean()) if mean_col in agg_df.columns else None

        # YoY or period-on-period growth if time data exists
        if "_year" in agg_df.columns and sum_col in agg_df.columns:
            yearly = agg_df.groupby("_year")[sum_col].sum()
            if len(yearly) >= 2:
                years  = sorted(yearly.index)
                latest = float(yearly[years[-1]])
                prev   = float(yearly[years[-2]])
                growth = ((latest - prev) / prev * 100) if prev != 0 else 0
                kpis["yoy_growth_pct"]   = round(growth, 2)
                kpis["latest_year"]      = int(years[-1])
                kpis["previous_year"]    = int(years[-2])
                kpis["latest_year_val"]  = round(latest, 2)
                kpis["prev_year_val"]    = round(prev, 2)

        # Top group by first categorical key
        cat_keys = [k for k in meta.get("group_keys", []) if not k.startswith("_")]
        if cat_keys and sum_col in agg_df.columns:
            top_key = cat_keys[0]
            top_group = (agg_df.groupby(top_key)[sum_col].sum()
                         .sort_values(ascending=False))
            kpis["top_category_col"]    = top_key
            kpis["top_category_name"]   = str(top_group.index[0])
            kpis["top_category_value"]  = round(float(top_group.iloc[0]), 2)
            kpis["top_category_share"]  = round(
                float(top_group.iloc[0]) / float(top_group.sum()) * 100, 1)

    return kpis


# ── text summary for RAG context 
def build_processing_context(df: pd.DataFrame, agg_df: pd.DataFrame,
                              meta: dict, kpis: dict) -> str:
    """Build a text description of the aggregated dataset for the RAG chatbot."""
    lines = [
        f"Original dataset: {meta['original_rows']:,} rows × {meta['original_cols']} columns.",
        f"After aggregation: {meta['agg_rows']:,} summary rows (reduction: {kpis['reduction_pct']:.1f}%).",
        f"Aggregation group keys: {', '.join(meta['group_keys']) if meta['group_keys'] else 'none (global stats)'}.",
        f"Value columns aggregated: {', '.join(meta['value_cols'])}.",
    ]

    if "primary_col" in kpis:
        lines.append(f"Primary metric '{kpis['primary_col']}': total = {kpis['primary_total']:,.2f}, "
                     f"mean = {kpis.get('primary_mean', 'N/A')}.")

    if "yoy_growth_pct" in kpis:
        direction = "increased" if kpis["yoy_growth_pct"] > 0 else "decreased"
        lines.append(
            f"Year-over-year: '{kpis['primary_col']}' {direction} by "
            f"{abs(kpis['yoy_growth_pct']):.1f}% from {kpis['previous_year']} "
            f"({kpis['prev_year_val']:,.0f}) to {kpis['latest_year']} ({kpis['latest_year_val']:,.0f})."
        )

    if "top_category_name" in kpis:
        lines.append(
            f"Top '{kpis['top_category_col']}': '{kpis['top_category_name']}' "
            f"contributes {kpis['top_category_share']:.1f}% of total {kpis['primary_col']}."
        )

    # Per-group stats from agg_df
    for key in meta.get("group_keys", [])[:3]:
        if key.startswith("_") or key not in agg_df.columns:
            continue
        for vc in meta.get("value_cols", [])[:2]:
            sum_c = f"{vc}_sum"
            if sum_c not in agg_df.columns:
                continue
            top = (agg_df.groupby(key)[sum_c].sum()
                   .sort_values(ascending=False).head(5))
            total = top.sum()
            parts = []
            for cat, val in top.items():
                pct = val / total * 100 if total > 0 else 0
                parts.append(f"'{cat}'={val:,.1f} ({pct:.1f}%)")
            lines.append(f"{key} breakdown (by {vc} sum): {'; '.join(parts)}.")

    return "\n".join(lines)