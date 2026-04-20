
# CORE IMPORTS

from __future__ import annotations

import io
import os
import re
import sys
import warnings
import hashlib
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# Optional fast CSV loader
try:
    import polars as pl
    POLARS_OK = True
except ImportError:
    POLARS_OK = False

# Optional ML explainability
try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# Optional semantic search
try:
    from sentence_transformers import SentenceTransformer
    ST_OK = True
except ImportError:
    ST_OK = False

try:
    import faiss
    FAISS_OK = True
except ImportError:
    FAISS_OK = False

# Sklearn
from sklearn.model_selection import (train_test_split, cross_val_score,
                                      StratifiedKFold, KFold)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (RandomForestClassifier, RandomForestRegressor,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score

# Plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st

st.set_page_config(
    page_title="AI Business Analyst Copilot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# THEME CONSTANTS

BG      = "#0A0C14"
CARD    = "#13172A"
CARD2   = "#1C2240"
BORDER  = "#1E2540"
TEXT    = "#E2E8F0"
MUTED   = "#6B7494"
ACCENT  = "#6C63FF"
GREEN   = "#10B981"
RED     = "#EF4444"
YELLOW  = "#F59E0B"
CYAN    = "#06B6D4"
PURPLE  = "#8B5CF6"
PALETTE = [ACCENT, GREEN, YELLOW, RED, CYAN, PURPLE,
           "#F97316", "#EC4899", "#14B8A6", "#84CC16"]


def _fig(fig, title="", height=420):
    """Apply dark theme to any plotly figure."""
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="sans-serif"),
        height=height,
        margin=dict(t=50, b=40, l=12, r=12),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED), linecolor="#1E2340")
    fig.update_yaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED), linecolor="#1E2340")
    return fig


# SECTION 1 — DATA INGESTION

DOMAIN_KEYWORDS = {
    "retail/sales":  ["sales","revenue","discount","product","category","region","units","price","order","store"],
    "finance":       ["profit","loss","income","expense","balance","asset","cash","budget","roi","stock"],
    "healthcare":    ["patient","diagnosis","treatment","hospital","doctor","medication","bmi","blood","disease"],
    "hr/workforce":  ["employee","salary","tenure","department","attrition","churn","hire","performance"],
    "marketing":     ["campaign","click","impression","conversion","lead","ctr","cpa","channel","spend"],
    "logistics":     ["shipment","delivery","route","warehouse","inventory","freight","carrier","delay"],
    "ecommerce":     ["cart","session","visit","bounce","checkout","basket","refund","return"],
}

TARGET_KEYWORDS = ["target","label","churn","fraud","default","sales","revenue","price",
                   "salary","outcome","result","class","status","score","rating","y","output"]


def load_file(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel. Uses Polars for CSV (3-5x faster on large files)."""
    name = uploaded_file.name.lower()
    raw  = uploaded_file.read()
    uploaded_file.seek(0)

    if name.endswith(".csv"):
        if POLARS_OK:
            try:
                df = pl.read_csv(
                    io.BytesIO(raw),
                    infer_schema_length=5000,
                    null_values=["", "NA", "N/A", "null", "NULL", "None"],
                    ignore_errors=True,
                ).to_pandas()
                return df
            except Exception:
                pass
        for enc in ["utf-8", "latin-1", "cp1252"]:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc,
                                   low_memory=False, na_values=["", "NA", "N/A", "null"])
            except UnicodeDecodeError:
                continue
        raise ValueError("Could not decode CSV — try saving as UTF-8.")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw), engine="openpyxl")

    raise ValueError(f"Unsupported format '{name}'. Upload CSV or Excel.")


def classify_columns(df: pd.DataFrame) -> dict:
    """Classify each column: numerical / categorical / datetime / id_like / text."""
    result = {"numerical": [], "categorical": [], "datetime": [], "id_like": [], "text": []}
    dt_pat = re.compile(r"\d{4}[-/]\d{2}[-/]\d{2}|\d{2}[-/]\d{2}[-/]\d{4}")

    for col in df.columns:
        s        = df[col]
        n_unique = s.nunique()
        n_total  = len(s.dropna())
        if n_total == 0:
            continue

        if pd.api.types.is_datetime64_any_dtype(s):
            result["datetime"].append(col); continue

        if s.dtype == object:
            sample = s.dropna().head(50).astype(str)
            if sample.str.match(dt_pat).mean() > 0.7:
                result["datetime"].append(col); continue

        if pd.api.types.is_numeric_dtype(s):
            if (n_unique / n_total) < 0.02 and n_unique <= 20:
                result["categorical"].append(col)
            else:
                result["numerical"].append(col)
            continue

        ratio = n_unique / n_total
        if ratio > 0.7 and n_unique > 50:
            result["id_like"].append(col); continue
        if s.dropna().astype(str).str.len().mean() > 60:
            result["text"].append(col); continue
        result["categorical"].append(col)

    return result


def infer_domain(df: pd.DataFrame) -> tuple[str, str]:
    """Return (domain_name, one-sentence summary)."""
    cols_str = " ".join(df.columns.str.lower())
    scores   = {d: sum(1 for kw in kws if kw in cols_str)
                for d, kws in DOMAIN_KEYWORDS.items()}
    domain   = max(scores, key=scores.get) if max(scores.values()) > 0 else "general analytics"
    ct       = classify_columns(df)
    summaries = {
        "retail/sales":  "retail or sales transactions across regions and product categories",
        "finance":       "financial records covering income, expenses, and profitability",
        "healthcare":    "healthcare data with patient and clinical attributes",
        "hr/workforce":  "human-resources data covering employees, tenure, and performance",
        "marketing":     "marketing or campaign-performance data with channel metrics",
        "logistics":     "logistics or supply-chain data covering deliveries and routes",
        "ecommerce":     "e-commerce data with sessions, products, and conversions",
        "general analytics": "structured tabular data suitable for business analytics",
    }
    summary = (f"This dataset represents {summaries.get(domain, 'business data')}. "
               f"It contains {len(df):,} records with {len(ct['numerical'])} numerical "
               f"and {len(ct['categorical'])} categorical features.")
    return domain, summary


def suggest_targets(df: pd.DataFrame) -> list[str]:
    """Sort columns by likelihood of being the target variable."""
    scored = [(c, sum(1 for kw in TARGET_KEYWORDS if kw in c.lower().replace("_"," ")))
              for c in df.columns]
    return [c for c, _ in sorted(scored, key=lambda x: x[1], reverse=True)]


def column_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One-row-per-column summary table."""
    ct       = classify_columns(df)
    type_map = {c: t.replace("_"," ").title() for t, cols in ct.items() for c in cols}
    rows = []
    for col in df.columns:
        s      = df[col]
        n_miss = int(s.isnull().sum())
        rows.append({
            "Column":  col,
            "Role":    type_map.get(col, "Unknown"),
            "Dtype":   str(s.dtype),
            "Missing": f"{n_miss:,} ({n_miss/len(df)*100:.1f}%)",
            "Unique":  f"{s.nunique():,}",
            "Sample":  ", ".join(str(v) for v in s.dropna().head(3)),
        })
    return pd.DataFrame(rows)


# SECTION 2 — AGGREGATION PIPELINE  (500 k → ≤ 2 k rows)


def _detect_time_cols(df: pd.DataFrame) -> list[str]:
    found = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            found.append(col); continue
        if any(k in col.lower() for k in ["date","month","year","time","period","week","quarter"]):
            found.append(col)
    return found


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sparse cols, dedup, downcast, strip whitespace."""
    df = df.dropna(thresh=len(df) * 0.20, axis=1).drop_duplicates()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan, "": np.nan})
    for col in df.select_dtypes(["int64","int32"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes("float64").columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    return df.reset_index(drop=True)


def _deduplicate_list(items: list) -> list:
    """
    FIX: Replaces the walrus-operator list comprehension that caused
    UnboundLocalError in some Python versions.
    Preserves order and removes duplicates.
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def aggregate_dataset(df: pd.DataFrame,
                      target_col: Optional[str] = None,
                      max_rows: int = 2000) -> tuple[pd.DataFrame, dict]:
    """
    Aggregate raw dataframe to ≤ max_rows summary rows.
    Returns (agg_df, meta_dict).
    """
    meta = {"original_rows": len(df), "original_cols": len(df.columns),
            "group_keys": [], "value_cols": [], "time_col": None,
            "agg_rows": 0, "reduction_ratio": 0.0}

    df = clean_df(df)

    # ── detect & parse time column 
    time_cols = _detect_time_cols(df)
    if time_cols:
        tc = time_cols[0]
        meta["time_col"] = tc
        if not pd.api.types.is_datetime64_any_dtype(df[tc]):
            try:
                df[tc] = pd.to_datetime(df[tc], infer_format=True, errors="coerce")
            except Exception:
                pass
        if pd.api.types.is_datetime64_any_dtype(df[tc]):
            df["_year"]    = df[tc].dt.year
            df["_quarter"] = df[tc].dt.quarter
            df["_month"]   = df[tc].dt.month

    # ── group-by candidates 
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    id_re    = re.compile(r"\bid\b|\bindex\b|\bkey\b|\bcode\b|\bserial\b", re.I)

    cat_cands = []
    for col in df.select_dtypes(include=["object","category"]).columns:
        nu = df[col].nunique()
        if 2 <= nu <= 40 and not id_re.search(col):
            cat_cands.append((col, nu))

    time_derived = [c for c in ["_year","_quarter","_month"] if c in df.columns]
    group_keys   = time_derived + [c for c, _ in sorted(cat_cands, key=lambda x: x[1])]

    # FIX: Use dedicated deduplication function instead of walrus-operator comprehension
    group_keys = _deduplicate_list(group_keys)

    # ── value columns 
    exclude    = set(group_keys) | {meta["time_col"]} | {c for c, _ in cat_cands}
    value_cols = [c for c in num_cols if c not in exclude
                  and c not in ("_year","_quarter","_month")]

    if not value_cols:
        desc = df.describe().reset_index()
        meta["agg_rows"] = len(desc)
        meta["reduction_ratio"] = round(len(desc) / meta["original_rows"], 6)
        return desc, meta

    # ── progressive groupby 
    def _groupby(keys: list[str]) -> pd.DataFrame:
        if not keys:
            row = {f"{c}_{a}": getattr(df[c], a)()
                   for c in value_cols for a in ("sum","mean","count","std")}
            return pd.DataFrame([row])
        grp = df.groupby(keys, observed=True)[value_cols].agg(["sum","mean","count","std"])
        grp.columns = [f"{c}_{a}" for c, a in grp.columns]
        return grp.reset_index()

    used_keys = group_keys.copy()
    while True:
        try:
            result = _groupby(used_keys)
            if len(result) <= max_rows or not used_keys:
                break
            used_keys = used_keys[:-1]
        except Exception:
            used_keys = used_keys[:-1]
            if not used_keys:
                result = _groupby([])
                break

    meta["group_keys"]     = used_keys
    meta["value_cols"]     = value_cols
    meta["agg_rows"]       = len(result)
    meta["reduction_ratio"] = round(len(result) / meta["original_rows"], 6)
    return result, meta


def extract_kpis(df: pd.DataFrame, agg_df: pd.DataFrame,
                 meta: dict, target_col: Optional[str] = None) -> dict:
    """Extract headline KPIs from aggregated data."""
    kpis = {
        "total_rows":    meta["original_rows"],
        "total_cols":    meta["original_cols"],
        "agg_rows":      meta["agg_rows"],
        "reduction_pct": round((1 - meta["reduction_ratio"]) * 100, 1),
        "group_keys":    meta["group_keys"],
        "value_cols":    meta["value_cols"],
    }
    vcols = meta.get("value_cols", [])
    if not vcols:
        return kpis

    primary = (target_col if target_col and f"{target_col}_sum" in agg_df.columns
               else vcols[0])
    sc = f"{primary}_sum"
    mc = f"{primary}_mean"

    if sc in agg_df.columns:
        kpis["primary_col"]   = primary
        kpis["primary_total"] = float(agg_df[sc].sum())
        kpis["primary_mean"]  = float(agg_df[mc].mean()) if mc in agg_df.columns else None

        if "_year" in agg_df.columns:
            yearly = agg_df.groupby("_year")[sc].sum()
            if len(yearly) >= 2:
                yrs    = sorted(yearly.index)
                latest = float(yearly[yrs[-1]])
                prev   = float(yearly[yrs[-2]])
                growth = (latest - prev) / prev * 100 if prev else 0
                kpis.update({
                    "yoy_growth_pct":  round(growth, 2),
                    "latest_year":     int(yrs[-1]),
                    "previous_year":   int(yrs[-2]),
                    "latest_year_val": round(latest, 2),
                    "prev_year_val":   round(prev, 2),
                })

        cat_keys = [k for k in meta.get("group_keys",[]) if not k.startswith("_")]
        if cat_keys and sc in agg_df.columns:
            top_key = cat_keys[0]
            top_grp = agg_df.groupby(top_key)[sc].sum().sort_values(ascending=False)
            kpis.update({
                "top_category_col":   top_key,
                "top_category_name":  str(top_grp.index[0]),
                "top_category_value": round(float(top_grp.iloc[0]), 2),
                "top_category_share": round(float(top_grp.iloc[0]) / float(top_grp.sum()) * 100, 1),
            })
    return kpis


def build_proc_context(df: pd.DataFrame, agg_df: pd.DataFrame,
                       meta: dict, kpis: dict) -> str:
    lines = [
        f"Original dataset: {meta['original_rows']:,} rows × {meta['original_cols']} columns.",
        f"After aggregation: {meta['agg_rows']:,} summary rows (reduction: {kpis['reduction_pct']:.1f}%).",
        f"Group keys: {', '.join(meta['group_keys']) or 'none'}.",
        f"Value columns: {', '.join(meta['value_cols'])}.",
    ]
    if "primary_col" in kpis:
        lines.append(f"Primary metric '{kpis['primary_col']}': "
                     f"total={kpis['primary_total']:,.2f}, mean={kpis.get('primary_mean','N/A')}.")
    if "yoy_growth_pct" in kpis:
        d = "increased" if kpis["yoy_growth_pct"] > 0 else "decreased"
        lines.append(f"Year-over-year: '{kpis['primary_col']}' {d} by "
                     f"{abs(kpis['yoy_growth_pct']):.1f}% from {kpis['previous_year']} "
                     f"({kpis['prev_year_val']:,.0f}) to {kpis['latest_year']} "
                     f"({kpis['latest_year_val']:,.0f}).")
    if "top_category_name" in kpis:
        lines.append(f"Top '{kpis['top_category_col']}': '{kpis['top_category_name']}' "
                     f"= {kpis['top_category_share']:.1f}% of total {kpis['primary_col']}.")
    for key in meta.get("group_keys",[])[:2]:
        if key.startswith("_") or key not in agg_df.columns:
            continue
        for vc in meta.get("value_cols",[])[:1]:
            sc = f"{vc}_sum"
            if sc not in agg_df.columns:
                continue
            top   = agg_df.groupby(key)[sc].sum().sort_values(ascending=False).head(5)
            total = top.sum()
            parts = [f"'{k}'={v:,.1f}({v/total*100:.1f}%)" for k, v in top.items()]
            lines.append(f"{key} breakdown ({vc}): {'; '.join(parts)}.")
    return "\n".join(lines)


# SECTION 3 — EDA (operates on AGGREGATED data only)

def eda_missing(df: pd.DataFrame) -> tuple[go.Figure, str]:
    miss = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    miss = miss[miss > 0]
    if miss.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values", showarrow=False,
                           font=dict(color=GREEN, size=15))
        return _fig(fig, "Missing Values", 180), \
               "Dataset is complete — no missing values. This is ideal for modeling."
    colors = [RED if v > 30 else YELLOW if v > 10 else GREEN for v in miss.values]
    fig = go.Figure(go.Bar(x=miss.index, y=miss.values, marker_color=colors,
                            text=[f"{v:.1f}%" for v in miss.values],
                            textposition="outside"))
    _fig(fig, "Missing Value Analysis (%)", 360)
    parts = []
    high = miss[miss > 30]
    med  = miss[(miss > 10) & (miss <= 30)]
    low  = miss[miss <= 10]
    if not high.empty:
        parts.append(f"**Critical (>30%):** {', '.join(f'`{c}`' for c in high.index)} — drop or impute.")
    if not med.empty:
        parts.append(f"**Moderate:** {', '.join(f'`{c}`' for c in med.index)} — median/mode imputation.")
    if not low.empty:
        parts.append(f"**Minor (<10%):** {', '.join(f'`{c}`' for c in low.index)} — safe to impute.")
    return fig, " | ".join(parts)


def eda_trend(agg_df: pd.DataFrame, meta: dict) -> tuple[Optional[go.Figure], str]:
    time_cols = [c for c in ["_year","_quarter","_month"] if c in agg_df.columns]
    val_cols  = [f"{v}_sum" for v in meta.get("value_cols",[])[:2] if f"{v}_sum" in agg_df.columns]
    if not time_cols or not val_cols:
        return None, "No time dimension detected — trend chart unavailable."
    tc  = time_cols[0]
    fig = go.Figure()
    for i, vc in enumerate(val_cols):
        ts = agg_df.groupby(tc)[vc].sum().reset_index().sort_values(tc)
        ts.columns = ["period","value"]
        fig.add_trace(go.Scatter(
            x=ts["period"].astype(str), y=ts["value"],
            mode="lines+markers", name=vc.replace("_sum",""),
            line=dict(color=PALETTE[i], width=2.5), marker=dict(size=7),
            fill="tozeroy" if i == 0 else None,
            fillcolor="rgba(108,99,255,0.07)" if i == 0 else None,
        ))
    _fig(fig, f"Trend over {tc.replace('_','').title()}", 420)
    ts2 = agg_df.groupby(tc)[val_cols[0]].sum().sort_index()
    if len(ts2) >= 2:
        pct = (ts2.iloc[-1] - ts2.iloc[-2]) / ts2.iloc[-2] * 100 if ts2.iloc[-2] else 0
        emoji  = "📈" if pct > 0 else "📉"
        metric = val_cols[0].replace("_sum","")
        ins = (f"{emoji} **`{metric}` trended {'up' if pct>0 else 'down'}** "
               f"({pct:+.1f}% vs prior). Total: {ts2.sum():,.0f}. "
               f"{'Positive trajectory — maintain strategy.' if pct > 5 else 'Decline warrants investigation.' if pct < -5 else 'Relatively stable.'}")
    else:
        ins = "Insufficient time periods to compute trend."
    return fig, ins


def eda_category(agg_df: pd.DataFrame, meta: dict) -> list[tuple[go.Figure, str]]:
    cat_keys  = [k for k in meta.get("group_keys",[]) if not k.startswith("_")]
    val_cols  = [f"{v}_sum" for v in meta.get("value_cols",[])[:1] if f"{v}_sum" in agg_df.columns]
    if not cat_keys or not val_cols:
        return []
    results, vc = [], val_cols[0]
    metric = vc.replace("_sum","")
    for key in cat_keys[:3]:
        if key not in agg_df.columns:
            continue
        grp   = agg_df.groupby(key)[vc].sum().sort_values(ascending=False).head(15)
        total = grp.sum()
        top_s = grp.iloc[0] / total * 100 if total > 0 else 0
        top2s = grp.iloc[:2].sum() / total * 100 if total > 0 else 0
        colors = [PALETTE[0] if i==0 else PALETTE[1] if i==1 else MUTED for i in range(len(grp))]
        fig = go.Figure(go.Bar(
            x=grp.index.astype(str), y=grp.values,
            marker_color=colors,
            text=[f"{v/total*100:.1f}%" for v in grp.values],
            textposition="outside",
        ))
        _fig(fig, f"{metric.title()} by {key}", 380)
        ins = (f"**`{key}`:** '{grp.index[0]}' = {top_s:.1f}% of {metric}. "
               f"Top 2 combined: {top2s:.1f}%. "
               f"{'Concentration risk.' if top_s > 50 else 'Balanced distribution.'}")
        results.append((fig, ins))
    return results


def eda_correlation(agg_df: pd.DataFrame, meta: dict,
                    target_col: Optional[str] = None) -> tuple[go.Figure, str]:
    num_df  = agg_df.select_dtypes(include=np.number)
    useful  = [c for c in num_df.columns if c.endswith(("_sum","_mean","_count"))]
    if not useful:
        useful = num_df.columns.tolist()
    if len(useful) < 2:
        return _fig(go.Figure(), "Correlation Matrix", 200), "Insufficient numerical columns."
    sub  = num_df[useful[:12]].dropna(axis=1, how="all")
    corr = sub.corr().round(3)
    fig  = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        colorbar=dict(title="r", tickfont=dict(color=TEXT)),
    ))
    _fig(fig, "Correlation Matrix (Aggregated)", 500)
    pairs = [(corr.columns[i], corr.columns[j], corr.iloc[i,j])
             for i in range(len(corr.columns))
             for j in range(i+1, len(corr.columns))]
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    ins_parts = []
    if pairs:
        c1, c2, val = pairs[0]
        d = "positive" if val > 0 else "negative"
        ins_parts.append(f"**Strongest:** `{c1}` ↔ `{c2}` (r={val:.3f}, {d}).")
    if target_col:
        tc_clean = target_col.replace(" ","_")
        for suf in ("_sum","_mean",""):
            cand = f"{tc_clean}{suf}"
            if cand in corr.columns:
                top3 = corr[cand].drop(cand).abs().sort_values(ascending=False).head(3)
                ins_parts.append(f"**Top predictors of `{target_col}`:** "
                                 + ", ".join(f"`{c}` (r={corr[cand][c]:.2f})" for c in top3.index))
                break
    return fig, " | ".join(ins_parts) if ins_parts else "No strong correlations detected."


def eda_distributions(agg_df: pd.DataFrame) -> tuple[go.Figure, str]:
    cols = [c for c in agg_df.columns if c.endswith(("_sum","_mean"))
            and agg_df[c].notna().sum() > 5][:6]
    if not cols:
        cols = agg_df.select_dtypes(include=np.number).columns.tolist()[:6]
    if not cols:
        return _fig(go.Figure(), "", 180), "No numerical columns for distribution analysis."
    n    = len(cols)
    rows = (n + 1) // 2
    fig  = make_subplots(rows=rows, cols=2,
                         subplot_titles=[c.replace("_"," ").title() for c in cols],
                         vertical_spacing=0.14)
    skew_info = {}
    for idx, col in enumerate(cols):
        r, c = divmod(idx, 2)
        s = agg_df[col].dropna()
        skew_info[col] = s.skew()
        fig.add_trace(go.Histogram(x=s, name=col, nbinsx=25,
                                    marker_color=PALETTE[idx % len(PALETTE)],
                                    opacity=0.85, showlegend=False),
                      row=r+1, col=c+1)
    _fig(fig, "Aggregated Metric Distributions", 270*rows)
    hi_skew = {k: v for k, v in skew_info.items() if abs(v) > 1.5}
    ins = ("**Distributions.** "
           + (f"Highly skewed: {', '.join(f'`{k}` (skew={v:.2f})' for k,v in list(hi_skew.items())[:3])}. "
              "Log-transform recommended." if hi_skew
              else "All metrics have moderate skewness — standard scaling is fine."))
    return fig, ins


def eda_top_n(agg_df: pd.DataFrame, meta: dict, n: int = 10) -> tuple[go.Figure, str]:
    cat_keys = [k for k in meta.get("group_keys",[]) if not k.startswith("_")]
    val_cols = [f"{v}_sum" for v in meta.get("value_cols",[])[:1] if f"{v}_sum" in agg_df.columns]
    if not cat_keys or not val_cols:
        return _fig(go.Figure(), "", 180), "No categorical grouping available."
    key    = cat_keys[0]
    vc     = val_cols[0]
    metric = vc.replace("_sum","")
    grp    = agg_df.groupby(key)[vc].sum().sort_values(ascending=True).tail(n)
    total  = grp.sum()
    colors = [ACCENT if i==len(grp)-1 else GREEN if i==len(grp)-2 else MUTED
              for i in range(len(grp))]
    fig = go.Figure(go.Bar(
        y=grp.index.astype(str), x=grp.values, orientation="h",
        marker_color=colors,
        text=[f"{v/total*100:.1f}%" for v in grp.values],
        textposition="outside",
    ))
    _fig(fig, f"Top {n} {key} by {metric.title()}", 60 + len(grp)*38)
    gap = (grp.iloc[-1] - grp.iloc[0]) / grp.iloc[0] * 100 if grp.iloc[0] > 0 else 0
    ins = (f"**Gap:** '{grp.index[-1]}' outperforms '{grp.index[0]}' by {gap:,.0f}% in {metric}. "
           f"Closing this gap = {grp.iloc[-1]-grp.iloc[0]:,.0f} incremental {metric}.")
    return fig, ins


def build_eda_context(df: pd.DataFrame, agg_df: pd.DataFrame,
                      meta: dict, target_col: Optional[str] = None) -> str:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    miss     = df.isnull().sum() / len(df) * 100
    lines    = [
        f"Dataset has {len(df):,} rows and {len(df.columns)} columns.",
        f"Numerical columns ({len(num_cols)}): {', '.join(num_cols[:10])}.",
        f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}.",
    ]
    high_miss = miss[miss > 10]
    if high_miss.empty:
        lines.append("No significant missing values.")
    else:
        for col, pct in high_miss.items():
            lines.append(f"Column '{col}' has {pct:.1f}% missing values.")
    if num_cols:
        desc = df[num_cols[:8]].describe().round(3)
        for col in desc.columns:
            lines.append(f"'{col}': mean={desc[col]['mean']}, std={desc[col]['std']}, "
                         f"min={desc[col]['min']}, max={desc[col]['max']}, "
                         f"skew={round(df[col].skew(), 3)}.")
    if len(num_cols) > 1:
        corr = df[num_cols[:10]].corr()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i,j]) > 0.5:
                    lines.append(f"'{c1}' and '{c2}' have correlation {corr.iloc[i,j]:.3f}.")
    lines.append(f"\nAggregated data ({len(agg_df)} rows after groupby {meta.get('group_keys','')}):")
    for key in meta.get("group_keys",[])[:2]:
        if key.startswith("_") or key not in agg_df.columns:
            continue
        for vc in meta.get("value_cols",[])[:1]:
            sc = f"{vc}_sum"
            if sc not in agg_df.columns:
                continue
            top   = agg_df.groupby(key)[sc].sum().sort_values(ascending=False).head(5)
            total = top.sum()
            lines.append(f"'{key}' top groups by '{vc}': " +
                         "; ".join(f"'{k}'={v:,.1f}({v/total*100:.1f}%)" for k,v in top.items()) + ".")
    if target_col and target_col in df.columns:
        s = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"Target '{target_col}': mean={s.mean():.3f}, std={s.std():.3f}, "
                         f"skew={s.skew():.3f}, task=regression.")
        else:
            vc = s.value_counts()
            lines.append(f"Target '{target_col}': {vc.nunique()} classes, "
                         f"dist={dict(vc.head(4))}, task=classification.")
    return "\n".join(lines)


# SECTION 4 — PREDICTIVE MODELING


def _detect_task(series: pd.Series) -> str:
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    return "classification" if series.nunique() <= 10 else "regression"


def _preprocess_for_model(agg_df: pd.DataFrame, target_col: str):
    """Encode, impute, return (X, y, feature_names, task, label_names)."""
    df = agg_df.copy()
    # drop count/std columns (leaky summary stats)
    drop = [c for c in df.columns if c.endswith(("_count","_std")) and c != target_col]
    df   = df.drop(columns=drop, errors="ignore")

    # resolve target (may need _sum suffix)
    if target_col not in df.columns:
        for suf in ("_sum","_mean",""):
            cand = f"{target_col}{suf}"
            if cand in df.columns:
                target_col = cand
                break

    y_raw = df[target_col].copy()

    # FIX: Convert Categorical dtype → object so LabelEncoder and _detect_task
    # never receive a pandas Categorical (which raises "TypeError: 'category'")
    if hasattr(y_raw, "cat"):
        y_raw = y_raw.astype(str)
    elif pd.api.types.is_categorical_dtype(y_raw):
        y_raw = y_raw.astype(str)

    X_df  = df.drop(columns=[target_col])

    # Also guard feature columns against Categorical dtype
    for col in X_df.select_dtypes(include=["object","category"]).columns:
        X_df[col] = LabelEncoder().fit_transform(X_df[col].astype(str))

    feature_names = X_df.columns.tolist()
    task = _detect_task(y_raw)

    if task == "classification":
        le   = LabelEncoder()
        y    = le.fit_transform(y_raw.astype(str))
        lbls = list(le.classes_)
    else:
        # FIX: ensure numeric conversion works even if y_raw came in as object
        y    = pd.to_numeric(y_raw, errors="coerce").fillna(0).values.astype(float)
        lbls = None

    X = SimpleImputer(strategy="median").fit_transform(X_df)
    return X, y, feature_names, task, lbls


def train_models(agg_df: pd.DataFrame, target_col: str) -> dict:
    X, y, feature_names, task, label_names = _preprocess_for_model(agg_df, target_col)
    n         = len(X)
    test_size = 0.20 if n >= 50 else 0.30
    n_cv      = min(5, max(2, n // 10))
    strat     = y if (task == "classification" and n >= 20) else None

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                               random_state=42, stratify=strat)

    if task == "classification":
        models  = {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=150, max_depth=8,
                                                           class_weight="balanced", random_state=42, n_jobs=-1),
            "Gradient Boosting":   GradientBoostingClassifier(n_estimators=100, max_depth=4,
                                                                learning_rate=0.1, random_state=42),
        }
        cv      = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        models  = {
            "Ridge Regression":  Ridge(alpha=1.0),
            "Random Forest":     RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42, n_jobs=-1),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                            learning_rate=0.1, random_state=42),
        }
        cv      = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scoring = "r2"

    results, trained = {}, {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred_te = model.predict(X_te)
        y_pred_tr = model.predict(X_tr)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        if task == "classification":
            acc_te = round(accuracy_score(y_te, y_pred_te) * 100, 2)
            acc_tr = round(accuracy_score(y_tr, y_pred_tr) * 100, 2)
            f1     = round(f1_score(y_te, y_pred_te, average="weighted", zero_division=0) * 100, 2)
            results[name] = {
                "Accuracy (%)":  acc_te,
                "F1-Score (%)":  f1,
                "CV Score (%)":  round(cv_scores.mean() * 100, 2),
                "CV Std":        round(cv_scores.std()  * 100, 2),
                "Train Acc (%)": acc_tr,
                "Overfit Gap":   round(acc_tr - acc_te, 2),
                "primary":       f1,
                "task":          "classification",
            }
        else:
            rmse  = round(float(np.sqrt(mean_squared_error(y_te, y_pred_te))), 4)
            r2_te = round(r2_score(y_te, y_pred_te) * 100, 2)
            r2_tr = round(r2_score(y_tr, y_pred_tr) * 100, 2)
            results[name] = {
                "R² (%)":       r2_te,
                "RMSE":         rmse,
                "CV R² (%)":    round(cv_scores.mean() * 100, 2),
                "CV Std":       round(cv_scores.std()  * 100, 2),
                "Train R² (%)": r2_tr,
                "Overfit Gap":  round(r2_tr - r2_te, 2),
                "primary":      r2_te,
                "task":         "regression",
            }
        trained[name] = model

    best_name  = max(results, key=lambda k: results[k]["primary"])
    best_model = trained[best_name]

    # feature importance
    if hasattr(best_model, "feature_importances_"):
        imps = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        coef = best_model.coef_
        imps = np.abs(coef if coef.ndim == 1 else coef.mean(axis=0))
    else:
        imps = np.ones(len(feature_names)) / len(feature_names)

    imp_df = (pd.DataFrame({"Feature": feature_names, "Importance": imps})
              .sort_values("Importance", ascending=False).head(15).reset_index(drop=True))

    # SHAP
    shap_df = None
    if SHAP_OK and hasattr(best_model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(best_model)
            sv        = explainer.shap_values(X_te[:min(100, len(X_te))])
            if isinstance(sv, list):
                sv = np.abs(np.array(sv)).mean(axis=0)
            shap_df = (pd.DataFrame({"Feature": feature_names, "SHAP": np.abs(sv).mean(axis=0)})
                       .sort_values("SHAP", ascending=False).head(15))
        except Exception:
            shap_df = None

    pred_sample           = pd.DataFrame({"Actual": y_te[:20], "Predicted": best_model.predict(X_te[:20])})
    pred_sample["Error"]  = (pred_sample["Actual"] - pred_sample["Predicted"]).abs()

    return {
        "results":        results,
        "best_name":      best_name,
        "best_model":     best_model,
        "importance_df":  imp_df,
        "shap_df":        shap_df,
        "task":           task,
        "label_names":    label_names,
        "feature_names":  feature_names,
        "pred_sample":    pred_sample,
    }


def plot_model_comparison(results: dict) -> go.Figure:
    names = list(results.keys())
    task  = results[names[0]]["task"]
    if task == "classification":
        fig = go.Figure(data=[
            go.Bar(name="Accuracy (%)", x=names,
                   y=[results[n]["Accuracy (%)"] for n in names],
                   marker_color=ACCENT, text=[f"{results[n]['Accuracy (%)']:.1f}%" for n in names],
                   textposition="outside"),
            go.Bar(name="F1-Score (%)", x=names,
                   y=[results[n]["F1-Score (%)"] for n in names],
                   marker_color=GREEN, text=[f"{results[n]['F1-Score (%)']:.1f}%" for n in names],
                   textposition="outside"),
            go.Bar(name="CV Score (%)", x=names,
                   y=[results[n]["CV Score (%)"] for n in names],
                   marker_color=YELLOW, text=[f"{results[n]['CV Score (%)']:.1f}%" for n in names],
                   textposition="outside"),
        ])
        fig.update_layout(barmode="group", yaxis_range=[0,120])
    else:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["R² Score (%)", "RMSE (lower=better)"])
        fig.add_trace(go.Bar(x=names, y=[results[n]["R² (%)"] for n in names],
                              marker_color=ACCENT, showlegend=False,
                              text=[f"{results[n]['R² (%)']:.1f}%" for n in names],
                              textposition="outside"), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=[results[n]["RMSE"] for n in names],
                              marker_color=RED, showlegend=False,
                              text=[str(results[n]["RMSE"]) for n in names],
                              textposition="outside"), row=1, col=2)
    return _fig(fig, "Model Performance Comparison", 420)


def plot_overfitting(results: dict) -> go.Figure:
    names = list(results.keys())
    task  = results[names[0]]["task"]
    if task == "classification":
        train = [results[n]["Train Acc (%)"] for n in names]
        test  = [results[n]["Accuracy (%)"]  for n in names]
        label = "Accuracy (%)"
    else:
        train = [results[n]["Train R² (%)"] for n in names]
        test  = [results[n]["R² (%)"]       for n in names]
        label = "R² (%)"
    fig = go.Figure(data=[
        go.Bar(name="Train", x=names, y=train, marker_color=YELLOW, opacity=0.8),
        go.Bar(name="Test",  x=names, y=test,  marker_color=ACCENT),
    ])
    fig.update_layout(barmode="group")
    return _fig(fig, f"Overfitting Check — Train vs Test {label}", 380)


def plot_feature_importance(imp_df: pd.DataFrame, shap_df=None) -> go.Figure:
    if shap_df is not None:
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Model Importance","SHAP Importance"])
        fig.add_trace(go.Bar(y=imp_df["Feature"][::-1], x=imp_df["Importance"][::-1],
                              orientation="h", marker_color=ACCENT, showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(y=shap_df["Feature"][::-1], x=shap_df["SHAP"][::-1],
                              orientation="h", marker_color=PURPLE, showlegend=False), row=1, col=2)
        height = 480
    else:
        fig = go.Figure(go.Bar(
            y=imp_df["Feature"][::-1], x=imp_df["Importance"][::-1],
            orientation="h",
            marker_color=[ACCENT if i<3 else GREEN if i<6 else MUTED
                          for i in range(len(imp_df)-1,-1,-1)],
            text=[f"{v:.3f}" for v in imp_df["Importance"][::-1]],
            textposition="outside",
        ))
        height = 450
    fig = _fig(fig, "Feature Importance — Best Model", height)
    fig.update_layout(margin=dict(l=160))
    return fig


def model_insight(mr: dict) -> str:
    best = mr["best_name"]
    res  = mr["results"]
    task = mr["task"]
    imp  = mr["importance_df"]
    parts = []
    if task == "classification":
        acc = res[best]["Accuracy (%)"]
        f1  = res[best]["F1-Score (%)"]
        gap = res[best]["Overfit Gap"]
        parts.append(f"**Best: {best}** — Accuracy {acc:.1f}%, F1 {f1:.1f}%. "
                     f"{'⚠️ Overfitting gap=' + str(gap) + '%' if gap > 15 else '✅ Healthy generalization.'}")
    else:
        r2   = res[best]["R² (%)"]
        rmse = res[best]["RMSE"]
        gap  = res[best]["Overfit Gap"]
        parts.append(f"**Best: {best}** — R²={r2:.1f}%, RMSE={rmse}. "
                     f"{'⚠️ Overfitting detected.' if gap > 20 else '✅ Good generalization.'}")
    top3    = imp["Feature"].head(3).tolist()
    top_pct = imp["Importance"].head(3).sum() / imp["Importance"].sum() * 100
    parts.append(f"**Top drivers:** {', '.join(f'`{f}`' for f in top3)} "
                 f"= {top_pct:.1f}% of predictive power.")
    return " | ".join(parts)


def build_model_context(mr: dict) -> str:
    best = mr["best_name"]
    res  = mr["results"]
    imp  = mr["importance_df"]
    lines = [f"Best model: {best}. Task: {mr['task']}."]
    for name, m in res.items():
        metrics = {k: v for k, v in m.items() if k not in ("primary","task")}
        lines.append(f"Model '{name}': " + ", ".join(f"{k}={v}" for k,v in metrics.items()) + ".")
    total_imp = imp["Importance"].sum()
    lines.append("Feature importances (top 10):")
    for _, row in imp.head(10).iterrows():
        pct = row["Importance"] / total_imp * 100 if total_imp > 0 else 0
        lines.append(f"  '{row['Feature']}': {row['Importance']:.4f} ({pct:.1f}%)")
    lines.append(f"Top 3 predictive features: {', '.join(imp['Feature'].head(3).tolist())}.")
    if mr.get("shap_df") is not None:
        lines.append(f"Top 3 by SHAP: {', '.join(mr['shap_df']['Feature'].head(3).tolist())}.")
    return "\n".join(lines)


# SECTION 5 — DECISION ENGINE


def generate_decisions(df: pd.DataFrame, agg_df: pd.DataFrame,
                       meta: dict, kpis: dict,
                       model_results: Optional[dict] = None,
                       domain: str = "general analytics") -> dict:
    recs, risks, optims = [], [], []

    # data quality
    miss      = df.isnull().sum() / len(df) * 100
    high_miss = miss[miss > 20]
    if not high_miss.empty:
        wc = high_miss.idxmax()
        risks.append({
            "title":    f"Data Gap in `{wc}` ({high_miss.max():.1f}% missing)",
            "body":     f"Column `{wc}` is missing {high_miss.max():.1f}% of values. "
                        f"Model reliability is reduced. **Action:** Enforce mandatory data capture within 30 days.",
            "severity": "High" if high_miss.max() > 30 else "Medium",
        })

    dup = int(df.duplicated().sum())
    if dup > 0:
        risks.append({
            "title":    f"Duplicate Records: {dup:,} rows ({dup/len(df)*100:.1f}%)",
            "body":     f"{dup:,} duplicate rows inflate metrics and bias models. "
                        f"**Action:** Audit upstream pipeline for deduplication.",
            "severity": "High" if dup/len(df) > 0.05 else "Medium",
        })
    else:
        optims.append({
            "title":    "Dataset Integrity Confirmed",
            "body":     "No duplicates found — pipeline is clean.",
            "expected_impact": "Baseline quality maintained.",
        })

    # YoY trend
    if "yoy_growth_pct" in kpis:
        g       = kpis["yoy_growth_pct"]
        primary = kpis.get("primary_col","metric")
        ly      = kpis.get("latest_year","latest")
        py      = kpis.get("previous_year","prior")
        if g < -10:
            risks.append({
                "title":    f"Significant Decline: {g:+.1f}% YoY in `{primary}`",
                "body":     f"`{primary}` fell {abs(g):.1f}% from {py} ({kpis.get('prev_year_val',0):,.0f}) "
                            f"to {ly} ({kpis.get('latest_year_val',0):,.0f}). "
                            f"**Immediate actions:** Investigate root cause, review pricing & competitive position.",
                "severity": "Critical",
            })
        elif g < 0:
            risks.append({
                "title":    f"Moderate Decline: {g:+.1f}% YoY in `{primary}`",
                "body":     f"`{primary}` contracted {abs(g):.1f}% in {ly}. Monitor closely.",
                "severity": "Medium",
            })
        elif g > 20:
            recs.append({
                "title":    f"Strong Growth Momentum: {g:+.1f}% YoY",
                "body":     f"`{primary}` grew {g:.1f}% to {kpis.get('latest_year_val',0):,.0f} in {ly}. "
                            f"**Action:** Reinvest in top segments; scale campaigns; prepare capacity.",
                "priority": "High", "type": "Growth Opportunity",
            })
        else:
            recs.append({
                "title":    f"Stable Growth: {g:+.1f}% YoY",
                "body":     f"`{primary}` shows stable {g:.1f}% growth. Focus on efficiency & market share.",
                "priority": "Medium", "type": "Maintenance",
            })

    # segment concentration
    if "top_category_share" in kpis:
        share   = kpis["top_category_share"]
        cat     = kpis["top_category_col"]
        name    = kpis["top_category_name"]
        val     = kpis["top_category_value"]
        primary = kpis.get("primary_col","metric")
        if share > 60:
            risks.append({
                "title":    f"Concentration Risk: '{name}' = {share:.1f}% of {primary}",
                "body":     f"Single segment `{name}` drives {share:.1f}% ({val:,.0f}) of {primary}. "
                            f"**Action:** Diversify — invest in bottom-20% of {cat} segments.",
                "severity": "High" if share > 70 else "Medium",
            })
        else:
            recs.append({
                "title":    f"Leverage `{name}` — Top Segment at {share:.1f}%",
                "body":     f"'{name}' leads with {share:.1f}% share. "
                            f"**Recommendation:** Replicate this segment's strategy across underperformers.",
                "priority": "High", "type": "Segment Strategy",
            })

    # performance gap
    vcols    = meta.get("value_cols",[])
    cat_keys = [k for k in meta.get("group_keys",[]) if not k.startswith("_")]
    if cat_keys and vcols:
        key = cat_keys[0]
        vc  = vcols[0]
        sc  = f"{vc}_sum"
        if key in agg_df.columns and sc in agg_df.columns:
            grp = agg_df.groupby(key)[sc].sum().sort_values()
            if len(grp) >= 3:
                bot3 = grp.head(3)
                top1 = grp.iloc[-1]
                gap_pct = (top1 - bot3.mean()) / bot3.mean() * 100 if bot3.mean() > 0 else 0
                optims.append({
                    "title":    f"Close Performance Gap in Bottom {key} Segments",
                    "body":     f"Underperforming segments — {', '.join(f'`{c}`' for c in bot3.index)} — "
                                f"avg {bot3.mean():,.0f} vs top {top1:,.0f} ({gap_pct:,.0f}% gap). "
                                f"10% lift = {bot3.sum()*0.1:,.0f} additional {vc}.",
                    "expected_impact": f"~{bot3.sum()*0.1:,.0f} incremental {vc} "
                                       f"(+{bot3.sum()*0.1/grp.sum()*100:.1f}%)",
                })

    # model-driven
    if model_results:
        best = model_results["best_name"]
        res  = model_results["results"]
        task = model_results["task"]
        imp  = model_results["importance_df"]
        gap  = res[best]["Overfit Gap"]
        pm   = res[best].get("Accuracy (%)", res[best].get("R² (%)"))
        if gap > 15:
            risks.append({
                "title":    f"Model Overfitting — {best} (gap {gap:.1f}%)",
                "body":     f"Train-test gap of {gap:.1f}% suggests the model may underperform in production. "
                            f"**Action:** Add regularization or collect more diverse training data.",
                "severity": "Medium",
            })
        tf  = imp.iloc[0]["Feature"]
        tp  = imp.iloc[0]["Importance"] / imp["Importance"].sum() * 100
        recs.append({
            "title":    f"Prioritize `{tf}` — {tp:.1f}% of Prediction Power",
            "body":     f"`{tf}` is the single strongest predictor at {tp:.1f}%. "
                        f"Optimizing this variable delivers the highest business ROI.",
            "priority": "High", "type": "Model-Driven",
        })
        if pm >= 80:
            recs.append({
                "title":    f"Deploy {best} — {pm:.1f}% Validated",
                "body":     f"{best} achieves {pm:.1f}% with healthy generalization. "
                            f"Set up monitoring to retrain when performance drops >5%.",
                "priority": "High", "type": "Deployment Ready",
            })

    if meta["original_rows"] < 1000:
        risks.append({
            "title":    f"Small Dataset ({meta['original_rows']:,} rows)",
            "body":     "Model estimates have high variance. "
                        "Collect ≥5,000 records for robust ML.",
            "severity": "Medium",
        })

    return {"recommendations": recs, "risks": risks, "optimizations": optims}


def build_decision_context(decisions: dict, kpis: dict) -> str:
    lines = []
    if "yoy_growth_pct" in kpis:
        g = kpis["yoy_growth_pct"]
        lines.append(f"Year-over-year performance: {g:+.1f}% change in "
                     f"'{kpis.get('primary_col','metric')}' "
                     f"from {kpis.get('previous_year','prior')} to {kpis.get('latest_year','latest')}.")
    if "top_category_name" in kpis:
        lines.append(f"Top segment: '{kpis['top_category_name']}' in '{kpis['top_category_col']}' "
                     f"= {kpis['top_category_share']:.1f}% of total {kpis.get('primary_col','metric')}.")
    for section, items in [("Business Recommendations", decisions.get("recommendations",[])),
                            ("Risk Alerts",             decisions.get("risks",[])),
                            ("Optimizations",           decisions.get("optimizations",[]))]:
        lines.append(f"\n{section} ({len(items)}):")
        for item in items:
            prio = item.get("priority", item.get("severity","Med"))
            lines.append(f"  [{prio}] {item['title']}: {item['body'][:200]}")
    return "\n".join(lines)


# SECTION 6 — RAG CHATBOT


SYSTEM_PROMPT = (
    "You are a senior business analyst AI. "
    "Answer ONLY using the provided dataset insights and model outputs. "
    "Every answer MUST include specific numbers, key drivers, reasoning, and an action. "
    "Do NOT give generic answers."
)


def _chunk(text: str, size: int = 180, overlap: int = 40) -> list[str]:
    words  = text.split()
    chunks, step = [], size - overlap
    for i in range(0, max(1, len(words) - overlap), step):
        c = " ".join(words[i: i + size])
        if len(c.strip()) > 20:
            chunks.append(c)
    return chunks or [text]


class _VectorIndex:
    def __init__(self):
        self.model  = None
        self.index  = None
        self.chunks = []
        self._ready = False

    def build(self, context: str):
        self.chunks = _chunk(context)
        if not self.chunks:
            return
        if ST_OK and not self._ready:
            try:
                self.model  = SentenceTransformer("all-MiniLM-L6-v2")
                self._ready = True
            except Exception:
                pass
        if not self._ready or not FAISS_OK:
            return
        emb = self.model.encode(self.chunks, show_progress_bar=False,
                                batch_size=64).astype("float32")
        faiss.normalize_L2(emb)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)

    def retrieve(self, query: str, k: int = 7) -> list[str]:
        if not self.chunks:
            return []
        if self.index is None or not self._ready:
            qw = set(query.lower().split())
            return sorted(self.chunks,
                          key=lambda c: len(qw & set(c.lower().split())),
                          reverse=True)[:k] or self.chunks[:k]
        q = self.model.encode([query], show_progress_bar=False).astype("float32")
        faiss.normalize_L2(q)
        _, idx = self.index.search(q, min(k, len(self.chunks)))
        return [self.chunks[i] for i in idx[0] if i < len(self.chunks)]


def _grounded_answer(query: str, chunks: list[str], full: str) -> str:
    q   = query.lower()
    ctx = "\n".join(chunks) if chunks else full

    def _find(pat):
        m = re.search(pat, ctx, re.IGNORECASE)
        return m.group(1).rstrip(".") if m else None

    # feature importance
    if any(w in q for w in ["feature","important","driver","impact","factor","matter","predict"]):
        imp_matches = re.findall(r"'([^']+)':\s*([\d.]+)\s*\(([\d.]+)%\)", ctx)
        top3_match  = _find(r"Top 3 predictive features?:\s*([^\n]+)")
        if imp_matches:
            top3     = sorted(imp_matches, key=lambda x: float(x[1]), reverse=True)[:3]
            feat_str = "; ".join(f"**{f}** ({p}%)" for f,_,p in top3)
            return (f"The top predictive features are: {feat_str}. "
                    f"**Business implication:** Optimizing these delivers the highest ROI — "
                    f"prioritize them in strategy planning.")
        if top3_match:
            return (f"Top features: **{top3_match}**. "
                    f"Focus resources on understanding and optimizing these variables.")
        return "Run model training in the Modeling tab to see ranked feature importances."

    # model performance
    if any(w in q for w in ["model","accuracy","r2","r²","rmse","f1","score","best model","performance"]):
        best_m = _find(r"Best model:\s*([^.\n]+)")
        acc    = _find(r"Accuracy \(%\)=([\d.]+)")
        f1     = _find(r"F1-Score \(%\)=([\d.]+)")
        r2     = _find(r"R² \(%\)=([\d.]+)")
        gap    = _find(r"Overfit Gap=([\d.\-]+)")
        if best_m and (acc or r2):
            metric = f"accuracy {acc}%, F1 {f1}%" if acc else f"R²={r2}%"
            gap_note = (f" Train-test gap of {gap}% — slight overfitting risk."
                        if gap and float(gap) > 10 else " Generalization is healthy.")
            return (f"Best model: **{best_m}** with {metric}.{gap_note} "
                    f"{'Suitable for automated decision support.' if acc and float(acc) >= 75 else 'Consider more features or data.'}")
        return "Train models in the Modeling tab to see accuracy, F1, and R² metrics."

    # growth / trend
    if any(w in q for w in ["growth","trend","increase","decrease","decline","drop","yoy","year"]):
        yoy    = _find(r"Year-over-year.*?([+\-]?[\d.]+)%\s*change")
        col    = _find(r"change in '([^']+)'")
        latest = _find(r"to (\d{4})\.")
        prev   = _find(r"from (\d{4}) to")
        if yoy and col:
            d = "grew" if float(yoy) > 0 else "declined"
            return (f"**`{col}` {d} {abs(float(yoy)):.1f}% YoY** "
                    f"(from {prev or 'prior year'} to {latest or 'latest year'}). "
                    f"{'Positive trajectory — continue investing in top segments.' if float(yoy) > 0 else 'Decline — review segment data, pricing, and competitive dynamics.'}")
        return "No time-series trend detected. Add a date column to enable YoY analysis."

    # segments
    if any(w in q for w in ["segment","region","category","top","worst","performing","breakdown"]):
        top_seg   = _find(r"Top segment: '([^']+)'")
        top_col   = _find(r"in '([^']+)'\s*=")
        top_share = _find(r"=\s*([\d.]+)%\s*of total")
        bottom    = re.findall(r"Underperforming[^:]+:\s*([^\-]+)", ctx)
        parts = []
        if top_seg and top_share:
            parts.append(f"**Top segment:** '{top_seg}' in `{top_col}` = {top_share}% of total.")
        if bottom:
            parts.append(f"**Underperformers:** {bottom[0].strip()[:100]}. Targeted action here yields the highest incremental gain.")
        return " ".join(parts) + " See EDA tab for full breakdowns." if parts else \
               "Segment breakdowns are in the EDA tab — run the full pipeline first."

    # data quality / missing
    if any(w in q for w in ["missing","null","quality","complete","gap","duplicate"]):
        no_miss = "No significant missing values" in ctx or "Dataset is complete" in ctx
        if no_miss:
            return "✅ Dataset has no significant missing values — clean and ready for modeling."
        miss_info = re.findall(r"Column '([^']+)' has ([\d.]+)% missing", ctx)
        if miss_info:
            parts = [f"**`{c}`**: {p}% missing" for c,p in miss_info[:3]]
            return "Data quality issues: " + "; ".join(parts) + ". Fix at source to improve model reliability."
        dup = _find(r"([\d,]+) duplicate")
        if dup:
            return f"**{dup} duplicate records** detected — inflates metrics and biases models. Resolve upstream."
        return "Run the pipeline to see a full data quality report in the Overview tab."

    # risks
    if any(w in q for w in ["risk","alert","danger","warning","concern"]):
        risks = re.findall(r"\[(\w+)\]\s*([^:]+):\s*([^\n]{30,120})", ctx)
        high  = [r for r in risks if r[0] in ("High","Critical","Medium")][:4]
        if high:
            parts = [f"**[{s}] {t}** — {b[:100]}" for s,t,b in high]
            return "Active risk alerts:\n\n" + "\n\n".join(parts) + "\n\nSee Decisions tab for full details."
        return "Risk analysis is in the Decisions tab — run the full pipeline to generate alerts."

    # recommendations
    if any(w in q for w in ["recommend","suggest","should","action","strategy","improve","optimize","next"]):
        recs = re.findall(r"\[High\]\s*([^:]+):\s*([^\n]{40,200})", ctx)[:3]
        if recs:
            parts = [f"**{t}:** {b[:120]}" for t,b in recs]
            return "Top business recommendations:\n\n" + "\n\n".join(parts) + "\n\nSee Decisions tab for full details."
        return "Recommendations are in the Decisions tab. Run the pipeline to generate them."

    # dataset size
    if any(w in q for w in ["row","size","record","how many","shape","large"]):
        orig = _find(r"Original dataset:\s*([\d,]+)\s*rows")
        cols = _find(r"rows\s*×\s*(\d+)\s*columns")
        aggr = _find(r"After aggregation:\s*([\d,]+)\s*summary rows")
        red  = _find(r"reduction:\s*([\d.]+)%")
        if orig:
            return (f"Dataset: **{orig} rows × {cols or '?'} columns**. "
                    f"Aggregated to **{aggr} summary rows** ({red}% compression). "
                    f"Full dataset processed efficiently via groupby summarization.")
        return "Dataset size is on the Overview tab."

    # correlations
    if any(w in q for w in ["correlation","relate","relationship","together","linear"]):
        corr_m = re.findall(r"'([^']+)' and '([^']+)' have correlation ([\-\d.]+)", ctx)
        if corr_m:
            top = sorted(corr_m, key=lambda x: abs(float(x[2].rstrip("."))), reverse=True)[:3]
            parts = [f"**`{c1}`** ↔ **`{c2}`**: r={float(v.rstrip('.')):.3f} "
                     f"({'positive' if float(v.rstrip('.'))>0 else 'negative'})"
                     for c1,c2,v in top]
            return "Key correlations: " + "; ".join(parts) + ". See EDA tab for the full heatmap."
        return "Correlation heatmap is in the EDA tab — run the pipeline first."

    # generic fallback
    nums = re.findall(r"[\w\s]+?(?:=|:)\s*([\d.,]+(?:%|k|M)?)", ctx)[:5]
    if nums:
        return ("Dataset analysis summary: " + "; ".join(nums[:4]) + ". "
                "Ask about: features, model accuracy, growth trends, segments, risks, or recommendations.")
    return ("I can answer grounded questions about your data. Try:\n"
            "• 'Which features impact predictions most?'\n"
            "• 'What is the best model accuracy?'\n"
            "• 'Why did [metric] change year-over-year?'\n"
            "• 'What are the top risks?'\n"
            "• 'What actions should the business take?'")


class Chatbot:
    """RAG chatbot — grounded exclusively in aggregated context."""

    def __init__(self):
        self._idx     = _VectorIndex()
        self.full_ctx = ""
        self.is_ready = False
        self.history  = []

    def build(self, *context_parts: str):
        self.full_ctx = "\n\n".join(p for p in context_parts if p)
        self._idx.build(self.full_ctx)
        self.is_ready = True

    def chat(self, question: str, api_key: Optional[str] = None) -> str:
        if not self.is_ready:
            return "⚠️ Run the pipeline first to activate the chatbot."
        chunks = self._idx.retrieve(question, k=7)
        answer = _grounded_answer(question, chunks, self.full_ctx)
        if api_key:
            try:
                import urllib.request, json as _json
                ctx_str = "\n\n".join(chunks) or self.full_ctx[:2500]
                payload = _json.dumps({
                    "model": "claude-sonnet-4-6",
                    "max_tokens": 500,
                    "system": SYSTEM_PROMPT,
                    "messages": [{"role": "user",
                                  "content": f"CONTEXT:\n{ctx_str}\n\nQUESTION: {question}"}],
                }).encode()
                req = urllib.request.Request(
                    "https://api.anthropic.com/v1/messages",
                    data=payload,
                    headers={"x-api-key": api_key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = _json.loads(resp.read())
                    answer = data["content"][0]["text"]
            except Exception:
                pass
        self.history.append(("user", question))
        self.history.append(("bot", answer))
        return answer

    def clear(self):
        self.history.clear()

    @staticmethod
    def suggestions() -> list[str]:
        return [
            "Which features impact predictions most?",
            "What is the best model and its accuracy?",
            "Why did the primary metric change year-over-year?",
            "Which segments are underperforming?",
            "What are the top business risks?",
            "What actions should the business take?",
            "Are there any data quality issues?",
            "What correlations exist in the dataset?",
        ]


# SECTION 7 — REPORT GENERATOR


def generate_report(df: pd.DataFrame, agg_df: pd.DataFrame, meta: dict,
                    kpis: dict, domain: str, domain_summary: str,
                    model_results: Optional[dict], decisions: dict,
                    target_col: Optional[str]) -> str:
    now  = datetime.now().strftime("%B %d, %Y — %H:%M")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    miss     = df.isnull().sum() / len(df) * 100
    dup      = int(df.duplicated().sum())

    lines = [
        "# 📊 AI Business Analyst Copilot — Business Report",
        f"*Generated: {now}*", "", "---", "",
        "## 1. Executive Summary", "",
        f"> {domain_summary}", "",
    ]

    if "primary_col" in kpis:
        lines += [
            "**Key Metrics:**", "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Dataset Size | {meta['original_rows']:,} rows × {meta['original_cols']} columns |",
            f"| Aggregated To | {meta['agg_rows']:,} rows ({kpis['reduction_pct']:.1f}% compression) |",
            f"| `{kpis['primary_col']}` Total | {kpis['primary_total']:,.2f} |",
        ]
        if "yoy_growth_pct" in kpis:
            d = "↑" if kpis["yoy_growth_pct"] > 0 else "↓"
            lines.append(f"| YoY Growth | {d} {abs(kpis['yoy_growth_pct']):.1f}% |")
        if "top_category_name" in kpis:
            lines.append(f"| Top {kpis['top_category_col']} | {kpis['top_category_name']} "
                         f"({kpis['top_category_share']:.1f}%) |")
        lines.append("")

    lines += [
        "## 2. Dataset Overview", "",
        f"**Domain:** {domain.title()}",
        f"- **Records:** {len(df):,}",
        f"- **Columns:** {len(df.columns)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)",
        f"- **Duplicates:** {dup:,} ({dup/len(df)*100:.1f}%)",
        f"- **Group-by keys:** {', '.join(meta.get('group_keys',[])) or 'None'}", "",
    ]

    hi_miss = miss[miss > 5]
    if not hi_miss.empty:
        lines += ["**Data Quality Issues:**", ""]
        for col, pct in hi_miss.sort_values(ascending=False).items():
            badge = "🔴" if pct > 30 else "🟡" if pct > 10 else "🟢"
            lines.append(f"- {badge} `{col}`: {pct:.1f}% missing")
        lines.append("")
    else:
        lines.append("✅ **No significant missing values.**\n")

    lines += ["## 3. Key EDA Insights", ""]
    if "yoy_growth_pct" in kpis:
        g = kpis["yoy_growth_pct"]
        lines.append(f"{'📈' if g>0 else '📉'} **YoY:** `{kpis['primary_col']}` "
                     f"{'grew' if g>0 else 'declined'} {abs(g):.1f}% "
                     f"from {kpis['previous_year']} ({kpis['prev_year_val']:,.0f}) "
                     f"to {kpis['latest_year']} ({kpis['latest_year_val']:,.0f}).\n")
    if "top_category_name" in kpis:
        lines.append(f"**Top Segment:** `{kpis['top_category_name']}` = "
                     f"{kpis['top_category_share']:.1f}% of `{kpis['primary_col']}`.\n")
    if len(num_cols) > 1:
        corr   = df[num_cols[:10]].corr()
        strong = [(num_cols[i], num_cols[j], corr.iloc[i,j])
                  for i in range(len(num_cols[:10]))
                  for j in range(i+1, len(num_cols[:10]))
                  if abs(corr.iloc[i,j]) >= 0.6]
        if strong:
            lines.append("**Strong Correlations:**")
            for c1,c2,v in sorted(strong, key=lambda x: abs(x[2]), reverse=True)[:5]:
                lines.append(f"- `{c1}` ↔ `{c2}`: r={v:.3f} ({'positive' if v>0 else 'negative'})")
            lines.append("")

    if target_col and target_col in df.columns:
        lines += [f"## 4. Target Variable: `{target_col}`", ""]
        s = df[target_col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            lines += [f"- Task: **Regression**",
                      f"- Mean: {s.mean():.4f}  |  Std: {s.std():.4f}",
                      f"- Range: {s.min():.4f} → {s.max():.4f}",
                      f"- Skewness: {s.skew():.3f} "
                      f"{'(⚠️ log-transform)' if abs(s.skew())>1 else '(✅ near-normal)'}", ""]
        else:
            vc = s.value_counts()
            lines += [f"- Task: **Classification** ({vc.nunique()} classes)"]
            for cls, cnt in vc.items():
                lines.append(f"  - `{cls}`: {cnt:,} ({cnt/len(s)*100:.1f}%)")
            if vc.iloc[0]/len(s) > 0.70:
                lines.append("\n⚠️ Class imbalance — use `class_weight='balanced'`.")
            lines.append("")

    if model_results:
        res  = model_results["results"]
        best = model_results["best_name"]
        task = model_results["task"]
        imp  = model_results["importance_df"]
        lines += ["## 5. ML Model Performance", "", f"**🏆 Best Model: {best}**", ""]
        if task == "classification":
            lines += ["| Model | Accuracy | F1 | CV Score | Overfit |",
                      "|-------|----------|----|----------|---------|"]
            for n, m in res.items():
                star = " ⭐" if n == best else ""
                flag = " ⚠️" if m["Overfit Gap"] > 15 else ""
                lines.append(f"| {n}{star} | {m['Accuracy (%)']}% | {m['F1-Score (%)']}% | "
                              f"{m['CV Score (%)']}% ±{m['CV Std']} | {m['Overfit Gap']}{flag} |")
        else:
            lines += ["| Model | R² | RMSE | CV R² | Overfit |",
                      "|-------|----|----|------|---------|"]
            for n, m in res.items():
                star = " ⭐" if n == best else ""
                flag = " ⚠️" if m["Overfit Gap"] > 20 else ""
                lines.append(f"| {n}{star} | {m['R² (%)']}% | {m['RMSE']} | "
                              f"{m['CV R² (%)']}% ±{m['CV Std']} | {m['Overfit Gap']}{flag} |")
        lines.append("")
        total_imp = imp["Importance"].sum()
        lines.append("**Top 5 Features:**")
        for i,(_, row) in enumerate(imp.head(5).iterrows()):
            pct = row["Importance"]/total_imp*100
            lines.append(f"{i+1}. `{row['Feature']}` — {pct:.1f}%")
        lines.append("")

    lines += ["## 6. Business Recommendations", ""]
    for i, rec in enumerate(decisions.get("recommendations",[]), 1):
        p = rec.get("priority","Med")
        b = "🟢" if p=="High" else "🟡" if p=="Medium" else "⚪"
        lines += [f"### {b} {i}. {rec['title']}",
                  f"**Priority:** {p} | **Type:** {rec.get('type','')}", "", rec["body"], ""]

    lines += ["## 7. Risk Analysis", ""]
    for risk in decisions.get("risks",[]):
        s = risk.get("severity","Medium")
        b = "🔴" if s=="Critical" else "🟠" if s=="High" else "🟡"
        lines += [f"### {b} {risk['title']}", f"**Severity:** {s}", "", risk["body"], ""]

    if decisions.get("optimizations"):
        lines += ["## 8. Optimization Opportunities", ""]
        for opt in decisions["optimizations"]:
            lines += [f"### 💡 {opt['title']}",
                      f"**Expected Impact:** {opt.get('expected_impact','TBD')}", "", opt["body"], ""]

    lines += [
        "## 9. Next Steps", "",
        "1. **Week 1:** Address critical data quality issues.",
        "2. **Month 1:** Deploy best model with A/B testing.",
        "3. **Quarter 1:** Implement segment-targeted strategies.",
        "4. **Ongoing:** Automate retraining when model drops >5%.",
        "5. **Governance:** Set up data quality KPI dashboard.", "",
        "---",
        "*AI Business Analyst Copilot — all insights are data-derived.*",
    ]
    return "\n".join(lines)


# SECTION 8 — EXAMPLE DATA GENERATORS


def _make_retail(n: int = 50_000) -> pd.DataFrame:
    rng        = np.random.default_rng(42)
    regions    = ["North","South","East","West","Central"]
    categories = ["Electronics","Clothing","Food & Beverage","Home & Garden","Sports","Beauty","Automotive"]
    channels   = ["Online","In-Store","Mobile App","Wholesale"]

    # FIX: Convert DatetimeIndex to list of pandas Timestamps so .month works correctly
    months = list(pd.date_range("2020-01-01", periods=48, freq="MS"))

    n_per      = n // (len(regions) * len(categories))
    rows       = []
    region_mult   = {"North":1.1,"South":0.9,"East":1.2,"West":1.0,"Central":0.95}
    category_mult = {"Electronics":2.5,"Clothing":1.0,"Food & Beverage":0.6,
                     "Home & Garden":1.3,"Sports":1.1,"Beauty":0.9,"Automotive":2.0}
    for region in regions:
        for category in categories:
            for _ in range(int(n_per * rng.uniform(0.7, 1.3))):
                # FIX: Use integer index to pick from list, giving a pandas Timestamp
                mo_idx   = int(rng.integers(0, len(months)))
                mo       = months[mo_idx]
                # mo is now a pandas Timestamp — .month works correctly
                seasonal = 1.0 + 0.3 * np.sin((mo.month - 1) * np.pi / 6)
                disc     = rng.uniform(0, 45)
                units    = int(rng.exponential(30) + 1)
                price    = rng.uniform(10, 300) * category_mult[category]
                mkt      = rng.uniform(200, 8000)
                sales    = (units * price * (1 - disc/100) * region_mult[region] * seasonal
                            + mkt * 0.25 + rng.normal(0, 300))
                rows.append({"date": mo, "region": region, "category": category,
                              "channel": channels[int(rng.integers(0, len(channels)))],
                              "discount_pct": round(disc, 1),
                              "units_sold": units,
                              "unit_price": round(price, 2),
                              "marketing_spend": round(mkt, 0),
                              "customer_rating": round(float(rng.uniform(2.5, 5.0)), 1),
                              "sales": round(max(sales, 0), 2)})
    df = pd.DataFrame(rows)
    for col in ["customer_rating","marketing_spend"]:
        idx = rng.integers(0, len(df), int(len(df)*0.02))
        df.loc[idx, col] = np.nan
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def _make_churn(n: int = 20_000) -> pd.DataFrame:
    rng       = np.random.default_rng(99)
    contracts = ["Month-to-Month","One Year","Two Year"]
    plans     = ["Basic","Standard","Premium"]
    rows      = []
    for _ in range(n):
        tenure   = int(rng.integers(1, 72))
        monthly  = round(float(rng.uniform(20, 120)), 2)
        contract = contracts[int(rng.choice([0,1,2], p=[0.55, 0.25, 0.20]))]
        plan     = plans[int(rng.integers(0, len(plans)))]
        calls    = int(rng.integers(0, 12))
        products = int(rng.integers(1, 5))
        prob     = float(np.clip(0.30 + 0.25*(monthly>90) + 0.30*(tenure<12)
                           + 0.15*(calls>5) - 0.35*(contract=="Two Year"), 0.02, 0.95))
        rows.append({
            "tenure_months":   tenure,
            "monthly_charges": monthly,
            "total_charges":   round(tenure * monthly * float(rng.uniform(0.9, 1.1)), 2),
            "contract_type":   contract,
            "plan":            plan,
            "support_calls":   calls,
            "num_products":    products,
            "churn":           "Yes" if float(rng.uniform()) < prob else "No",
        })
    return pd.DataFrame(rows)


EXAMPLES = {
    "Large Retail (50k rows)": {
        "fn": _make_retail, "target": "sales",
        "desc": "50,000 retail transactions — 5 regions, 7 categories, 4 channels, 4 years",
    },
    "Customer Churn (20k rows)": {
        "fn": _make_churn, "target": "churn",
        "desc": "20,000 customer records with tenure, charges, contract, and churn label",
    },
}

# SECTION 9 — GLOBAL CSS

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
html, body, [class*="css"] {{
    font-family: 'DM Sans', sans-serif;
    background-color: {BG};
    color: {TEXT};
}}
.hdr {{ padding:20px 0 12px; border-bottom:1px solid {BORDER}; margin-bottom:20px; }}
.hdr-title {{
    font-size:26px; font-weight:700; letter-spacing:-.5px;
    background:linear-gradient(135deg,{ACCENT},{CYAN});
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}}
.hdr-sub {{ font-size:13px; color:{MUTED}; margin-top:2px; }}
.kpi-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
    padding:16px 20px; flex:1; min-width:130px;
    transition:border-color .2s,transform .15s;
}}
.kpi-card:hover {{ border-color:{ACCENT}; transform:translateY(-1px); }}
.kpi-val {{ font-size:26px; font-weight:700; color:{ACCENT};
            font-family:'DM Mono',monospace; line-height:1.1; }}
.kpi-val.g {{ color:{GREEN}; }} .kpi-val.r {{ color:{RED}; }} .kpi-val.c {{ color:{CYAN}; }}
.kpi-lbl {{ font-size:11px; color:{MUTED}; text-transform:uppercase; letter-spacing:.7px; margin-top:4px; }}
.ins {{
    padding:12px 16px; border-radius:0 8px 8px 0; margin:8px 0;
    font-size:13.5px; line-height:1.65; border-left:3px solid {ACCENT};
    background:linear-gradient(135deg,#14183A,{CARD});
}}
.ins.g {{ border-color:{GREEN};  background:linear-gradient(135deg,#0A2018,{CARD}); }}
.ins.r {{ border-color:{RED};    background:linear-gradient(135deg,#1E0A0A,{CARD}); }}
.ins.y {{ border-color:{YELLOW}; background:linear-gradient(135deg,#1E1800,{CARD}); }}
.ins.c {{ border-color:{CYAN};   background:linear-gradient(135deg,#041820,{CARD}); }}
.chat-user {{
    background:{CARD2}; border-radius:12px 12px 4px 12px;
    padding:10px 14px; margin:8px 0; max-width:78%; margin-left:auto; font-size:14px;
}}
.chat-bot {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:4px 12px 12px 12px;
    padding:12px 16px; margin:8px 0; max-width:90%; font-size:14px; line-height:1.7;
}}
.chat-lbl {{ font-size:10px; color:{MUTED}; text-transform:uppercase; letter-spacing:.8px; margin-bottom:3px; }}
.dec-card {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:10px;
    padding:14px 18px; margin:8px 0; transition:border-color .2s;
}}
.dec-card.hi {{ border-left:3px solid {GREEN}; }}
.dec-card.me {{ border-left:3px solid {YELLOW}; }}
.dec-card.cr {{ border-left:3px solid {RED}; }}
.dec-card:hover {{ border-color:{ACCENT}; }}
.dec-title {{ font-weight:600; font-size:15px; margin-bottom:6px; }}
.dec-body  {{ font-size:13px; color:#B0B8D0; line-height:1.65; }}
.model-card {{ background:{CARD}; border:1px solid {BORDER}; border-radius:10px; padding:12px 16px; margin:6px 0; }}
.model-card.best {{ border-color:{ACCENT}; background:linear-gradient(135deg,#14183A,{CARD}); }}
.badge {{ display:inline-block; padding:2px 9px; border-radius:20px; font-size:11px; font-weight:600; }}
.badge-g {{ background:#0D2E1E; color:{GREEN}; }}
.badge-y {{ background:#1E1A00; color:{YELLOW}; }}
.badge-r {{ background:#1E0A0A; color:{RED}; }}
.badge-b {{ background:#13183A; color:{ACCENT}; }}
.pipeline {{
    display:flex; gap:0; align-items:center; background:{CARD};
    border:1px solid {BORDER}; border-radius:10px; padding:10px 16px; margin:12px 0; overflow-x:auto;
}}
.pipe-step {{ font-size:12px; color:{MUTED}; white-space:nowrap; padding:2px 6px; }}
.pipe-step.done {{ color:{GREEN}; font-weight:600; }}
.pipe-arrow {{ color:{BORDER}; margin:0 2px; }}
[data-testid="stSidebar"] {{ background:#0C0F1A !important; border-right:1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ font-size:13.5px; font-weight:500; }}
.stProgress > div > div {{ background-color:{ACCENT} !important; }}
::-webkit-scrollbar {{ width:5px; }}
::-webkit-scrollbar-thumb {{ background:{BORDER}; border-radius:3px; }}
</style>
""", unsafe_allow_html=True)


# SECTION 10 — SESSION STATE


def _init_state():
    defaults = dict(
        df=None, agg_df=None, meta=None, kpis=None,
        domain="", domain_summary="",
        target=None,
        eda_ctx="", model_results=None, decisions=None,
        chatbot=Chatbot(),
        chat_history=[],
        done=dict(ingest=False, process=False, eda=False, model=False, decisions=False),
        report_text="",
        file_name="",
    )
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()
S = st.session_state


# SECTION 11 — PIPELINE HELPERS


def _do_ingest(df: pd.DataFrame, file_name: str):
    S.df, S.file_name        = df, file_name
    S.domain, S.domain_summary = infer_domain(df)
    S.done  = dict(ingest=True, process=False, eda=False, model=False, decisions=False)
    S.model_results = None
    S.decisions     = None
    S.report_text   = ""
    S.chat_history  = []
    S.chatbot       = Chatbot()


def _do_process():
    agg_df, meta = aggregate_dataset(S.df, target_col=S.target, max_rows=2000)
    kpis         = extract_kpis(S.df, agg_df, meta, target_col=S.target)
    S.agg_df, S.meta, S.kpis = agg_df, meta, kpis
    S.done["process"] = True


def _rebuild_chatbot():
    proc_ctx  = build_proc_context(S.df, S.agg_df, S.meta, S.kpis)
    eda_ctx   = build_eda_context(S.df, S.agg_df, S.meta, S.target)
    model_ctx = build_model_context(S.model_results) if S.model_results else ""
    dec_ctx   = build_decision_context(S.decisions, S.kpis) if S.decisions else ""
    S.eda_ctx = eda_ctx
    S.chatbot.build(proc_ctx, eda_ctx, model_ctx, dec_ctx)


def _pipeline_bar():
    steps = [
        ("Ingest",    S.done["ingest"]),
        ("Aggregate", S.done["process"]),
        ("EDA",       S.done["eda"]),
        ("Model",     S.done["model"]),
        ("Decisions", S.done["decisions"]),
    ]
    parts = []
    for name, done in steps:
        cls = "done" if done else "pipe-step"
        parts.append(f'<span class="{cls}">{"✓ " if done else ""}{name}</span>')
        parts.append('<span class="pipe-arrow">→</span>')
    st.markdown('<div class="pipeline">' + "".join(parts[:-1]) + "</div>",
                unsafe_allow_html=True)


# SECTION 12 — SIDEBAR


with st.sidebar:
    st.markdown(f"""
    <div style="padding:8px 0 18px;">
      <div style="font-size:20px;font-weight:700;background:linear-gradient(135deg,{ACCENT},{CYAN});
                  -webkit-background-clip:text;-webkit-text-fill-color:transparent;">🧠 Copilot v2</div>
      <div style="font-size:11px;color:{MUTED};">Large-Scale AI Business Analyst</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📂 Data Source")

    with st.expander("🎯 Example Datasets", expanded=False):
        for ex_name, ex_info in EXAMPLES.items():
            st.caption(ex_info["desc"])
            if st.button(f"Load — {ex_name}", key=f"ex_{ex_name}", use_container_width=True):
                with st.spinner(f"Generating {ex_name}..."):
                    df_ex = ex_info["fn"]()
                _do_ingest(df_ex, f"{ex_name}.csv")
                S.target = ex_info["target"]
                st.success(f"✅ {len(df_ex):,} rows loaded")
                st.rerun()

    uploaded = st.file_uploader("Upload CSV / Excel (up to 500 MB)",
                                type=["csv","xlsx","xls"])
    if uploaded:
        with st.spinner("Loading file..."):
            try:
                df_up = load_file(uploaded)
                _do_ingest(df_up, uploaded.name)
                st.success(f"✅ {len(df_up):,} rows × {len(df_up.columns)} columns")
            except Exception as e:
                st.error(f"Load error: {e}")

    st.divider()

    if S.df is not None:
        st.markdown("### 🎯 Target Column")
        suggested  = suggest_targets(S.df)
        target_sel = st.selectbox("Select target",
                                   ["(None — EDA only)"] + suggested,
                                   index=1 if suggested else 0)
        S.target = None if target_sel == "(None — EDA only)" else target_sel

        st.divider()
        st.markdown("### ⚙️ Pipeline Controls")

        if st.button("▶ Run Full Pipeline", type="primary", use_container_width=True):
            with st.spinner("Step 1/4 — Aggregating data..."):
                _do_process()
            with st.spinner("Step 2/4 — Running EDA..."):
                S.done["eda"] = True
            with st.spinner("Step 3/4 — Generating decisions..."):
                S.decisions = generate_decisions(S.df, S.agg_df, S.meta, S.kpis,
                                                  model_results=None, domain=S.domain)
                S.done["decisions"] = True
            with st.spinner("Step 4/4 — Building AI context..."):
                _rebuild_chatbot()
            st.success("✅ Pipeline complete!")
            st.rerun()

        if S.done["process"] and S.target:
            if st.button("🤖 Train ML Models", use_container_width=True):
                with st.spinner("Training models (30–90 s)..."):
                    try:
                        S.model_results   = train_models(S.agg_df, S.target)
                        S.done["model"]   = True
                        S.decisions       = generate_decisions(S.df, S.agg_df, S.meta, S.kpis,
                                                               model_results=S.model_results,
                                                               domain=S.domain)
                        S.done["decisions"] = True
                        _rebuild_chatbot()
                        st.success("✅ Models trained!")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Training error: {e}")

        st.divider()
        df_s = S.df
        c1, c2 = st.columns(2)
        c1.metric("Rows", f"{len(df_s):,}")
        c2.metric("Cols", len(df_s.columns))
        c1.metric("Num",  len(df_s.select_dtypes(include=np.number).columns))
        c2.metric("Cat",  len(df_s.select_dtypes(include=["object","category"]).columns))
        st.caption(f"Domain: **{S.domain.title()}**")


# SECTION 13 — MAIN HEADER


st.markdown("""
<div class="hdr">
  <div class="hdr-title">🧠 AI Business Analyst Copilot</div>
  <div class="hdr-sub">Large-Scale Edition — handles up to 500,000 rows via aggregation pipeline</div>
</div>
""", unsafe_allow_html=True)

# ── Welcome screen 
if S.df is None:
    st.markdown("""
    <div style="text-align:center;padding:50px 20px 30px;">
        <div style="font-size:60px;margin-bottom:16px;">🧠</div>
        <h2 style="font-size:26px;font-weight:700;">Upload your dataset to get started</h2>
        <p style="color:#6B7494;max-width:560px;margin:8px auto 32px;font-size:15px;">
            From raw CSV to actionable decisions in minutes.
            Handles up to 500,000 rows via intelligent aggregation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    cols  = st.columns(6)
    cards = [("📥","Ingest","CSV/Excel up to 500MB"),
             ("⚙️","Aggregate","500k→2k rows pipeline"),
             ("📊","EDA","Insight-focused charts"),
             ("🤖","AutoML","3 models + SHAP"),
             ("🎯","Decisions","Recs & risk alerts"),
             ("💬","RAG Chat","Grounded AI answers")]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div class="kpi-card" style="text-align:center;min-width:0;">
              <div style="font-size:24px;">{icon}</div>
              <div style="font-weight:600;font-size:13px;margin-top:6px;">{title}</div>
              <div style="font-size:11px;color:#6B7494;margin-top:3px;">{desc}</div>
            </div>""", unsafe_allow_html=True)
    st.info("👈 Upload a file or load an example dataset from the sidebar to begin.")
    st.stop()


# SECTION 14 — TABS

_pipeline_bar()
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Overview", "📊 EDA", "🤖 Modeling", "💬 Chatbot", "🎯 Decisions", "📄 Report"
])


# ── TAB 1: OVERVIEW 
with tab1:
    df = S.df
    st.markdown(f'<div class="ins c">🧠 <strong>Dataset Intelligence:</strong> {S.domain_summary}</div>',
                unsafe_allow_html=True)

    # Row 1 — raw stats
    c1,c2,c3,c4,c5,c6 = st.columns(6)
    raw_kpis = [
        (c1, f"{len(df):,}",   "Total Records",  ""),
        (c2, str(len(df.columns)), "Features",    ""),
        (c3, str(len(df.select_dtypes(include=np.number).columns)), "Numerical", ""),
        (c4, str(len(df.select_dtypes(include=["object","category"]).columns)), "Categorical",""),
        (c5, f"{int(df.duplicated().sum()):,}", "Duplicates",
             "r" if df.duplicated().sum() > 0 else "g"),
        (c6, f"{(df.isnull().sum()>0).sum()}", "Missing Cols",
             "y" if (df.isnull().sum()>0).sum() > 0 else "g"),
    ]
    for col, val, lbl, color in raw_kpis:
        with col:
            st.markdown(f"""
            <div class="kpi-card">
              <div class="kpi-val {color}">{val}</div>
              <div class="kpi-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    # Row 2 — business KPIs (after pipeline)
    if S.done["process"] and S.kpis:
        kpis = S.kpis
        st.markdown("---")
        st.markdown("#### 📈 Business KPIs")
        k1,k2,k3,k4 = st.columns(4)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val">{kpis.get("agg_rows",0):,}</div>'
                        f'<div class="kpi-lbl">Aggregated Rows</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card"><div class="kpi-val c">{kpis.get("reduction_pct",0):.1f}%</div>'
                        f'<div class="kpi-lbl">Data Compressed</div></div>', unsafe_allow_html=True)
        if "primary_total" in kpis:
            pt = kpis["primary_total"]
            fmt = f"{pt/1e6:.2f}M" if pt>1e6 else f"{pt/1e3:.1f}K" if pt>1e3 else f"{pt:,.1f}"
            with k3:
                st.markdown(f'<div class="kpi-card"><div class="kpi-val g">{fmt}</div>'
                            f'<div class="kpi-lbl">Total {kpis["primary_col"].title()}</div></div>',
                            unsafe_allow_html=True)
        if "yoy_growth_pct" in kpis:
            g = kpis["yoy_growth_pct"]
            with k4:
                st.markdown(f'<div class="kpi-card"><div class="kpi-val {"g" if g>0 else "r"}">{g:+.1f}%</div>'
                            f'<div class="kpi-lbl">YoY Growth</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    cl, cr = st.columns([2,1])
    with cl:
        st.markdown("#### 👁️ Data Preview")
        st.dataframe(df.head(10), use_container_width=True, height=280)
    with cr:
        st.markdown("#### 📋 Column Summary")
        st.dataframe(column_summary(df), use_container_width=True, height=280, hide_index=True)

    st.markdown("#### 📐 Descriptive Statistics")
    num_df = df.select_dtypes(include=np.number)
    if not num_df.empty:
        st.dataframe(num_df.describe().round(4), use_container_width=True)

    st.download_button("⬇️ Download Dataset (CSV)",
                       data=df.to_csv(index=False).encode(),
                       file_name=S.file_name or "dataset.csv", mime="text/csv")


# ── TAB 2: EDA 
with tab2:
    st.markdown("### 🔍 Exploratory Data Analysis")
    if not S.done["process"]:
        st.info("▶ Click **Run Full Pipeline** in the sidebar.")
        st.stop()

    agg_df = S.agg_df
    meta   = S.meta

    st.markdown("#### 🕳️ Missing Values")
    fig_m, ins_m = eda_missing(S.df)
    st.plotly_chart(fig_m, use_container_width=True)
    st.markdown(f'<div class="ins">{ins_m}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📈 Time-Series Trend")
    fig_t, ins_t = eda_trend(agg_df, meta)
    if fig_t:
        st.plotly_chart(fig_t, use_container_width=True)
        st.markdown(f'<div class="ins g">{ins_t}</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="ins y">⚠️ {ins_t}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏆 Top Segment Comparison")
    fig_top, ins_top = eda_top_n(agg_df, meta, n=10)
    st.plotly_chart(fig_top, use_container_width=True)
    st.markdown(f'<div class="ins">{ins_top}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 🏷️ Category Breakdowns")
    breakdowns = eda_category(agg_df, meta)
    if breakdowns:
        for fig_cat, ins_cat in breakdowns:
            st.plotly_chart(fig_cat, use_container_width=True)
            st.markdown(f'<div class="ins c">{ins_cat}</div>', unsafe_allow_html=True)
    else:
        st.info("No categorical group keys found for breakdown.")

    st.markdown("---")
    st.markdown("#### 🔗 Correlation Analysis")
    fig_corr, ins_corr = eda_correlation(agg_df, meta, S.target)
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown(f'<div class="ins y">{ins_corr}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("#### 📊 Distributions")
    fig_dist, ins_dist = eda_distributions(agg_df)
    st.plotly_chart(fig_dist, use_container_width=True)
    st.markdown(f'<div class="ins">{ins_dist}</div>', unsafe_allow_html=True)


# ── TAB 3: MODELING 
with tab3:
    st.markdown("### 🤖 Predictive Modeling Engine")
    if not S.done["process"]:
        st.info("▶ Run the pipeline first."); st.stop()
    if not S.target:
        st.warning("Select a target column in the sidebar."); st.stop()

    st.markdown(f'<div class="ins c">🎯 Target: <strong><code>{S.target}</code></strong> — '
                f'AutoML trains 3 models with 5-fold cross-validation and SHAP explainability.</div>',
                unsafe_allow_html=True)

    if not S.done["model"]:
        st.info("Click **🤖 Train ML Models** in the sidebar."); st.stop()

    mr   = S.model_results
    best = mr["best_name"]
    task = mr["task"]
    res  = mr["results"]

    pm = res[best].get("Accuracy (%)", res[best].get("R² (%)"))
    st.markdown(f'<div class="ins g">🏆 <strong>Best Model: {best}</strong> | '
                f'Task: {task.title()} | '
                f'{"Accuracy" if task=="classification" else "R²"}: {pm:.1f}%</div>',
                unsafe_allow_html=True)

    st.markdown("#### 📊 Model Comparison")
    for name, m in res.items():
        is_best    = name == best
        card_class = "model-card best" if is_best else "model-card"
        b_badge    = '<span class="badge badge-g">⭐ Best</span>' if is_best else ""
        of_badge   = (f'<span class="badge badge-y">⚠️ Overfit {m["Overfit Gap"]:.1f}%</span>'
                      if m["Overfit Gap"] > 15 else "")
        if task == "classification":
            metrics = (f"Acc: <b>{m['Accuracy (%)']}%</b> | F1: <b>{m['F1-Score (%)']}%</b> | "
                       f"CV: {m['CV Score (%)']}% ±{m['CV Std']}%")
        else:
            metrics = (f"R²: <b>{m['R² (%)']}%</b> | RMSE: <b>{m['RMSE']}</b> | "
                       f"CV: {m['CV R² (%)']}% ±{m['CV Std']}%")
        st.markdown(f"""
        <div class="{card_class}">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="font-weight:600;">{name}</span><span>{b_badge} {of_badge}</span>
          </div>
          <div style="font-size:13px;color:#8090B0;margin-top:5px;">{metrics}</div>
        </div>""", unsafe_allow_html=True)

    ca, cb = st.columns(2)
    with ca:
        st.plotly_chart(plot_model_comparison(res), use_container_width=True)
    with cb:
        st.plotly_chart(plot_overfitting(res), use_container_width=True)

    st.markdown("#### 🔑 Feature Importance")
    st.plotly_chart(plot_feature_importance(mr["importance_df"], mr.get("shap_df")),
                    use_container_width=True)
    st.markdown(f'<div class="ins">{model_insight(mr)}</div>', unsafe_allow_html=True)

    with st.expander("🔮 Prediction Sample (test set)"):
        st.dataframe(mr["pred_sample"].round(4), use_container_width=True, hide_index=True)


# ── TAB 4: CHATBOT 
with tab4:
    st.markdown("### 💬 AI Business Analyst Chatbot")
    bot = S.chatbot
    if not bot.is_ready:
        if S.done["process"]:
            with st.spinner("Building AI context..."):
                _rebuild_chatbot()
            st.rerun()
        else:
            st.info("Run the pipeline first."); st.stop()

    ctx_parts = [p for p in ["📊 EDA" if S.eda_ctx else None,
                              "🤖 Models" if S.model_results else None,
                              "🎯 Decisions" if S.decisions else None,
                              "⚙️ Aggregation"] if p]
    st.markdown(f'<div class="ins c">🧠 <strong>AI Context:</strong> {" + ".join(ctx_parts)} — '
                f'All answers grounded in your actual data.</div>', unsafe_allow_html=True)

    st.markdown("**💡 Quick Questions:**")
    sq = Chatbot.suggestions()
    sc1, sc2 = st.columns(2)
    for i, q in enumerate(sq[:6]):
        with (sc1 if i%2==0 else sc2):
            if st.button(q, key=f"sq_{i}", use_container_width=True):
                with st.spinner("Analyzing..."):
                    ans = bot.chat(q)
                S.chat_history.append(("user", q))
                S.chat_history.append(("bot", ans))
                st.rerun()

    st.markdown("---")
    if S.chat_history:
        st.markdown('<div style="max-height:500px;overflow-y:auto;">', unsafe_allow_html=True)
        for role, msg in S.chat_history:
            if role == "user":
                st.markdown(f'<div><div class="chat-lbl" style="text-align:right;">You</div>'
                            f'<div class="chat-user">{msg}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div><div class="chat-lbl">🧠 AI Analyst</div>'
                            f'<div class="chat-bot">{msg}</div></div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    ci, cs, cc = st.columns([5,1,1])
    with ci:
        question = st.text_input("Ask a question",
                                  placeholder="e.g. Which features drive predictions?",
                                  label_visibility="collapsed", key="q_input")
    with cs:
        send = st.button("Send", type="primary", use_container_width=True)
    with cc:
        if st.button("Clear", use_container_width=True):
            S.chat_history = []; bot.clear(); st.rerun()

    if (send or question) and question:
        with st.spinner("Analyzing..."):
            ans = bot.chat(question)
        S.chat_history.append(("user", question))
        S.chat_history.append(("bot", ans))
        st.rerun()


# ── TAB 5: DECISIONS 
with tab5:
    st.markdown("### 🎯 Decision Engine")
    if not S.done["decisions"] or not S.decisions:
        st.info("▶ Run the pipeline to generate business decisions."); st.stop()

    dec    = S.decisions
    recs   = dec.get("recommendations",[])
    risks  = dec.get("risks",[])
    optims = dec.get("optimizations",[])
    crit   = sum(1 for r in risks if r.get("severity") in ("Critical","High"))

    k1,k2,k3 = st.columns(3)
    with k1:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val g">{len(recs)}</div>'
                    f'<div class="kpi-lbl">Recommendations</div></div>', unsafe_allow_html=True)
    with k2:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val {"r" if crit else "g"}">{len(risks)}</div>'
                    f'<div class="kpi-lbl">Risk Alerts ({crit} High/Critical)</div></div>', unsafe_allow_html=True)
    with k3:
        st.markdown(f'<div class="kpi-card"><div class="kpi-val c">{len(optims)}</div>'
                    f'<div class="kpi-lbl">Optimizations</div></div>', unsafe_allow_html=True)

    if recs:
        st.markdown("---"); st.markdown("#### ✅ Business Recommendations")
        for rec in recs:
            p    = rec.get("priority","Medium")
            cls  = "hi" if p=="High" else "me"
            pb   = f'<span class="badge badge-g">{p}</span>' if p=="High" else f'<span class="badge badge-y">{p}</span>'
            tb   = f'<span class="badge badge-b">{rec.get("type","")}</span>'
            st.markdown(f'<div class="dec-card {cls}"><div class="dec-title">{rec["title"]} {pb} {tb}</div>'
                        f'<div class="dec-body">{rec["body"]}</div></div>', unsafe_allow_html=True)

    if risks:
        st.markdown("---"); st.markdown("#### ⚠️ Risk Alerts")
        for risk in risks:
            sv  = risk.get("severity","Medium")
            cls = "cr" if sv=="Critical" else "hi" if sv=="High" else "me"
            cb2 = "badge-r" if sv in ("Critical","High") else "badge-y"
            st.markdown(f'<div class="dec-card {cls}"><div class="dec-title">{risk["title"]} '
                        f'<span class="badge {cb2}">{sv}</span></div>'
                        f'<div class="dec-body">{risk["body"]}</div></div>', unsafe_allow_html=True)

    if optims:
        st.markdown("---"); st.markdown("#### 💡 Optimization Opportunities")
        for opt in optims:
            st.markdown(f'<div class="dec-card"><div class="dec-title">💡 {opt["title"]}</div>'
                        f'<div style="font-size:12px;color:{GREEN};margin:4px 0;">'
                        f'Expected Impact: {opt.get("expected_impact","TBD")}</div>'
                        f'<div class="dec-body">{opt["body"]}</div></div>', unsafe_allow_html=True)


# ── TAB 6: REPORT 
with tab6:
    st.markdown("### 📄 Business Report Generator")
    gen = st.button("📝 Generate Full Business Report", type="primary")

    if gen or S.report_text:
        if gen:
            with st.spinner("Generating report..."):
                if not S.done["process"]:
                    _do_process()
                if not S.decisions:
                    S.decisions = generate_decisions(S.df, S.agg_df, S.meta, S.kpis,
                                                      model_results=S.model_results,
                                                      domain=S.domain)
                S.report_text = generate_report(
                    df=S.df, agg_df=S.agg_df, meta=S.meta, kpis=S.kpis,
                    domain=S.domain, domain_summary=S.domain_summary,
                    model_results=S.model_results, decisions=S.decisions,
                    target_col=S.target,
                )
        st.markdown(S.report_text)
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("⬇️ Download (Markdown)", data=S.report_text.encode(),
                               file_name="business_report.md", mime="text/markdown",
                               use_container_width=True)
        with c2:
            plain = re.sub(r"[*`#]", "", S.report_text)
            st.download_button("⬇️ Download (Plain Text)", data=plain.encode(),
                               file_name="business_report.txt", mime="text/plain",
                               use_container_width=True)
    else:
        st.markdown("""
        <div class="ins">
          📄 Click <strong>Generate Full Business Report</strong> to produce:<br><br>
          • Executive summary with KPIs<br>
          • Dataset overview and data quality assessment<br>
          • Key EDA insights with specific numbers<br>
          • ML model comparison table<br>
          • Data-grounded business recommendations<br>
          • Risk analysis and optimization opportunities<br>
          • 5-step next-steps roadmap
        </div>
        """, unsafe_allow_html=True)