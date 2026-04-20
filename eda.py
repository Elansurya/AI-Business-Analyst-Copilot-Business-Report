from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── theme 
BG   = "#0A0C14"
CARD = "#13172A"
TEXT = "#E2E8F0"
MUTED = "#6B7494"
ACCENT = "#6C63FF"
GREEN  = "#10B981"
RED    = "#EF4444"
YELLOW = "#F59E0B"
PALETTE = ["#6C63FF","#10B981","#F59E0B","#EF4444","#06B6D4",
           "#8B5CF6","#F97316","#EC4899","#14B8A6","#84CC16"]

def _layout(fig, title="", height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=15)),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT, family="Inter, sans-serif"),
        height=height,
        margin=dict(t=55, b=45, l=15, r=15),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED), linecolor="#1E2340")
    fig.update_yaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED), linecolor="#1E2340")
    return fig


# ── 1. Missing value heatmap 
def missing_heatmap(df: pd.DataFrame) -> tuple[go.Figure, str]:
    missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = missing[missing > 0]

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ No missing values", showarrow=False,
                           font=dict(color=GREEN, size=15))
        _layout(fig, "Missing Values", 180)
        return fig, "Dataset is complete — no missing values detected. This is ideal for modeling."

    colors = [RED if v > 30 else YELLOW if v > 10 else GREEN for v in missing.values]
    fig = go.Figure(go.Bar(
        x=missing.index, y=missing.values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in missing.values], textposition="outside",
    ))
    _layout(fig, "Missing Value Analysis (%)", 360)

    high = missing[missing > 30]
    med  = missing[(missing > 10) & (missing <= 30)]
    low  = missing[missing <= 10]
    parts = []
    if not high.empty:
        parts.append(f"**Critical gaps** in {', '.join(f'`{c}`' for c in high.index)} (>30%) — "
                     "these columns will be dropped during preprocessing.")
    if not med.empty:
        parts.append(f"**Moderate gaps** in {', '.join(f'`{c}`' for c in med.index)} — "
                     "median/mode imputation recommended.")
    if not low.empty:
        parts.append(f"**Minor gaps** in {', '.join(f'`{c}`' for c in low.index)} — safe to impute.")
    return fig, " ".join(parts)


# ── 2. Time-series trend 
def trend_chart(agg_df: pd.DataFrame, meta: dict) -> tuple[Optional[go.Figure], str]:
    """Plot a time-series trend if time columns exist in aggregated data."""
    time_cols = [c for c in ["_year","_quarter","_month"] if c in agg_df.columns]
    value_cols = [f"{v}_sum" for v in meta.get("value_cols", [])[:2]
                  if f"{v}_sum" in agg_df.columns]

    if not time_cols or not value_cols:
        return None, "No time dimension detected in this dataset."

    time_col = time_cols[0]
    fig = go.Figure()

    for i, vc in enumerate(value_cols):
        ts = agg_df.groupby(time_col)[vc].sum().reset_index()
        ts.columns = ["period", "value"]
        ts = ts.sort_values("period")

        fig.add_trace(go.Scatter(
            x=ts["period"].astype(str), y=ts["value"],
            mode="lines+markers",
            name=vc.replace("_sum",""),
            line=dict(color=PALETTE[i], width=2.5),
            marker=dict(size=7, color=PALETTE[i]),
            fill="tozeroy" if i == 0 else None,
            fillcolor=f"rgba(108,99,255,0.07)" if i == 0 else None,
        ))

    _layout(fig, f"Trend over {time_col.replace('_','').title()}", 420)

    # Compute trend direction
    primary_vc = value_cols[0]
    ts = agg_df.groupby(time_col)[primary_vc].sum().sort_index()
    if len(ts) >= 2:
        pct_change = (ts.iloc[-1] - ts.iloc[-2]) / ts.iloc[-2] * 100 if ts.iloc[-2] != 0 else 0
        direction = "up" if pct_change > 0 else "down"
        emoji = "📈" if pct_change > 0 else "📉"
        metric_name = primary_vc.replace("_sum", "")
        insight = (
            f"{emoji} **`{metric_name}` trended {direction}** in the latest period "
            f"({pct_change:+.1f}% vs prior). "
            f"Total across all periods: {ts.sum():,.0f}. "
            f"{'Growth trajectory is positive — maintain current strategy.' if pct_change > 5 else 'Decline warrants investigation of contributing factors.' if pct_change < -5 else 'Performance is relatively stable period-over-period.'}"
        )
    else:
        insight = "Insufficient time periods to compute trend direction."

    return fig, insight


# ── 3. Category breakdown 
def category_breakdown(agg_df: pd.DataFrame, meta: dict) -> list[tuple[go.Figure, str]]:
    """One chart per categorical group key, showing value distribution."""
    cat_keys = [k for k in meta.get("group_keys", []) if not k.startswith("_")]
    value_cols = [f"{v}_sum" for v in meta.get("value_cols", [])[:1]
                  if f"{v}_sum" in agg_df.columns]

    if not cat_keys or not value_cols:
        return []

    results = []
    vc = value_cols[0]
    metric_name = vc.replace("_sum", "")

    for key in cat_keys[:3]:
        if key not in agg_df.columns:
            continue
        grouped = agg_df.groupby(key)[vc].sum().sort_values(ascending=False).head(15)
        total = grouped.sum()
        top_share = grouped.iloc[0] / total * 100 if total > 0 else 0
        top2_share = grouped.iloc[:2].sum() / total * 100 if total > 0 else 0

        colors = [PALETTE[0] if i == 0 else PALETTE[1] if i == 1 else MUTED
                  for i in range(len(grouped))]

        fig = go.Figure(go.Bar(
            x=grouped.index.astype(str), y=grouped.values,
            marker_color=colors,
            text=[f"{v/total*100:.1f}%" for v in grouped.values],
            textposition="outside",
        ))
        _layout(fig, f"{metric_name.title()} by {key}", 380)

        insight = (
            f"**`{key}` breakdown:** Top segment '{grouped.index[0]}' "
            f"accounts for {top_share:.1f}% of total {metric_name}. "
            f"Top 2 combined: {top2_share:.1f}%. "
            f"{'High concentration — diversification risk if top segment underperforms.' if top_share > 50 else 'Relatively balanced distribution across segments.'}"
        )
        results.append((fig, insight))

    return results


# ── 4. Correlation heatmap 
def correlation_heatmap(agg_df: pd.DataFrame, meta: dict,
                        target_col: Optional[str] = None) -> tuple[go.Figure, str]:
    # Use aggregated sum/mean columns
    agg_num = agg_df.select_dtypes(include=np.number)
    # filter to _sum and _mean cols (most meaningful)
    useful = [c for c in agg_num.columns if c.endswith(("_sum","_mean","_count"))]
    if not useful:
        useful = agg_num.columns.tolist()
    if len(useful) < 2:
        fig = go.Figure()
        _layout(fig, "Correlation Matrix", 200)
        return fig, "Insufficient numerical columns for correlation analysis."

    sub = agg_num[useful[:12]].dropna(axis=1, how="all")
    corr = sub.corr().round(3)

    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr.values, texttemplate="%{text:.2f}",
        colorbar=dict(title="r", tickfont=dict(color=TEXT)),
    ))
    _layout(fig, "Correlation Matrix (Aggregated Features)", 500)

    # Find strongest pair
    pairs = []
    cols = list(corr.columns)
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            pairs.append((cols[i], cols[j], corr.iloc[i,j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    insight_parts = []
    if pairs:
        c1, c2, val = pairs[0]
        direction = "positive" if val > 0 else "negative"
        insight_parts.append(
            f"**Strongest correlation:** `{c1}` ↔ `{c2}` (r={val:.3f}, {direction}). "
            f"{'These move together — optimizing one may improve the other.' if val > 0 else 'Inverse relationship — trade-offs exist between these metrics.'}"
        )

    target_clean = target_col.replace(" ","_") if target_col else None
    for suffix in ("_sum","_mean",""):
        candidate = f"{target_clean}{suffix}" if target_clean else None
        if candidate and candidate in corr.columns:
            top_preds = corr[candidate].drop(candidate).abs().sort_values(ascending=False).head(3)
            insight_parts.append(
                f"**Top predictors of `{target_col}`:** "
                + ", ".join(f"`{c}` (r={corr[candidate][c]:.2f})" for c in top_preds.index)
            )
            break

    return fig, " | ".join(insight_parts) if insight_parts else "No strong correlations detected."


# ── 5. Distribution overview 
def distribution_overview(agg_df: pd.DataFrame, meta: dict) -> tuple[go.Figure, str]:
    """Histogram grid of aggregated _sum and _mean columns."""
    cols_to_plot = [c for c in agg_df.columns
                    if c.endswith(("_sum","_mean")) and agg_df[c].notna().sum() > 5][:6]

    if not cols_to_plot:
        cols_to_plot = agg_df.select_dtypes(include=np.number).columns.tolist()[:6]

    if not cols_to_plot:
        fig = go.Figure()
        _layout(fig,"",180)
        return fig, "No numerical columns available for distribution analysis."

    n = len(cols_to_plot)
    rows = (n + 1) // 2
    fig = make_subplots(rows=rows, cols=2,
                        subplot_titles=[c.replace("_"," ").title() for c in cols_to_plot],
                        vertical_spacing=0.14)

    skew_info = {}
    for idx, col in enumerate(cols_to_plot):
        r, c = divmod(idx, 2)
        s = agg_df[col].dropna()
        skew_info[col] = s.skew()
        fig.add_trace(go.Histogram(
            x=s, name=col, nbinsx=25,
            marker_color=PALETTE[idx % len(PALETTE)],
            opacity=0.85, showlegend=False,
        ), row=r+1, col=c+1)

    _layout(fig, "Aggregated Metric Distributions", 270*rows)

    high_skew = {k: v for k, v in skew_info.items() if abs(v) > 1.5}
    insight = (
        f"**Distributions of aggregated metrics.** "
        + (f"Highly skewed: {', '.join(f'`{k}` (skew={v:.2f})' for k,v in list(high_skew.items())[:3])}. "
           "Log-transform may improve model performance." if high_skew
           else "All metrics show moderate skewness — standard scaling is sufficient.")
    )
    return fig, insight


# ── 6. Top-N comparison bar 
def top_n_comparison(agg_df: pd.DataFrame, meta: dict, n: int = 10) -> tuple[go.Figure, str]:
    """Horizontal bar: top N entities by primary metric."""
    cat_keys = [k for k in meta.get("group_keys",[]) if not k.startswith("_")]
    value_cols = [f"{v}_sum" for v in meta.get("value_cols",[])[:1]
                  if f"{v}_sum" in agg_df.columns]

    if not cat_keys or not value_cols:
        fig = go.Figure(); _layout(fig,"",180)
        return fig, "No categorical grouping available for comparison."

    key = cat_keys[0]
    vc  = value_cols[0]
    metric_name = vc.replace("_sum","")

    grouped = (agg_df.groupby(key)[vc].sum()
               .sort_values(ascending=True).tail(n))
    total = grouped.sum()

    colors = [ACCENT if i == len(grouped)-1 else
              GREEN  if i == len(grouped)-2 else MUTED
              for i in range(len(grouped))]

    fig = go.Figure(go.Bar(
        y=grouped.index.astype(str), x=grouped.values,
        orientation="h",
        marker_color=colors,
        text=[f"{v/total*100:.1f}%" for v in grouped.values],
        textposition="outside",
    ))
    _layout(fig, f"Top {n} {key} by {metric_name.title()}", 60 + len(grouped)*38)

    bottom = grouped.index[0]
    top    = grouped.index[-1]
    gap    = (grouped.iloc[-1] - grouped.iloc[0]) / grouped.iloc[0] * 100 if grouped.iloc[0] > 0 else 0
    insight = (
        f"**Performance gap:** '{top}' outperforms '{bottom}' by {gap:,.0f}% in {metric_name}. "
        f"Bottom performers represent an optimization opportunity — "
        f"targeted intervention could close {(grouped.iloc[-1]-grouped.iloc[0]):,.0f} units of gap."
    )
    return fig, insight


# ── build EDA text context for RAG 
def build_eda_context(df_raw: pd.DataFrame, agg_df: pd.DataFrame,
                      meta: dict, target_col: Optional[str] = None) -> str:
    lines = []

    # Raw-level summary
    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=["object","category"]).columns.tolist()
    miss = (df_raw.isnull().sum() / len(df_raw) * 100)
    high_miss = miss[miss > 10]

    lines.append(f"Dataset has {len(df_raw):,} rows and {len(df_raw.columns)} columns.")
    lines.append(f"Numerical columns ({len(num_cols)}): {', '.join(num_cols[:10])}.")
    lines.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:10])}.")

    if not high_miss.empty:
        for col, pct in high_miss.items():
            lines.append(f"Column '{col}' has {pct:.1f}% missing values.")
    else:
        lines.append("No significant missing values in the dataset.")

    # Descriptive stats from raw
    if num_cols:
        desc = df_raw[num_cols[:8]].describe().round(3)
        for col in desc.columns:
            lines.append(
                f"'{col}': mean={desc[col]['mean']}, std={desc[col]['std']}, "
                f"min={desc[col]['min']}, max={desc[col]['max']}, "
                f"skew={round(df_raw[col].skew(), 3)}."
            )

    # Correlations from raw
    if len(num_cols) > 1:
        corr = df_raw[num_cols[:10]].corr()
        for i, c1 in enumerate(corr.columns):
            for j, c2 in enumerate(corr.columns):
                if i < j and abs(corr.iloc[i,j]) > 0.5:
                    lines.append(
                        f"'{c1}' and '{c2}' have correlation {corr.iloc[i,j]:.3f}."
                    )

    # Aggregated context
    lines.append(f"\nAggregated data ({len(agg_df)} rows after groupby {meta.get('group_keys','')}):")
    for key in meta.get("group_keys",[])[:2]:
        if key.startswith("_") or key not in agg_df.columns:
            continue
        for vc in meta.get("value_cols",[])[:1]:
            sc = f"{vc}_sum"
            if sc not in agg_df.columns:
                continue
            top = agg_df.groupby(key)[sc].sum().sort_values(ascending=False).head(5)
            total = top.sum()
            lines.append(f"'{key}' top groups by '{vc}': " +
                         "; ".join(f"'{k}'={v:,.1f}({v/total*100:.1f}%)" for k,v in top.items()) + ".")

    # Target
    if target_col and target_col in df_raw.columns:
        s = df_raw[target_col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            lines.append(f"Target '{target_col}': mean={s.mean():.3f}, "
                         f"std={s.std():.3f}, skew={s.skew():.3f}, task=regression.")
        else:
            vc = s.value_counts()
            lines.append(f"Target '{target_col}': {vc.nunique()} classes, "
                         f"dist={dict(vc.head(4))}, task=classification.")

    return "\n".join(lines)