from __future__ import annotations
from typing import Optional

import numpy as np
import pandas as pd


# ── main entry 
def generate_decisions(
    df_raw: pd.DataFrame,
    agg_df: pd.DataFrame,
    meta: dict,
    kpis: dict,
    model_results: Optional[dict] = None,
    domain: str = "general analytics",
) -> dict:
    """
    Returns a dict with:
      recommendations: list of {title, body, priority, type}
      risks: list of {title, body, severity}
      optimizations: list of {title, body, expected_impact}
    """
    recs   = []
    risks  = []
    optims = []

    # ── 1. Data quality decisions 
    miss = (df_raw.isnull().sum() / len(df_raw) * 100)
    high_miss = miss[miss > 20]
    if not high_miss.empty:
        worst_col = high_miss.idxmax()
        risks.append({
            "title":    f"Data Gap in `{worst_col}` ({high_miss.max():.1f}% missing)",
            "body":     f"Column `{worst_col}` is missing {high_miss.max():.1f}% of values. "
                        f"This reduces model reliability for predictions involving this feature. "
                        f"**Action:** Implement mandatory data capture in source systems within 30 days.",
            "severity": "High" if high_miss.max() > 30 else "Medium",
        })

    dup_count = int(df_raw.duplicated().sum())
    if dup_count > 0:
        dup_pct = dup_count / len(df_raw) * 100
        risks.append({
            "title":    f"Duplicate Records: {dup_count:,} rows ({dup_pct:.1f}%)",
            "body":     f"{dup_count:,} duplicate records detected ({dup_pct:.1f}% of data). "
                        f"Duplicates inflate metrics and bias models. "
                        f"**Action:** Audit upstream data pipeline for deduplication logic.",
            "severity": "High" if dup_pct > 5 else "Medium",
        })
    else:
        optims.append({
            "title":    "Dataset Integrity Confirmed",
            "body":     "No duplicate records found. Data pipeline is clean.",
            "expected_impact": "Low — maintaining quality baseline.",
        })

    # ── 2. Growth / trend decisions 
    if "yoy_growth_pct" in kpis:
        growth = kpis["yoy_growth_pct"]
        primary = kpis.get("primary_col", "primary metric")
        latest_y = kpis.get("latest_year", "latest period")
        prev_y   = kpis.get("previous_year", "prior period")

        if growth < -10:
            risks.append({
                "title":    f"Significant Revenue Decline: {growth:+.1f}% YoY",
                "body":     f"`{primary}` declined {abs(growth):.1f}% from {prev_y} to {latest_y} "
                            f"(from {kpis.get('prev_year_val',0):,.0f} to {kpis.get('latest_year_val',0):,.0f}). "
                            f"**Immediate actions required:** Investigate root cause, review pricing strategy, "
                            f"and assess competitive landscape.",
                "severity": "Critical",
            })
        elif growth < 0:
            risks.append({
                "title":    f"Moderate Decline in {primary}: {growth:+.1f}% YoY",
                "body":     f"`{primary}` contracted {abs(growth):.1f}% in {latest_y}. "
                            f"Early-stage decline — monitor closely and identify affected segments.",
                "severity": "Medium",
            })
        elif growth > 20:
            recs.append({
                "title":    f"Strong Growth: Capitalize on {growth:+.1f}% YoY Momentum",
                "body":     f"`{primary}` grew {growth:.1f}% to {kpis.get('latest_year_val',0):,.0f} in {latest_y}. "
                            f"**Recommendations:** Reinvest in top-performing segments, scale successful campaigns, "
                            f"and prepare capacity for continued growth.",
                "priority": "High",
                "type":     "Growth Opportunity",
            })
        else:
            recs.append({
                "title":    f"Steady Performance: {growth:+.1f}% YoY Growth",
                "body":     f"`{primary}` shows stable growth of {growth:.1f}% in {latest_y}. "
                            f"**Recommendation:** Focus on efficiency improvements and market share expansion.",
                "priority": "Medium",
                "type":     "Maintenance",
            })

    # ── 3. Segment concentration risk 
    if "top_category_share" in kpis:
        share = kpis["top_category_share"]
        cat   = kpis["top_category_col"]
        name  = kpis["top_category_name"]
        val   = kpis["top_category_value"]
        primary = kpis.get("primary_col","metric")

        if share > 60:
            risks.append({
                "title":    f"High Concentration Risk: '{name}' = {share:.1f}% of {primary}",
                "body":     f"A single {cat} (`{name}`) contributes {share:.1f}% of total {primary} ({val:,.0f}). "
                            f"Over-reliance on one segment creates business risk. "
                            f"**Action:** Diversify revenue streams; invest in the bottom 20% of {cat} segments.",
                "severity": "High" if share > 70 else "Medium",
            })
        else:
            recs.append({
                "title":    f"Leverage `{name}` — Top Segment at {share:.1f}%",
                "body":     f"'{name}' leads in {primary} with {share:.1f}% share ({val:,.0f}). "
                            f"**Recommendation:** Replicate the strategies of the top segment in underperformers.",
                "priority": "High",
                "type":     "Segment Strategy",
            })

    # ── 4. Performance gap optimization 
    value_cols = meta.get("value_cols", [])
    cat_keys   = [k for k in meta.get("group_keys", []) if not k.startswith("_")]

    if cat_keys and value_cols:
        key = cat_keys[0]
        vc  = value_cols[0]
        sc  = f"{vc}_sum"
        if key in agg_df.columns and sc in agg_df.columns:
            grouped = agg_df.groupby(key)[sc].sum().sort_values()
            if len(grouped) >= 3:
                bottom3 = grouped.head(3)
                top1    = grouped.iloc[-1]
                gap_pct = (top1 - bottom3.mean()) / bottom3.mean() * 100 if bottom3.mean() > 0 else 0
                optims.append({
                    "title":    f"Close Performance Gap in Bottom {key} Segments",
                    "body":     f"Underperforming {key} segments — {', '.join(f'`{c}`' for c in bottom3.index)} — "
                                f"average {bottom3.mean():,.0f} {vc} vs top performer at {top1:,.0f} ({gap_pct:,.0f}% gap). "
                                f"**Optimization:** A 10% improvement in bottom segments = "
                                f"{bottom3.sum() * 0.1:,.0f} additional {vc}. "
                                f"Prioritize targeted campaigns and resource allocation for these segments.",
                    "expected_impact": f"~{bottom3.sum()*0.1:,.0f} incremental {vc} (+{bottom3.sum()*0.1/grouped.sum()*100:.1f}%)",
                })

    # ── 5. Model-driven decisions 
    if model_results:
        best     = model_results["best_name"]
        results  = model_results["results"]
        task     = model_results["task"]
        imp      = model_results["importance_df"]
        gap      = results[best]["Overfit Gap"]
        primary_metric = results[best].get("Accuracy (%)", results[best].get("R² (%)"))

        if gap > 15:
            risks.append({
                "title":    f"Model Overfitting Alert — {best} (Gap: {gap:.1f}%)",
                "body":     f"{best} shows a {gap:.1f}% train-test performance gap. "
                            f"This means the model may underperform on new data. "
                            f"**Action:** Add regularization, reduce model complexity, or collect more training data.",
                "severity": "Medium",
            })

        top_feature = imp.iloc[0]["Feature"]
        top_pct     = imp.iloc[0]["Importance"] / imp["Importance"].sum() * 100

        recs.append({
            "title":    f"Prioritize `{top_feature}` — Drives {top_pct:.1f}% of Predictions",
            "body":     f"Feature `{top_feature}` is the single most important predictor, "
                        f"driving {top_pct:.1f}% of {best}'s decision logic. "
                        f"**Business implication:** Changes to `{top_feature}` will have disproportionate "
                        f"impact on outcomes. Monitoring and optimizing this variable is the highest-ROI action.",
            "priority": "High",
            "type":     "Model-Driven Insight",
        })

        if primary_metric < 70 and task == "classification":
            recs.append({
                "title":    "Improve Model Accuracy with More Features",
                "body":     f"Current best accuracy is {primary_metric:.1f}% — below the 80% reliability threshold. "
                            f"**Recommendations:** Collect additional features, apply feature engineering, "
                            f"or gather more labeled training data to improve predictive reliability.",
                "priority": "Medium",
                "type":     "Model Improvement",
            })
        elif primary_metric >= 80:
            recs.append({
                "title":    f"Deploy {best} — {primary_metric:.1f}% Performance Validated",
                "body":     f"{best} achieves {primary_metric:.1f}% performance with healthy generalization. "
                            f"**Recommendation:** Proceed to production deployment. Set up monitoring pipelines "
                            f"to detect data drift and retrain when performance drops >5%.",
                "priority": "High",
                "type":     "Deployment Ready",
            })

    # ── 6. Dataset size recommendation 
    n = meta["original_rows"]
    if n < 1000:
        risks.append({
            "title":    f"Small Dataset ({n:,} rows) — Model Reliability Limited",
            "body":     f"With only {n:,} records, model performance estimates have high variance. "
                        f"**Action:** Collect at least 5,000 records for robust ML. "
                        f"Current results should be treated as directional, not definitive.",
            "severity": "Medium",
        })

    return {
        "recommendations": recs,
        "risks":           risks,
        "optimizations":   optims,
    }


def build_decision_context(decisions: dict, kpis: dict) -> str:
    """Serialize decisions to text for RAG context."""
    lines = []

    recs  = decisions.get("recommendations", [])
    risks = decisions.get("risks", [])
    optims = decisions.get("optimizations", [])

    if "yoy_growth_pct" in kpis:
        g = kpis["yoy_growth_pct"]
        lines.append(
            f"Year-over-year performance: {g:+.1f}% change in '{kpis.get('primary_col','metric')}' "
            f"from {kpis.get('previous_year','prior')} to {kpis.get('latest_year','latest')}."
        )

    if "top_category_name" in kpis:
        lines.append(
            f"Top segment: '{kpis['top_category_name']}' in '{kpis['top_category_col']}' "
            f"= {kpis['top_category_share']:.1f}% of total {kpis.get('primary_col','metric')}."
        )

    lines.append(f"\nBusiness Recommendations ({len(recs)}):")
    for r in recs:
        lines.append(f"  [{r.get('priority','Med')}] {r['title']}: {r['body'][:200]}")

    lines.append(f"\nRisk Alerts ({len(risks)}):")
    for r in risks:
        lines.append(f"  [{r.get('severity','Med')}] {r['title']}: {r['body'][:200]}")

    lines.append(f"\nOptimization Opportunities ({len(optims)}):")
    for o in optims:
        lines.append(f"  {o['title']}: Expected impact = {o.get('expected_impact','TBD')}. {o['body'][:150]}")

    return "\n".join(lines)



if __name__ == "__main__":
    import pandas as pd

    # 🔹 Dummy dataset
    df = pd.DataFrame({
        "sales": [100, 200, 150, 300, 250],
        "region": ["A", "B", "A", "B", "C"]
    })

    # 🔹 Aggregation
    agg_df = df.groupby("region").agg({"sales": "sum"}).reset_index()

    # 🔹 Meta + KPI
    meta = {
        "original_rows": len(df),
        "group_keys": ["region"],
        "value_cols": ["sales"]
    }

    kpis = {
        "primary_col": "sales"
    }

    # 🔹 Call function
    results = generate_decisions(df, agg_df, meta, kpis)

    # 🔥 PRINT OUTPUT
    print("\n===== RECOMMENDATIONS =====")
    for r in results["recommendations"]:
        print(f"- {r['title']}")
        print(f"  {r['body']}\n")

    print("\n===== RISKS =====")
    for r in results["risks"]:
        print(f"- {r['title']}")
        print(f"  {r['body']}\n")

    print("\n===== OPTIMIZATIONS =====")
    for o in results["optimizations"]:
        print(f"- {o['title']}")
        print(f"  {o['body']}\n")