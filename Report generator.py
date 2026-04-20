from __future__ import annotations
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


def generate_report(
    df_raw: pd.DataFrame,
    agg_df: pd.DataFrame,
    meta: dict,
    kpis: dict,
    domain: str,
    domain_summary: str,
    model_results: Optional[dict],
    decisions: dict,
    target_col: Optional[str],
) -> str:

    now = datetime.now().strftime("%B %d, %Y — %H:%M")
    lines = [
        "# 📊 AI Business Analyst Copilot — Business Report",
        f"*Generated: {now}*",
        "",
        "---",
        "",
        "## 1. Executive Summary",
        "",
        f"> {domain_summary}",
        "",
    ]

    # KPI highlights
    if "primary_col" in kpis:
        lines += [
            "**Key Metrics at a Glance:**",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Dataset Size | {meta['original_rows']:,} rows × {meta['original_cols']} columns |",
            f"| Data Reduced To | {meta['agg_rows']:,} aggregated rows ({kpis['reduction_pct']:.1f}% compression) |",
            f"| Primary Metric (`{kpis['primary_col']}`) Total | {kpis['primary_total']:,.2f} |",
        ]
        if "yoy_growth_pct" in kpis:
            direction = "↑" if kpis["yoy_growth_pct"] > 0 else "↓"
            lines.append(f"| YoY Growth | {direction} {abs(kpis['yoy_growth_pct']):.1f}% |")
        if "top_category_name" in kpis:
            lines.append(f"| Top {kpis['top_category_col']} | {kpis['top_category_name']} ({kpis['top_category_share']:.1f}% share) |")
        lines.append("")

    # ── 2. Dataset Overview 
    lines += [
        "## 2. Dataset Overview",
        "",
        f"**Domain:** {domain.title()}",
        f"**Source summary:** {domain_summary}",
        "",
    ]

    num_cols = df_raw.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df_raw.select_dtypes(include=["object","category"]).columns.tolist()
    miss = (df_raw.isnull().sum() / len(df_raw) * 100)
    high_miss = miss[miss > 5]
    dup_count = int(df_raw.duplicated().sum())

    lines += [
        f"- **Total records:** {len(df_raw):,}",
        f"- **Columns:** {len(df_raw.columns)} ({len(num_cols)} numerical, {len(cat_cols)} categorical)",
        f"- **Duplicate rows:** {dup_count:,} ({dup_count/len(df_raw)*100:.1f}%)",
        f"- **Columns with >5% missing:** {len(high_miss)}",
        f"- **Group-by keys used:** {', '.join(meta.get('group_keys', [])) or 'None (global aggregation)'}",
        "",
    ]

    # Data quality detail
    if not high_miss.empty:
        lines += ["**Data Quality Issues:**", ""]
        for col, pct in high_miss.sort_values(ascending=False).items():
            badge = "🔴" if pct > 30 else "🟡" if pct > 10 else "🟢"
            lines.append(f"- {badge} `{col}`: {pct:.1f}% missing")
        lines.append("")
    else:
        lines += ["✅ **No significant missing values** — dataset is complete.\n"]

    # ── 3. EDA Insights 
    lines += [
        "## 3. Key EDA Insights",
        "",
    ]

    if "yoy_growth_pct" in kpis:
        g = kpis["yoy_growth_pct"]
        emoji = "📈" if g > 0 else "📉"
        lines.append(
            f"{emoji} **Year-over-Year:** `{kpis['primary_col']}` "
            f"{'grew' if g > 0 else 'declined'} {abs(g):.1f}% from "
            f"{kpis['previous_year']} ({kpis['prev_year_val']:,.0f}) to "
            f"{kpis['latest_year']} ({kpis['latest_year_val']:,.0f})."
        )
        lines.append("")

    if "top_category_name" in kpis:
        lines.append(
            f"**Segment Concentration:** `{kpis['top_category_name']}` is the top `{kpis['top_category_col']}` "
            f"segment at {kpis['top_category_share']:.1f}% of total {kpis['primary_col']}."
        )
        lines.append("")

    # Correlations
    if len(num_cols) > 1:
        corr = df_raw[num_cols[:10]].corr()
        strong = []
        for i in range(len(num_cols[:10])):
            for j in range(i+1, len(num_cols[:10])):
                v = corr.iloc[i,j]
                if abs(v) >= 0.6:
                    strong.append((num_cols[i], num_cols[j], v))
        if strong:
            lines += ["**Strong Correlations:**", ""]
            for c1, c2, v in sorted(strong, key=lambda x: abs(x[2]), reverse=True)[:5]:
                d = "positive" if v > 0 else "negative"
                lines.append(f"- `{c1}` ↔ `{c2}`: r={v:.3f} ({d})")
            lines.append("")

    # ── 4. Target Variable 
    if target_col and target_col in df_raw.columns:
        lines += [f"## 4. Target Variable Analysis: `{target_col}`", ""]
        s = df_raw[target_col].dropna()
        if pd.api.types.is_numeric_dtype(s):
            lines += [
                f"- **Task type:** Regression",
                f"- **Mean:** {s.mean():.4f}  |  **Std Dev:** {s.std():.4f}",
                f"- **Range:** {s.min():.4f} → {s.max():.4f}",
                f"- **Skewness:** {s.skew():.3f} {'(⚠️ log-transform recommended)' if abs(s.skew()) > 1 else '(✅ near-normal)'}",
                "",
            ]
        else:
            vc = s.value_counts()
            lines += [
                f"- **Task type:** Classification ({vc.nunique()} classes)",
                f"- **Class distribution:**",
            ]
            for cls, cnt in vc.items():
                lines.append(f"  - `{cls}`: {cnt:,} ({cnt/len(s)*100:.1f}%)")
            imbalanced = vc.iloc[0] / len(s) > 0.70
            if imbalanced:
                lines.append("\n⚠️ **Class imbalance** — use `class_weight='balanced'` or SMOTE.")
            lines.append("")

    # ── 5. Model Performance 
    if model_results:
        lines += ["## 5. Machine Learning Model Performance", ""]
        results  = model_results["results"]
        best     = model_results["best_name"]
        task     = model_results["task"]
        imp      = model_results["importance_df"]

        lines.append(f"**🏆 Best Model: {best}**")
        lines.append("")

        if task == "classification":
            lines += ["| Model | Accuracy (%) | F1-Score (%) | CV Score (%) | Overfit Gap |",
                      "|-------|-------------|-------------|-------------|-------------|"]
            for name, m in results.items():
                star = " ⭐" if name == best else ""
                gap_flag = " ⚠️" if m["Overfit Gap"] > 15 else ""
                lines.append(f"| {name}{star} | {m['Accuracy (%)']} | {m['F1-Score (%)']} | "
                              f"{m['CV Score (%)']} ± {m['CV Std']} | {m['Overfit Gap']}{gap_flag} |")
        else:
            lines += ["| Model | R² (%) | RMSE | CV R² (%) | Overfit Gap |",
                      "|-------|--------|------|----------|-------------|"]
            for name, m in results.items():
                star = " ⭐" if name == best else ""
                gap_flag = " ⚠️" if m["Overfit Gap"] > 20 else ""
                lines.append(f"| {name}{star} | {m['R² (%)']} | {m['RMSE']} | "
                              f"{m['CV R² (%)']} ± {m['CV Std']} | {m['Overfit Gap']}{gap_flag} |")

        lines.append("")
        lines.append("**Top 5 Feature Importances:**")
        lines.append("")
        total_imp = imp["Importance"].sum()
        for i, (_, row) in enumerate(imp.head(5).iterrows()):
            pct = row["Importance"] / total_imp * 100
            lines.append(f"{i+1}. `{row['Feature']}` — {pct:.1f}% of predictive importance")
        lines.append("")

    # ── 6. Business Recommendations 
    lines += ["## 6. Business Recommendations", ""]
    for i, rec in enumerate(decisions.get("recommendations", []), 1):
        priority = rec.get("priority","Med")
        badge = "🟢" if priority == "High" else "🟡" if priority == "Medium" else "⚪"
        lines += [
            f"### {badge} {i}. {rec['title']}",
            f"**Priority:** {priority} | **Type:** {rec.get('type','')}",
            "",
            rec["body"],
            "",
        ]

    # ── 7. Risk Analysis 
    lines += ["## 7. Risk Analysis", ""]
    for risk in decisions.get("risks", []):
        sev = risk.get("severity","Medium")
        badge = "🔴" if sev == "Critical" else "🟠" if sev == "High" else "🟡"
        lines += [
            f"### {badge} {risk['title']}",
            f"**Severity:** {sev}",
            "",
            risk["body"],
            "",
        ]

    # ── 8. Optimization Opportunities 
    if decisions.get("optimizations"):
        lines += ["## 8. Optimization Opportunities", ""]
        for opt in decisions["optimizations"]:
            lines += [
                f"### 💡 {opt['title']}",
                f"**Expected Impact:** {opt.get('expected_impact','TBD')}",
                "",
                opt["body"],
                "",
            ]

    # ── 9. Next Steps 
    lines += [
        "## 9. Recommended Next Steps",
        "",
        "1. **Immediate (Week 1):** Address critical data quality issues and fix identified data gaps.",
        "2. **Short-term (Month 1):** Deploy the best-performing model with A/B testing for validation.",
        "3. **Medium-term (Quarter 1):** Implement segment-targeted strategies from recommendations.",
        "4. **Ongoing:** Set up automated retraining pipeline — retrain when model performance drops >5%.",
        "5. **Governance:** Establish data quality KPIs and monitor them in a BI dashboard.",
        "",
        "---",
        f"*Report generated by AI Business Analyst Copilot v2.0 — all insights are data-derived.*",
    ]

    return "\n".join(lines)


if __name__ == "__main__":
    import pandas as pd

    print("\n🚀 RUNNING FULL PIPELINE...\n")

    # 🔹 Dummy dataset
    df = pd.DataFrame({
        "sales": [100, 200, 150, 300, 250],
        "region": ["A", "B", "A", "B", "C"],
        "year": [2022, 2022, 2023, 2023, 2023]
    })

    # 🔹 Aggregation
    agg_df = df.groupby("region").agg({"sales": "sum"}).reset_index()

    # 🔹 Meta info
    meta = {
        "original_rows": len(df),
        "original_cols": len(df.columns),
        "agg_rows": len(agg_df),
        "group_keys": ["region"],
        "value_cols": ["sales"]
    }

    # 🔹 KPI
    kpis = {
        "primary_col": "sales",
        "primary_total": df["sales"].sum(),
        "reduction_pct": (1 - len(agg_df)/len(df)) * 100
    }

    # 🔹 Dummy decisions (simulate output)
    decisions = {
        "recommendations": [
            {"title": "Increase Sales in Region B", "body": "Region B shows growth potential", "priority": "High"}
        ],
        "risks": [
            {"title": "Data Missing Risk", "body": "Some values missing", "severity": "Medium"}
        ],
        "optimizations": [
            {"title": "Improve low regions", "body": "Focus on weak areas", "expected_impact": "10% growth"}
        ]
    }

    # 🔹 Generate report
    report = generate_report(
        df_raw=df,
        agg_df=agg_df,
        meta=meta,
        kpis=kpis,
        domain="sales",
        domain_summary="This dataset represents regional sales performance.",
        model_results=None,
        decisions=decisions,
        target_col=None
    )

    # 🔥 PRINT OUTPUT
    print("\n📊 GENERATED REPORT:\n")
    print(report)

    # 🔥 SAVE FILE
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\n✅ Report saved as report.md\n")