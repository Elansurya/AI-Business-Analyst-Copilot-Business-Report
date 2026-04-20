from __future__ import annotations
import os
import pickle
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor
)
from sklearn.metrics import (
    accuracy_score, f1_score, mean_squared_error,
    r2_score, classification_report
)

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

# ── colours 
BG      = "#0A0C14"
CARD    = "#13172A"
TEXT    = "#E2E8F0"
MUTED   = "#6B7494"
ACCENT  = "#6C63FF"
GREEN   = "#10B981"
RED     = "#EF4444"
YELLOW  = "#F59E0B"
PALETTE = ["#6C63FF", "#10B981", "#F59E0B", "#EF4444", "#06B6D4", "#8B5CF6", "#F97316"]

# ── model persistence paths 
MODELS_DIR = Path("models")


def _layout(fig: go.Figure, title: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(color=TEXT, size=14)),
        paper_bgcolor=BG, plot_bgcolor=CARD,
        font=dict(color=TEXT),
        height=height, margin=dict(t=50, b=40, l=10, r=10),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT)),
    )
    fig.update_xaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor="#1E2340", tickfont=dict(color=MUTED))
    return fig


# ── task detection 
def detect_task(series: pd.Series) -> str:
    """Return 'classification' or 'regression' based on target column."""
    if not pd.api.types.is_numeric_dtype(series):
        return "classification"
    return "classification" if series.nunique() <= 10 else "regression"


# ── preprocessing 
def preprocess(agg_df: pd.DataFrame, target_col: str):
    """
    Prepare aggregated DataFrame for modeling.
    Returns X (array), y (array), feature_names, task, label_names.
    """
    df = agg_df.copy()

    # Drop leaky / non-feature cols (keep _sum and _mean)
    drop_pattern = ["_count", "_std"]
    cols_to_drop = [
        c for c in df.columns
        if any(c.endswith(p) for p in drop_pattern) and c != target_col
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Find target column (or its aggregated version)
    if target_col not in df.columns:
        for suffix in ("_sum", "_mean", ""):
            cand = f"{target_col}{suffix}"
            if cand in df.columns:
                target_col = cand
                break

    y_raw = df[target_col].copy()
    X_df  = df.drop(columns=[target_col])

    # Encode categoricals in X
    for col in X_df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))

    feature_names = X_df.columns.tolist()

    task = detect_task(y_raw)
    if task == "classification":
        le_y = LabelEncoder()
        y    = le_y.fit_transform(y_raw.astype(str))
        label_names = list(le_y.classes_)
    else:
        y           = y_raw.values.astype(float)
        label_names = None

    # Impute missing values
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(X_df)

    return X, y, feature_names, task, label_names


# ── save model 
def save_model(model, model_name: str, overwrite: bool = False) -> Path:
    """
    Save a trained sklearn model to models/<model_name>.pkl.

    Parameters
    ----------
    model       : trained sklearn estimator
    model_name  : filename stem (spaces replaced with underscores)
    overwrite   : if False and file exists, a numbered suffix is added

    Returns
    -------
    Path to the saved .pkl file
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace(" ", "_").lower()
    pkl_path  = MODELS_DIR / f"{safe_name}.pkl"

    if not overwrite and pkl_path.exists():
        counter = 1
        while pkl_path.exists():
            pkl_path = MODELS_DIR / f"{safe_name}_{counter}.pkl"
            counter += 1

    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)

    print(f"✅ Model saved → {pkl_path.resolve()}")
    return pkl_path


# ── load model 
def load_model(pkl_path: str | Path):
    """
    Load a trained model from a .pkl file.

    Parameters
    ----------
    pkl_path : path to the .pkl file

    Returns
    -------
    Loaded sklearn estimator
    """
    pkl_path = Path(pkl_path)
    if not pkl_path.exists():
        raise FileNotFoundError(f"Model file not found: {pkl_path.resolve()}")

    with open(pkl_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Model loaded ← {pkl_path.resolve()}")
    return model


# ── training 
def train_models(agg_df: pd.DataFrame, target_col: str) -> dict:
    """
    Train 3 models, compare metrics, pick the best, save it as .pkl.
    Returns a results dict with metrics, plots data, and model artifacts.
    """
    X, y, feature_names, task, label_names = preprocess(agg_df, target_col)

    n         = len(X)
    test_size = 0.20 if n >= 50 else 0.30
    n_cv      = min(5, max(2, n // 10))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,
        stratify=y if task == "classification" and n >= 20 else None,
    )

    if task == "classification":
        models = {
            "Logistic Regression": LogisticRegression(
                max_iter=1000, C=1.0, class_weight="balanced", random_state=42
            ),
            "Random Forest": RandomForestClassifier(
                n_estimators=150, max_depth=8,
                class_weight="balanced", random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            ),
        }
        cv      = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=42)
        scoring = "f1_weighted"
    else:
        models = {
            "Ridge Regression": Ridge(alpha=1.0),
            "Random Forest": RandomForestRegressor(
                n_estimators=150, max_depth=8, random_state=42, n_jobs=-1
            ),
            "Gradient Boosting": GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42
            ),
        }
        cv      = KFold(n_splits=n_cv, shuffle=True, random_state=42)
        scoring = "r2"

    results = {}
    trained = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred    = model.predict(X_test)
        y_pred_tr = model.predict(X_train)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)

        if task == "classification":
            test_score  = accuracy_score(y_test, y_pred) * 100
            train_score = accuracy_score(y_train, y_pred_tr) * 100
            f1          = f1_score(y_test, y_pred, average="weighted", zero_division=0) * 100
            results[name] = {
                "Accuracy (%)":  round(test_score, 2),
                "F1-Score (%)":  round(f1, 2),
                "CV Score (%)":  round(cv_scores.mean() * 100, 2),
                "CV Std":        round(cv_scores.std() * 100, 2),
                "Train Acc (%)": round(train_score, 2),
                "Overfit Gap":   round(train_score - test_score, 2),
                "primary":       round(f1, 2),
                "task":          "classification",
            }
        else:
            rmse     = round(np.sqrt(mean_squared_error(y_test, y_pred)), 4)
            r2       = round(r2_score(y_test, y_pred) * 100, 2)
            train_r2 = round(r2_score(y_train, y_pred_tr) * 100, 2)
            results[name] = {
                "R² (%)":       r2,
                "RMSE":         rmse,
                "CV R² (%)":    round(cv_scores.mean() * 100, 2),
                "CV Std":       round(cv_scores.std() * 100, 2),
                "Train R² (%)": train_r2,
                "Overfit Gap":  round(train_r2 - r2, 2),
                "primary":      r2,
                "task":         "regression",
            }
        trained[name] = model

    best_name  = max(results, key=lambda k: results[k]["primary"])
    best_model = trained[best_name]

    # ── save best model 
    saved_path = save_model(best_model, best_name)

    # ── feature importance 
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "coef_"):
        coef        = best_model.coef_
        importances = np.abs(coef if coef.ndim == 1 else coef.mean(axis=0))
    else:
        importances = np.ones(len(feature_names)) / len(feature_names)

    importance_df = (
        pd.DataFrame({"Feature": feature_names, "Importance": importances})
        .sort_values("Importance", ascending=False)
        .head(15)
        .reset_index(drop=True)
    )

    # ── SHAP 
    shap_df = None
    if SHAP_OK and hasattr(best_model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(best_model)
            sv        = explainer.shap_values(X_test[:min(100, len(X_test))])
            if isinstance(sv, list):
                sv = np.abs(sv).mean(axis=0)
            shap_mean = np.abs(sv).mean(axis=0)
            shap_df   = (
                pd.DataFrame({"Feature": feature_names, "SHAP": shap_mean})
                .sort_values("SHAP", ascending=False)
                .head(15)
            )
        except Exception:
            shap_df = None

    # ── prediction sample 
    pred_sample = pd.DataFrame({
        "Actual":    y_test[:20],
        "Predicted": best_model.predict(X_test[:20]),
    })
    pred_sample["Error"] = (pred_sample["Actual"] - pred_sample["Predicted"]).abs()

    return {
        "results":       results,
        "best_name":     best_name,
        "best_model":    best_model,
        "saved_path":    saved_path,
        "importance_df": importance_df,
        "shap_df":       shap_df,
        "task":          task,
        "label_names":   label_names,
        "feature_names": feature_names,
        "pred_sample":   pred_sample,
        "X_test":        X_test,
        "y_test":        y_test,
        "n_cv":          n_cv,
    }


# ── plots 
def plot_comparison(results: dict) -> go.Figure:
    names = list(results.keys())
    task  = results[names[0]]["task"]

    if task == "classification":
        acc = [results[n]["Accuracy (%)"] for n in names]
        f1  = [results[n]["F1-Score (%)"] for n in names]
        cv  = [results[n]["CV Score (%)"] for n in names]
        fig = go.Figure(data=[
            go.Bar(name="Accuracy (%)", x=names, y=acc,
                   marker_color=ACCENT, text=[f"{v}%" for v in acc], textposition="outside"),
            go.Bar(name="F1-Score (%)", x=names, y=f1,
                   marker_color=GREEN,  text=[f"{v}%" for v in f1],  textposition="outside"),
            go.Bar(name="CV Score (%)", x=names, y=cv,
                   marker_color=YELLOW, text=[f"{v}%" for v in cv],  textposition="outside"),
        ])
        fig.update_layout(barmode="group", yaxis_range=[0, 120])
    else:
        r2   = [results[n]["R² (%)"] for n in names]
        rmse = [results[n]["RMSE"]   for n in names]
        fig  = make_subplots(rows=1, cols=2,
                             subplot_titles=["R² Score (%)", "RMSE (lower=better)"])
        fig.add_trace(go.Bar(x=names, y=r2, marker_color=ACCENT,
                              text=[f"{v}%" for v in r2], textposition="outside",
                              showlegend=False), row=1, col=1)
        fig.add_trace(go.Bar(x=names, y=rmse, marker_color=RED,
                              text=[str(v) for v in rmse], textposition="outside",
                              showlegend=False), row=1, col=2)

    _layout(fig, "Model Performance Comparison", 420)
    return fig


def plot_overfitting(results: dict) -> go.Figure:
    """Train vs test score comparison — overfitting detection."""
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
        go.Bar(name="Train Score", x=names, y=train, marker_color=YELLOW, opacity=0.75),
        go.Bar(name="Test Score",  x=names, y=test,  marker_color=ACCENT),
    ])
    fig.update_layout(barmode="group")
    _layout(fig, f"Overfitting Check — Train vs Test {label}", 380)
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, shap_df=None) -> go.Figure:
    if shap_df is not None:
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=["Model Feature Importance", "SHAP Importance"])
        fig.add_trace(go.Bar(
            y=importance_df["Feature"][::-1],
            x=importance_df["Importance"][::-1],
            orientation="h", marker_color=ACCENT, showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Bar(
            y=shap_df["Feature"][::-1],
            x=shap_df["SHAP"][::-1],
            orientation="h", marker_color="#8B5CF6", showlegend=False,
        ), row=1, col=2)
        height = 480
    else:
        fig = go.Figure(go.Bar(
            y=importance_df["Feature"][::-1],
            x=importance_df["Importance"][::-1],
            orientation="h",
            marker_color=[
                ACCENT if i < 3 else GREEN if i < 6 else MUTED
                for i in range(len(importance_df) - 1, -1, -1)
            ],
            text=[f"{v:.3f}" for v in importance_df["Importance"][::-1]],
            textposition="outside",
        ))
        height = 450

    _layout(fig, "Feature Importance — Best Model", height)
    fig.update_layout(margin=dict(l=150))
    return fig


# ── text insight 
def model_insight_text(model_results: dict) -> str:
    best    = model_results["best_name"]
    results = model_results["results"]
    task    = model_results["task"]
    imp     = model_results["importance_df"]

    parts = []
    if task == "classification":
        acc = results[best]["Accuracy (%)"]
        f1  = results[best]["F1-Score (%)"]
        gap = results[best]["Overfit Gap"]
        parts.append(
            f"**Best: {best}** — Accuracy {acc:.1f}%, F1 {f1:.1f}%. "
            + (f"⚠️ Possible overfitting (train-test gap={gap}%)."
               if gap > 15 else "✅ Generalization looks healthy.")
        )
    else:
        r2   = results[best]["R² (%)"]
        rmse = results[best]["RMSE"]
        gap  = results[best]["Overfit Gap"]
        parts.append(
            f"**Best: {best}** — R²={r2:.1f}%, RMSE={rmse}. "
            f"Explains {r2:.1f}% of variance. "
            + ("⚠️ Overfitting detected." if gap > 20 else "✅ Good generalization.")
        )

    top3    = imp["Feature"].head(3).tolist()
    top_pct = imp["Importance"].head(3).sum() / imp["Importance"].sum() * 100
    parts.append(
        f"**Top drivers:** {', '.join(f'`{f}`' for f in top3)} — "
        f"collectively {top_pct:.1f}% of predictive power."
    )
    return " | ".join(parts)


# ── model context for RAG 
def build_model_context(model_results: dict) -> str:
    best    = model_results["best_name"]
    results = model_results["results"]
    task    = model_results["task"]
    imp     = model_results["importance_df"]

    lines = [f"Best model: {best}. Task type: {task}."]
    for name, m in results.items():
        metrics = {k: v for k, v in m.items() if k not in ("primary", "task")}
        lines.append(f"Model '{name}': " + ", ".join(f"{k}={v}" for k, v in metrics.items()) + ".")

    lines.append("Feature importances (top 10):")
    total_imp = imp["Importance"].sum()
    for _, row in imp.head(10).iterrows():
        pct = row["Importance"] / total_imp * 100 if total_imp > 0 else 0
        lines.append(f"  '{row['Feature']}': {row['Importance']:.4f} ({pct:.1f}%)")

    top3 = imp["Feature"].head(3).tolist()
    lines.append(f"Top 3 predictive features: {', '.join(top3)}.")

    if model_results.get("shap_df") is not None:
        top_shap = model_results["shap_df"]["Feature"].head(3).tolist()
        lines.append(f"Top 3 by SHAP: {', '.join(top_shap)}.")

    return "\n".join(lines)


# ── entrypoint 
if __name__ == "__main__":
    from sklearn.datasets import load_iris, load_diabetes

    print("=" * 55)
    print("  Demo 1 — Classification (Iris dataset)")
    print("=" * 55)
    iris   = load_iris(as_frame=True)
    df_cls = iris.frame.rename(columns={"target": "species"})

    model_results_cls = train_models(df_cls, target_col="species")
    print(f"Best classifier : {model_results_cls['best_name']}")
    print(f"Saved to        : {model_results_cls['saved_path']}")
    print(model_insight_text(model_results_cls))
    print()

    # Reload and verify
    reloaded_cls = load_model(model_results_cls["saved_path"])
    preds = reloaded_cls.predict(model_results_cls["X_test"][:5])
    print(f"Sample predictions (reloaded model): {preds}")

    print()
    print("=" * 55)
    print("  Demo 2 — Regression (Diabetes dataset)")
    print("=" * 55)
    diabetes   = load_diabetes(as_frame=True)
    df_reg     = diabetes.frame

    model_results_reg = train_models(df_reg, target_col="target")
    print(f"Best regressor  : {model_results_reg['best_name']}")
    print(f"Saved to        : {model_results_reg['saved_path']}")
    print(model_insight_text(model_results_reg))
    print()

    # Reload and verify
    reloaded_reg = load_model(model_results_reg["saved_path"])
    preds_reg = reloaded_reg.predict(model_results_reg["X_test"][:5])
    print(f"Sample predictions (reloaded model): {preds_reg.round(2)}")