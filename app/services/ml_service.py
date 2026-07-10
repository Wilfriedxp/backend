"""
ml_service.py  —  backend/app/services/ml_service.py
Business-logic layer that bridges the API endpoints and the ML models.
Owns all training and inference operations; keeps endpoints thin.
"""
from __future__ import annotations

import logging
import sys
import tempfile
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.state import app_state
from app.core.config import settings

# ── Add ml directory to sys.path so training modules are importable ───────────
_ML_DIR = settings.ML_DIR
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from pipelines.preprocessing_pipeline import WebLogPreprocessor, PipelineConfig   # noqa: E402
from training.train_return_classifier import (                                     # noqa: E402
    ReturnUserClassifier, ClassifierConfig, create_return_labels, FEATURE_COLS as CLF_FEATURES,
)
from training.train_traffic_forecaster import (                                    # noqa: E402
    TrafficForecaster, ForecasterConfig, generate_daily_traffic,
    engineer_features, FEATURE_COLS as FCST_FEATURES, TARGET_COL,
)

log = logging.getLogger("ml_service")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_uploaded_log(raw_df: pd.DataFrame) -> Dict[str, Any]:
    """Run the preprocessing pipeline on a freshly uploaded log.

    Steps
    -----
    1. Save raw_df to a temp CSV (pipeline expects a file path).
    2. Run WebLogPreprocessor — produces per-user feature matrix.
    3. Aggregate daily traffic counts from the raw log.
    4. Store both in app_state.
    5. Return upload statistics.
    """
    log.info("Preprocessing %d log rows …", len(raw_df))

    # Write raw_df to a temp file for the preprocessor
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="upload_"
    ) as f:
        raw_df.to_csv(f, index=False)
        tmp = f.name

    try:
        preprocessor = WebLogPreprocessor(PipelineConfig())
        features_df  = preprocessor.run(tmp)
    finally:
        os.unlink(tmp)

    # Aggregate daily traffic (date → total visit count)
    raw_df_copy = raw_df.copy()
    raw_df_copy["timestamp"] = pd.to_datetime(raw_df_copy["timestamp"], errors="coerce")
    raw_df_copy = raw_df_copy.dropna(subset=["timestamp"])
    daily = (
        raw_df_copy.groupby(raw_df_copy["timestamp"].dt.date)
        .size()
        .reset_index(name="visits")
        .rename(columns={"timestamp": "date"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    daily["date"] = pd.to_datetime(daily["date"])

    # Persist to global state
    app_state.raw_df        = raw_df
    app_state.features_df   = features_df
    app_state.daily_traffic = daily

    # Build upload statistics
    stats = {
        "total_rows":         int(len(raw_df)),
        "unique_users":       int(features_df["user_id"].nunique()),
        "date_range_start":   str(daily["date"].min().date()),
        "date_range_end":     str(daily["date"].max().date()),
        "total_days":         int(len(daily)),
        "avg_daily_visits":   round(float(daily["visits"].mean()), 1),
        "total_page_views":   int(features_df["page_views"].sum()),
        "avg_session_min":    round(float(features_df["session_duration"].mean()), 2),
        "avg_bounce_rate":    round(float(features_df["bounce_rate"].mean()), 4),
        "avg_nav_depth":      round(float(features_df["navigation_depth"].mean()), 2),
    }
    app_state.upload_stats = stats
    log.info("Preprocessing complete | %d users | %d days", stats["unique_users"], stats["total_days"])
    return stats


# ─────────────────────────────────────────────────────────────────────────────
# Return-user model
# ─────────────────────────────────────────────────────────────────────────────
def train_return_model() -> Dict[str, Any]:
    """Train the return-user classifier on the currently loaded data.

    Requires ``app_state.raw_df`` and ``app_state.features_df`` to be set
    (i.e., ``/upload`` must be called first).
    """
    if app_state.raw_df is None or app_state.features_df is None:
        raise RuntimeError("No data loaded. Call POST /upload first.")

    log.info("Training return-user classifier …")
    cfg = ClassifierConfig()

    # Create temporal labels
    labels, train_window_df, cutoff = create_return_labels(app_state.raw_df, cfg)

    # Write training-window slice to temp file for preprocessor
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="train_window_"
    ) as f:
        train_window_df.to_csv(f, index=False)
        tmp = f.name

    try:
        features_df = WebLogPreprocessor(PipelineConfig()).run(tmp)
    finally:
        os.unlink(tmp)

    dataset = (
        features_df.merge(labels, on="user_id", how="inner")
        .dropna(subset=CLF_FEATURES + ["will_return"])
        .reset_index(drop=True)
    )

    if len(dataset) < 10:
        raise RuntimeError(
            f"Only {len(dataset)} labelled users — need at least 10 to train. "
            "Try uploading a larger log file."
        )

    from sklearn.model_selection import train_test_split

    X = dataset[CLF_FEATURES]
    y = dataset["will_return"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, stratify=y, random_state=cfg.random_state,
    )

    clf = ReturnUserClassifier(cfg)
    clf.cross_validate(X_train, y_train)
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save(settings.return_model_path)

    app_state.return_classifier = clf
    app_state.return_metrics    = clf.test_metrics

    result = {
        "users_trained_on": len(dataset),
        "test_metrics":     clf.test_metrics,
        "cv_summary": {
            "accuracy_mean":  round(float(clf.cv_results["accuracy"].mean()),  4),
            "f1_mean":        round(float(clf.cv_results["f1"].mean()),        4),
            "roc_auc_mean":   round(float(clf.cv_results["roc_auc"].mean()),   4),
            "accuracy_std":   round(float(clf.cv_results["accuracy"].std()),   4),
        },
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in (clf.feature_importances or {}).items()
        } if clf.feature_importances is not None else {},
    }
    log.info("Return-model trained | Acc %.4f | AUC %.4f",
             result["test_metrics"]["accuracy"], result["test_metrics"]["roc_auc"])
    return result


def predict_return(user_features_list: List[Dict]) -> List[Dict]:
    """Run return-user inference for one or more feature vectors."""
    # Load model if not in memory
    if app_state.return_classifier is None:
        if settings.return_model_path.exists():
            app_state.return_classifier = ReturnUserClassifier.load(settings.return_model_path)
        else:
            raise RuntimeError(
                "Return-user model not trained yet. Call POST /train-return-model."
            )

    clf  = app_state.return_classifier
    df   = pd.DataFrame(user_features_list)[CLF_FEATURES]
    preds = clf.predict(df)
    proba = clf.predict_proba(df)

    results = []
    for pred, prob in zip(preds, proba):
        band = "HIGH" if prob > 0.7 else ("MEDIUM" if prob > 0.4 else "LOW")
        results.append({
            "will_return":     bool(pred),
            "probability":     round(float(prob), 4),
            "label":           "Will Return" if pred else "Will NOT Return",
            "confidence_band": band,
        })
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Traffic forecasting model
# ─────────────────────────────────────────────────────────────────────────────
def train_traffic_model() -> Dict[str, Any]:
    """Train the traffic forecasting model on the loaded daily traffic data."""
    # Use real data if available, else synthetic
    if app_state.daily_traffic is not None and len(app_state.daily_traffic) >= 20:
        daily_df = app_state.daily_traffic
        log.info("Training traffic model on %d days of real data …", len(daily_df))
    else:
        log.info("Insufficient real data — using synthetic 365-day dataset.")
        daily_df = generate_daily_traffic(n_days=365, seed=42)

    cfg      = ForecasterConfig()
    feat_df  = engineer_features(daily_df)
    X = feat_df[FCST_FEATURES]
    y = feat_df[TARGET_COL]

    holdout = cfg.holdout_days
    X_train, X_test = X.iloc[:-holdout], X.iloc[-holdout:]
    y_train, y_test = y.iloc[:-holdout], y.iloc[-holdout:]

    forecaster = TrafficForecaster(cfg)
    forecaster.cross_validate(X_train, y_train)
    forecaster.train(X_train, y_train)
    forecaster.evaluate(X_test, y_test)
    forecaster.save(settings.traffic_model_path)

    app_state.traffic_forecaster = forecaster
    app_state.traffic_metrics    = forecaster.test_metrics

    cv = forecaster.cv_results
    result = {
        "training_days": len(daily_df),
        "test_metrics":  forecaster.test_metrics,
        "cv_summary": {
            "mae_mean":  round(float(cv["mae_mean"]),  2),
            "rmse_mean": round(float(cv["rmse_mean"]), 2),
            "r2_mean":   round(float(cv["r2_mean"]),   4),
            "mape_mean": round(float(cv["mape_mean"]), 2),
        },
        "feature_importances": {
            feat: round(float(imp), 4)
            for feat, imp in (forecaster.feature_importances or {}).items()
        } if forecaster.feature_importances is not None else {},
    }
    log.info("Traffic model trained | MAE %.2f | R² %.4f",
             result["test_metrics"]["mae"], result["test_metrics"]["r2"])
    return result


def predict_traffic() -> Dict:
    """Predict tomorrow's visitor count using the most recent 14 days of data."""
    if app_state.traffic_forecaster is None:
        if settings.traffic_model_path.exists():
            app_state.traffic_forecaster = TrafficForecaster.load(settings.traffic_model_path)
        else:
            raise RuntimeError(
                "Traffic forecast model not trained. Call POST /train-traffic-model first."
            )

    # Use real daily traffic if available, else synthetic last 14 days
    if app_state.daily_traffic is not None and len(app_state.daily_traffic) >= 14:
        recent_14 = app_state.daily_traffic["visits"].tail(14).tolist()
    else:
        synthetic = generate_daily_traffic(n_days=365, seed=42)
        recent_14 = synthetic["visits"].tail(14).tolist()

    return app_state.traffic_forecaster.predict_tomorrow(recent_14)
