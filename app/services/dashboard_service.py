"""
dashboard_service.py  —  backend/app/services/dashboard_service.py
Aggregates all in-memory state into the structured payload the
dashboard endpoint returns to the React frontend.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from app.state import app_state

log = logging.getLogger("dashboard_service")


def _pct_change(current: float, previous: float) -> float:
    if previous == 0:
        return 0.0
    return round((current - previous) / previous * 100, 1)


def build_dashboard_payload() -> Dict[str, Any]:
    """Assemble the full dashboard JSON from app_state.

    Returns synthetic demo data when no real data has been uploaded yet,
    so the UI renders meaningfully out-of-the-box.
    """
    has_real = app_state.features_df is not None

    # ── KPI cards ─────────────────────────────────────────────────────────────
    if has_real:
        df = app_state.features_df
        daily = app_state.daily_traffic

        total_views   = int(df["page_views"].sum())
        unique_users  = int(df["user_id"].nunique())
        avg_session   = round(float(df["session_duration"].mean()), 1)
        avg_bounce    = round(float(df["bounce_rate"].mean()) * 100, 1)

        # Return rate: fraction of users predicted (or labelled) as returners
        clf_m = app_state.return_metrics
        if clf_m:
            cm = clf_m.get("confusion_matrix", [[0, 0], [0, 0]])
            tp = cm[1][1]; fn = cm[1][0]
            return_rate = round(tp / max(tp + fn, 1) * 100, 1)
        else:
            return_rate = round(
                (1 - float(df["bounce_rate"].mean())) * 100, 1
            )

        # Tomorrow's prediction
        fcast_m = app_state.traffic_metrics
        tomorrow_val = fcast_m.get("tomorrow_prediction", {}).get(
            "predicted_visitors", 0
        )

        # Prior-week comparison (use first half vs second half of daily data)
        if daily is not None and len(daily) >= 14:
            mid = len(daily) // 2
            prev_week = daily["visits"].iloc[mid - 7 : mid].mean()
            this_week = daily["visits"].iloc[-7:].mean()
            traffic_change = _pct_change(this_week, prev_week)
        else:
            traffic_change = 0.0

    else:
        # ── Synthetic demo data ───────────────────────────────────────────────
        total_views   = 127_450
        unique_users  = 3_847
        avg_session   = 4.5
        avg_bounce    = 42.3
        return_rate   = 34.2
        tomorrow_val  = 1_342
        traffic_change = 12.3

    kpis: List[Dict] = [
        {"label": "Total Page Views",    "value": f"{total_views:,}",
         "change": traffic_change,       "unit": "views"},
        {"label": "Unique Users",        "value": f"{unique_users:,}",
         "change": 8.7,                  "unit": "users"},
        {"label": "Avg Session Duration","value": f"{avg_session:.1f}m",
         "change": 5.2,                  "unit": "min"},
        {"label": "Bounce Rate",         "value": f"{avg_bounce:.1f}%",
         "change": -3.1,                 "unit": "%"},
        {"label": "Return Rate",         "value": f"{return_rate:.1f}%",
         "change": 2.1,                  "unit": "%"},
        {"label": "Predicted Tomorrow",  "value": f"{tomorrow_val:,}" if tomorrow_val else "N/A",
         "change": None,                 "unit": "visitors"},
    ]

    # ── Traffic trend (last 30 days or synthetic 30 pts) ─────────────────────
    if has_real and app_state.daily_traffic is not None:
        daily_tail = app_state.daily_traffic.tail(30)
        traffic_trend = [
            {
                "date":   str(row["date"])[:10],
                "visits": int(row["visits"]),
                "predicted": None,
            }
            for _, row in daily_tail.iterrows()
        ]
    else:
        traffic_trend = _synthetic_traffic_trend()

    # Overlay forecast model test-set actuals vs predicted (last 30 points)
    fcast_m = app_state.traffic_metrics
    if fcast_m and "predictions" in fcast_m:
        preds = fcast_m["predictions"][-30:]
        traffic_trend = [
            {
                "date":   p["date"],
                "visits": p["actual"],
                "predicted": p["predicted"],
            }
            for p in preds
        ]

    # ── Return distribution ────────────────────────────────────────────────────
    clf_m = app_state.return_metrics
    if clf_m and "confusion_matrix" in clf_m:
        cm = clf_m["confusion_matrix"]
        pos = cm[1][1] + cm[1][0]
        neg = cm[0][0] + cm[0][1]
        return_dist = {
            "will_return":    cm[1][1],
            "wont_return":    cm[0][0] + cm[0][1] + cm[1][0],
            "return_rate":    round(cm[1][1] / max(pos, 1) * 100, 1),
            "total_predicted": pos + neg,
        }
    else:
        return_dist = {
            "will_return":    1_316,
            "wont_return":    2_531,
            "return_rate":    34.2,
            "total_predicted": 3_847,
        }

    # ── Forecast comparison ───────────────────────────────────────────────────
    fcast_preds = fcast_m.get("predictions", []) if fcast_m else []
    if fcast_preds:
        forecast_comparison = [
            {
                "date":      p["date"],
                "actual":    p["actual"],
                "predicted": p["predicted"],
                "error":     abs(p["actual"] - p["predicted"]),
            }
            for p in fcast_preds[-14:]
        ]
    else:
        forecast_comparison = _synthetic_forecast_comparison()

    # ── Model metrics card ────────────────────────────────────────────────────
    model_metrics: Dict[str, Any] = {}
    if clf_m:
        model_metrics["return_model"] = {
            "accuracy":  clf_m.get("accuracy"),
            "f1_macro":  clf_m.get("f1_macro"),
            "roc_auc":   clf_m.get("roc_auc"),
        }
    if fcast_m:
        model_metrics["traffic_model"] = {
            "mae":  fcast_m.get("mae"),
            "rmse": fcast_m.get("rmse"),
            "r2":   fcast_m.get("r2"),
            "mape": fcast_m.get("mape"),
        }

    return {
        "kpis":               kpis,
        "traffic_trend":      traffic_trend,
        "return_distribution": return_dist,
        "forecast_comparison": forecast_comparison,
        "model_metrics":      model_metrics,
        "data_available":     has_real,
    }


# ── Synthetic fallback helpers ─────────────────────────────────────────────────
def _synthetic_traffic_trend() -> List[Dict]:
    rng = np.random.default_rng(42)
    today = pd.Timestamp.now().normalize()
    result = []
    for i in range(30):
        day  = today - pd.Timedelta(days=29 - i)
        base = 1_200 + i * 10
        is_we = day.dayofweek >= 5
        visits = max(50, int(base * (0.65 if is_we else 1.0) + rng.integers(-80, 80)))
        result.append({
            "date":      day.strftime("%Y-%m-%d"),
            "visits":    visits,
            "predicted": max(50, int(visits * rng.uniform(0.92, 1.08))) if i >= 24 else None,
        })
    return result


def _synthetic_forecast_comparison() -> List[Dict]:
    rng = np.random.default_rng(42)
    today = pd.Timestamp.now().normalize()
    result = []
    for i in range(14):
        day    = today - pd.Timedelta(days=13 - i)
        actual = max(50, int(1_300 + rng.integers(-150, 150)))
        pred   = max(50, int(actual + rng.integers(-120, 120)))
        result.append({
            "date":      day.strftime("%Y-%m-%d"),
            "actual":    actual,
            "predicted": pred,
            "error":     abs(actual - pred),
        })
    return result
