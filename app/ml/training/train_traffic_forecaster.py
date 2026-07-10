"""
train_traffic_forecaster.py
============================
Random Forest Regressor that predicts tomorrow's website visitor count.

Project path : backend/app/ml/training/train_traffic_forecaster.py
Saves model  : backend/app/ml/models/traffic_forecast_model.pkl

Feature engineering from daily traffic history
-----------------------------------------------
  lag_1, lag_7, lag_14      visit counts 1 / 7 / 14 days ago
  rolling_mean_7/14         7-day and 14-day moving averages
  rolling_std_7             7-day rolling standard deviation
  day_of_week (0–6)         captures weekly seasonality
  month (1–12)              captures yearly seasonality
  day_of_month              intra-month patterns
  week_of_year              broader seasonal signal
  is_weekend (0/1)          weekend traffic drop
  is_month_end (0/1)        end-of-month traffic spikes

Evaluation strategy
-------------------
  TimeSeriesSplit (5-fold) — respects temporal order, no future leakage
  Hold-out : last 30 days of the dataset
  Metrics   : MAE · RMSE · R² · MAPE

Usage
-----
    python train_traffic_forecaster.py
    python train_traffic_forecaster.py --days 365 --n-estimators 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("traffic_forecaster")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "lag_1", "lag_7", "lag_14",
    "rolling_mean_7", "rolling_mean_14", "rolling_std_7",
    "day_of_week", "month", "day_of_month",
    "week_of_year", "is_weekend", "is_month_end",
]
TARGET_COL    = "visits"
HOLDOUT_DAYS  = 30          # last N days held out for final evaluation
MODEL_DIR     = Path(__file__).resolve().parents[1] / "models"
MODEL_PATH    = MODEL_DIR / "traffic_forecast_model.pkl"
MODEL_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ForecasterConfig:
    """Hyperparameters and evaluation settings.

    Attributes
    ----------
    n_estimators    : number of trees in the Random Forest
    max_depth       : max tree depth (None = fully grown)
    min_samples_leaf: leaf size regulariser
    max_features    : features considered per split
    cv_folds        : number of TimeSeriesSplit folds
    holdout_days    : days reserved for hold-out evaluation
    random_state    : global random seed
    """
    n_estimators:    int   = 200
    max_depth:       Optional[int] = None
    min_samples_leaf: int  = 2
    max_features:    str   = "sqrt"
    cv_folds:        int   = 5
    holdout_days:    int   = HOLDOUT_DAYS
    random_state:    int   = 42


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic traffic data generator
# ─────────────────────────────────────────────────────────────────────────────
def generate_daily_traffic(n_days: int = 365, seed: int = 42) -> pd.DataFrame:
    """Synthesise a realistic daily visitor time-series with:

    - A long-term growth trend (~50 % growth over the year)
    - Weekly seasonality (weekdays > weekends)
    - Monthly seasonality (peaks in Q4, dips in mid-summer)
    - Five campaign-driven traffic spikes
    - Gaussian noise (±12 %)

    Parameters
    ----------
    n_days : int   Number of days to generate.
    seed   : int   Random seed.

    Returns
    -------
    pd.DataFrame with columns [date, visits].
    """
    rng   = np.random.default_rng(seed)
    dates = pd.date_range(start="2024-01-01", periods=n_days, freq="D")

    # Long-term growth trend
    trend = np.linspace(1_000, 1_550, n_days)

    # Weekly seasonality: Mon → Sun weights
    day_w = {0: 1.22, 1: 1.28, 2: 1.22, 3: 1.16, 4: 1.08, 5: 0.68, 6: 0.58}
    weekly = np.array([day_w[d.dayofweek] for d in dates])

    # Monthly seasonality
    mon_w = {1: 0.85, 2: 0.88, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.10,
             7: 0.94, 8: 0.90, 9: 1.06, 10: 1.12, 11: 1.18, 12: 1.22}
    monthly = np.array([mon_w[d.month] for d in dates])

    # Gaussian noise
    noise = rng.normal(1.0, 0.12, n_days)

    # Campaign spikes on 5 random weekdays
    spikes = np.ones(n_days)
    spike_idx = rng.choice(
        [i for i, d in enumerate(dates) if d.dayofweek < 5],
        size=5, replace=False,
    )
    spikes[spike_idx] = rng.uniform(1.8, 2.6, 5)

    visits = (trend * weekly * monthly * noise * spikes).round().astype(int)
    visits = np.maximum(visits, 50)   # floor at 50 visitors

    df = pd.DataFrame({"date": dates, TARGET_COL: visits})
    log.info(
        "Generated %d days of traffic | mean=%.0f | min=%d | max=%d",
        n_days, visits.mean(), visits.min(), visits.max(),
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature engineering
# ─────────────────────────────────────────────────────────────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a daily traffic DataFrame into a feature matrix.

    All lag and rolling features use *shifted* values so no future data
    leaks into any row's feature vector.

    Parameters
    ----------
    df : DataFrame with [date, visits] sorted by date ascending.

    Returns
    -------
    DataFrame with FEATURE_COLS + [date, visits]; NaN rows dropped.
    """
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # ── Lag features ─────────────────────────────────────────────────────────
    df["lag_1"]  = df[TARGET_COL].shift(1)
    df["lag_7"]  = df[TARGET_COL].shift(7)
    df["lag_14"] = df[TARGET_COL].shift(14)

    # ── Rolling statistics (on *shifted* series — no leakage) ────────────────
    shifted = df[TARGET_COL].shift(1)
    df["rolling_mean_7"]  = shifted.rolling(7).mean()
    df["rolling_mean_14"] = shifted.rolling(14).mean()
    df["rolling_std_7"]   = shifted.rolling(7).std().fillna(0)

    # ── Calendar features ─────────────────────────────────────────────────────
    df["day_of_week"]  = df["date"].dt.dayofweek
    df["month"]        = df["date"].dt.month
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)

    df = df.dropna(subset=FEATURE_COLS).reset_index(drop=True)
    log.info("Feature engineering: %d usable records (from %d raw days).", len(df), len(df) + 14)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helper metrics
# ─────────────────────────────────────────────────────────────────────────────
def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# ─────────────────────────────────────────────────────────────────────────────
# Forecaster class
# ─────────────────────────────────────────────────────────────────────────────
class TrafficForecaster:
    """Random Forest pipeline for daily web traffic forecasting.

    Internal pipeline: StandardScaler → RandomForestRegressor.
    Evaluation uses TimeSeriesSplit to respect temporal ordering.

    Parameters
    ----------
    config : ForecasterConfig, optional
    """

    def __init__(self, config: Optional[ForecasterConfig] = None) -> None:
        self.cfg = config or ForecasterConfig()
        self.pipeline: Optional[Pipeline] = None
        self.cv_results: Dict = {}
        self.test_metrics: Dict = {}
        self.feature_importances: Optional[pd.Series] = None
        self._daily_df: Optional[pd.DataFrame] = None  # cached for predict_tomorrow

    # ── Build pipeline ────────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestRegressor(
                n_estimators    = self.cfg.n_estimators,
                max_depth       = self.cfg.max_depth,
                min_samples_leaf= self.cfg.min_samples_leaf,
                max_features    = self.cfg.max_features,
                random_state    = self.cfg.random_state,
                n_jobs          = -1,
            )),
        ])

    # ── Time-series cross-validation ─────────────────────────────────────────
    def cross_validate(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, np.ndarray]:
        """Run TimeSeriesSplit CV and store per-fold metrics.

        TimeSeriesSplit ensures each validation fold only contains data
        *after* the training fold — no lookahead bias.
        """
        tscv = TimeSeriesSplit(n_splits=self.cfg.cv_folds)
        fold_metrics: list[dict] = []

        log.info("Running %d-fold TimeSeriesSplit cross-validation …", self.cfg.cv_folds)
        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
            Xtr, Xva = X.iloc[tr_idx], X.iloc[va_idx]
            ytr, yva = y.iloc[tr_idx], y.iloc[va_idx]

            pipe = self._build_pipeline()
            pipe.fit(Xtr[FEATURE_COLS], ytr)
            ypred = pipe.predict(Xva[FEATURE_COLS])

            fold_metrics.append({
                "fold":       fold,
                "train_size": len(tr_idx),
                "val_size":   len(va_idx),
                "mae":        mean_absolute_error(yva, ypred),
                "rmse":       _rmse(yva.values, ypred),
                "r2":         r2_score(yva, ypred),
                "mape":       _mape(yva.values, ypred),
            })

        fold_df = pd.DataFrame(fold_metrics)
        self.cv_results = {
            "fold_details": fold_df,
            "mae_mean":  fold_df["mae"].mean(),   "mae_std":  fold_df["mae"].std(),
            "rmse_mean": fold_df["rmse"].mean(),  "rmse_std": fold_df["rmse"].std(),
            "r2_mean":   fold_df["r2"].mean(),    "r2_std":   fold_df["r2"].std(),
            "mape_mean": fold_df["mape"].mean(),  "mape_std": fold_df["mape"].std(),
        }
        log.info(
            "CV done | MAE %.1f ± %.1f | RMSE %.1f ± %.1f | R² %.4f ± %.4f | MAPE %.1f%% ± %.1f%%",
            self.cv_results["mae_mean"],  self.cv_results["mae_std"],
            self.cv_results["rmse_mean"], self.cv_results["rmse_std"],
            self.cv_results["r2_mean"],   self.cv_results["r2_std"],
            self.cv_results["mape_mean"], self.cv_results["mape_std"],
        )
        return self.cv_results

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the full pipeline on the training portion."""
        log.info("Training RF Regressor on %d records …", len(X_train))
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train[FEATURE_COLS], y_train)

        rf = self.pipeline.named_steps["rf"]
        self.feature_importances = pd.Series(
            rf.feature_importances_, index=FEATURE_COLS
        ).sort_values(ascending=False)

        log.info(
            "Training complete | top feature: %s (%.4f)",
            self.feature_importances.index[0],
            self.feature_importances.iloc[0],
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(
        self, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict:
        """Compute MAE, RMSE, R², MAPE on the hold-out test set."""
        if self.pipeline is None:
            raise RuntimeError("Call train() before evaluate().")

        y_pred = self.pipeline.predict(X_test[FEATURE_COLS])

        self.test_metrics = {
            "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
            "rmse": round(float(_rmse(y_test.values, y_pred)), 2),
            "r2":   round(float(r2_score(y_test, y_pred)), 4),
            "mape": round(float(_mape(y_test.values, y_pred)), 2),
            "predictions": [
                {"date": str(d), "actual": int(a), "predicted": max(0, int(round(p)))}
                for d, a, p in zip(
                    X_test.index.map(lambda i: i), y_test.values, y_pred
                )
            ],
        }
        log.info(
            "Test evaluation | MAE %.2f | RMSE %.2f | R² %.4f | MAPE %.2f%%",
            self.test_metrics["mae"], self.test_metrics["rmse"],
            self.test_metrics["r2"],  self.test_metrics["mape"],
        )
        return self.test_metrics

    # ── Predict tomorrow ──────────────────────────────────────────────────────
    def predict_tomorrow(
        self, recent_visits: list[int]
    ) -> Dict:
        """Predict tomorrow's visitor count from the last 14 daily counts.

        Parameters
        ----------
        recent_visits : list of ints, oldest first, length ≥ 14.

        Returns
        -------
        dict with: date, predicted_visitors, confidence_low, confidence_high
        """
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")
        if len(recent_visits) < 14:
            raise ValueError("Provide at least 14 days of visit history.")

        tomorrow = pd.Timestamp.now().normalize() + pd.Timedelta(days=1)
        hist = np.array(recent_visits[-14:], dtype=float)

        features = {
            "lag_1":          hist[-1],
            "lag_7":          hist[-7],
            "lag_14":         hist[-14],
            "rolling_mean_7": hist[-7:].mean(),
            "rolling_mean_14": hist.mean(),
            "rolling_std_7":  hist[-7:].std(ddof=0),
            "day_of_week":    tomorrow.dayofweek,
            "month":          tomorrow.month,
            "day_of_month":   tomorrow.day,
            "week_of_year":   int(tomorrow.isocalendar().week),
            "is_weekend":     int(tomorrow.dayofweek >= 5),
            "is_month_end":   int(tomorrow.is_month_end),
        }

        X = pd.DataFrame([features])[FEATURE_COLS]
        prediction = max(0, float(self.pipeline.predict(X)[0]))

        # 80 % confidence interval via individual tree predictions
        rf        = self.pipeline.named_steps["rf"]
        X_scaled  = self.pipeline.named_steps["scaler"].transform(X)
        tree_preds = np.array([t.predict(X_scaled)[0] for t in rf.estimators_])

        return {
            "date":                tomorrow.strftime("%Y-%m-%d"),
            "predicted_visitors":  max(0, int(round(prediction))),
            "confidence_low":      max(0, int(np.percentile(tree_preds, 10))),
            "confidence_high":     int(np.percentile(tree_preds, 90)),
            "std_deviation":       round(float(tree_preds.std()), 1),
        }

    # ── Persist ───────────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        """Serialise the model bundle to disk with joblib."""
        if self.pipeline is None:
            raise RuntimeError("Nothing to save — train the model first.")
        out = Path(path) if path else MODEL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)
        bundle = {
            "pipeline":            self.pipeline,
            "feature_cols":        FEATURE_COLS,
            "target_col":          TARGET_COL,
            "config":              self.cfg,
            "cv_results":          self.cv_results,
            "test_metrics":        self.test_metrics,
            "feature_importances": self.feature_importances,
            "trained_at":          datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_version":       MODEL_VERSION,
        }
        joblib.dump(bundle, out, compress=3)
        log.info("Model saved → %s  (%.1f KB)", out, out.stat().st_size / 1024)
        return out

    @classmethod
    def load(cls, path: Path) -> "TrafficForecaster":
        """Load a previously saved model bundle."""
        bundle = joblib.load(path)
        obj = cls(config=bundle.get("config"))
        obj.pipeline             = bundle["pipeline"]
        obj.cv_results           = bundle["cv_results"]
        obj.test_metrics         = bundle["test_metrics"]
        obj.feature_importances  = bundle.get("feature_importances")
        log.info(
            "Model loaded ← %s  (v%s, trained %s)",
            path, bundle.get("model_version", "?"), bundle.get("trained_at", "?"),
        )
        return obj

    # ── Console report ────────────────────────────────────────────────────────
    def print_report(
        self,
        daily_df: pd.DataFrame,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        tomorrow_pred: Dict,
    ) -> None:
        """Print a formatted evaluation report to stdout."""
        W   = 72
        SEP = "═" * W
        DIV = "─" * W

        def hdr(title: str) -> None:
            print(f"\n  {title}\n  {DIV}")

        print(f"\n{SEP}")
        print(f"{'  TRAFFIC FORECAST MODEL  ·  Evaluation Report':^{W}}")
        print(f"{SEP}")

        # ── Dataset ───────────────────────────────────────────────────────────
        hdr("Dataset")
        print(f"  Total days in log    :  {len(daily_df)}")
        print(f"  Usable records       :  {len(X_train) + len(X_test)}  "
              f"(after lag feature creation)")
        print(f"  Training records     :  {len(X_train)}")
        print(f"  Hold-out records     :  {len(X_test)}  (last {self.cfg.holdout_days} days)")
        print(f"  Target variable      :  {TARGET_COL}  "
              f"(mean={daily_df[TARGET_COL].mean():.0f}  "
              f"std={daily_df[TARGET_COL].std():.0f})")

        # ── Cross-validation ──────────────────────────────────────────────────
        cv = self.cv_results
        if cv:
            hdr(f"Time-Series CV  ·  {self.cfg.cv_folds}-fold TimeSeriesSplit  (no future leakage)")
            cw = 8
            print(
                f"  {'Fold':>4}  {'Train':>6}  {'Val':>5}  "
                f"{'MAE':>{cw}}  {'RMSE':>{cw}}  {'R²':>{cw}}  {'MAPE':>{cw}}"
            )
            divline = f"  {'─'*4}  {'─'*6}  {'─'*5}  " + f"{'─'*cw}  " * 4
            print(divline)
            for _, row in cv["fold_details"].iterrows():
                print(
                    f"  {int(row['fold']):>4}  {int(row['train_size']):>6}  "
                    f"{int(row['val_size']):>5}  "
                    f"{row['mae']:>{cw}.2f}  {row['rmse']:>{cw}.2f}  "
                    f"{row['r2']:>{cw}.4f}  {row['mape']:>{cw}.2f}%"
                )
            print(divline)
            print(
                f"  {'Mean':>4}  {'':>6}  {'':>5}  "
                f"{cv['mae_mean']:>{cw}.2f}  {cv['rmse_mean']:>{cw}.2f}  "
                f"{cv['r2_mean']:>{cw}.4f}  {cv['mape_mean']:>{cw}.2f}%"
            )
            print(
                f"  {'±Std':>4}  {'':>6}  {'':>5}  "
                f"{cv['mae_std']:>{cw}.2f}  {cv['rmse_std']:>{cw}.2f}  "
                f"{cv['r2_std']:>{cw}.4f}  {cv['mape_std']:>{cw}.2f}%"
            )

        # ── Hold-out metrics ──────────────────────────────────────────────────
        m = self.test_metrics
        hdr(f"Hold-out Test Set  (Last {self.cfg.holdout_days} Days)")
        print(f"  MAE   :  {m['mae']:>8.2f} visitors/day  — mean absolute error")
        print(f"  RMSE  :  {m['rmse']:>8.2f} visitors/day  — root mean squared error")
        print(f"  R²    :  {m['r2']:>8.4f}               — variance explained")
        print(f"  MAPE  :  {m['mape']:>8.2f}%             — mean absolute % error")

        # ── Tomorrow's prediction ─────────────────────────────────────────────
        hdr("Tomorrow's Visitor Prediction")
        print(f"  Date               :  {tomorrow_pred['date']}")
        print(f"  Predicted Visitors :  {tomorrow_pred['predicted_visitors']:,}")
        print(f"  80% Interval       :  "
              f"{tomorrow_pred['confidence_low']:,} – "
              f"{tomorrow_pred['confidence_high']:,} visitors")
        print(f"  Std deviation      :  ± {tomorrow_pred['std_deviation']:,.0f} (across {self.cfg.n_estimators} trees)")

        # ── Feature importances ───────────────────────────────────────────────
        if self.feature_importances is not None:
            fi = self.feature_importances
            hdr("Feature Importances  (Mean Decrease in Impurity)")
            BAR = 28
            print(f"  {'#':>2}   {'Feature':<18}   {'Score':>7}   Bar")
            print(f"  {'─'*2}   {'─'*18}   {'─'*7}   {'─'*BAR}")
            for rank, (feat, score) in enumerate(fi.items(), 1):
                bar = "█" * max(1, int(round(score * BAR / fi.max())))
                print(f"  {rank:>2}.  {feat:<18}   {score:.4f}   {bar}")

        print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrated training pipeline
# ─────────────────────────────────────────────────────────────────────────────
def run_training_pipeline(
    daily_df: Optional[pd.DataFrame] = None,
    config: Optional[ForecasterConfig] = None,
    save_path: Optional[Path] = None,
) -> TrafficForecaster:
    """End-to-end: data → features → CV → train → evaluate → save.

    Parameters
    ----------
    daily_df   : DataFrame[date, visits]. If None, synthetic data is generated.
    config     : ForecasterConfig. Defaults to ForecasterConfig().
    save_path  : Output path for .pkl bundle.

    Returns
    -------
    Fitted TrafficForecaster instance.
    """
    cfg = config or ForecasterConfig()

    log.info("╔══ Traffic Forecaster pipeline start ═════════════════════════╗")

    # 1. Data source
    if daily_df is None:
        log.info("  [1/5] Generating synthetic 365-day traffic data …")
        daily_df = generate_daily_traffic(n_days=365, seed=cfg.random_state)
    else:
        log.info("  [1/5] Using provided daily traffic data (%d days) …", len(daily_df))

    # 2. Feature engineering
    log.info("  [2/5] Engineering features …")
    feat_df = engineer_features(daily_df)
    X = feat_df[FEATURE_COLS]
    y = feat_df[TARGET_COL]

    # 3. Temporal train/test split (last holdout_days held out)
    log.info("  [3/5] Temporal train/test split (last %d days as hold-out) …", cfg.holdout_days)
    X_train, X_test = X.iloc[: -cfg.holdout_days], X.iloc[-cfg.holdout_days :]
    y_train, y_test = y.iloc[: -cfg.holdout_days], y.iloc[-cfg.holdout_days :]
    log.info("        Train: %d records  |  Test: %d records", len(X_train), len(X_test))

    # 4. Cross-validate + train + evaluate
    log.info("  [4/5] Cross-validating and training …")
    forecaster = TrafficForecaster(cfg)
    forecaster.cross_validate(X_train, y_train)
    forecaster.train(X_train, y_train)
    forecaster.evaluate(X_test, y_test)

    # 5. Predict tomorrow + save
    log.info("  [5/5] Predicting tomorrow's traffic and saving model …")
    recent_14 = daily_df[TARGET_COL].tail(14).tolist()
    tomorrow  = forecaster.predict_tomorrow(recent_14)

    forecaster.print_report(daily_df, X_train, X_test, y_test, tomorrow)

    saved = forecaster.save(save_path)
    try:
        rel = saved.relative_to(Path.cwd())
    except ValueError:
        rel = saved
    print(f"  ✓  Model saved → {rel}\n")

    log.info("╚══ Traffic Forecaster pipeline complete ══════════════════════╝")
    return forecaster


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the traffic forecasting Random Forest Regressor."
    )
    parser.add_argument("--days",         type=int,   default=365,  help="Synthetic log span in days.")
    parser.add_argument("--n-estimators", type=int,   default=200,  help="Number of RF trees.")
    parser.add_argument("--cv-folds",     type=int,   default=5,    help="TimeSeriesSplit folds.")
    parser.add_argument("--holdout",      type=int,   default=30,   help="Hold-out days.")
    parser.add_argument("--seed",         type=int,   default=42,   help="Random seed.")
    parser.add_argument("--out",          default=None,             help="Output .pkl path.")
    args = parser.parse_args()

    cfg = ForecasterConfig(
        n_estimators  = args.n_estimators,
        cv_folds      = args.cv_folds,
        holdout_days  = args.holdout,
        random_state  = args.seed,
    )
    run_training_pipeline(
        config    = cfg,
        save_path = Path(args.out) if args.out else None,
    )


if __name__ == "__main__":
    main()
