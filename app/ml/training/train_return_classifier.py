"""
train_return_classifier.py
==========================
Random Forest classifier that predicts whether a web user will return
to the site within a defined future time window.

Project path : backend/app/ml/training/train_return_classifier.py
Saves model  : backend/app/ml/models/return_user_classifier.pkl

Pipeline overview
-----------------
  Raw CSV log
      │
      ├─ create_return_labels()   ← temporal window split (no leakage)
      │       │                     Feature window  : days  0 → 45
      │       │                     Prediction window: days 46 → 60
      │       ↓
      │   will_return labels  (1 if user reappears in predict window)
      │
      └─ WebLogPreprocessor.run() ← runs on feature-window data only
              ↓
          5-feature matrix per user
                │
                ↓
          Merge features + labels
                │
         ┌──────┴──────┐
      X_train        X_test   ← stratified 75 / 25 split
         │
   StratifiedKFold CV (5 folds)  → cross-validation metrics
         │
   RandomForestClassifier.fit()  → trained model
         │
    Evaluate on X_test           → accuracy / precision / recall / F1 / AUC
         │
     joblib.dump()               → return_user_classifier.pkl

Usage
-----
    # From project root (bash):
    python -m backend.app.ml.training.train_return_classifier

    # Standalone (bash, from the training/ directory):
    python train_return_classifier.py

    # As module:
    from backend.app.ml.training.train_return_classifier import ReturnUserClassifier
    clf = ReturnUserClassifier.load("path/to/return_user_classifier.pkl")
    proba = clf.predict_proba(new_features_df)   # → numpy array of return probs
"""

from __future__ import annotations

import logging
import os
import sys
import tempfile
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ── Path setup: allow importing sibling `pipelines` package ──────────────────
_ML_DIR = Path(__file__).resolve().parent.parent   # → backend/app/ml/
if str(_ML_DIR) not in sys.path:
    sys.path.insert(0, str(_ML_DIR))

from pipelines.preprocessing_pipeline import (   # noqa: E402
    PipelineConfig,
    WebLogPreprocessor,
    generate_sample_log,
)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("rf_classifier")

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_COLS: list[str] = [
    "session_duration",
    "page_views",
    "visit_frequency",
    "bounce_rate",
    "navigation_depth",
]
TARGET_COL   = "will_return"
MODEL_DIR    = _ML_DIR / "models"
MODEL_PATH   = MODEL_DIR / "return_user_classifier.pkl"
MODEL_VERSION = "1.0.0"


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class ClassifierConfig:
    """Hyperparameters and pipeline settings for ReturnUserClassifier.

    Attributes
    ----------
    n_estimators      : number of trees in the Random Forest
    max_depth         : max tree depth  (None = grow until pure)
    min_samples_leaf  : min samples required at a leaf node
    max_features      : feature-selection strategy per split
    class_weight      : "balanced" compensates for class imbalance
    cv_folds          : number of StratifiedKFold folds
    test_size         : fraction of users held out for final evaluation
    random_state      : global random seed for reproducibility
    train_window_days : length of the feature-computation window in days
    predict_window_days: length of the label window that follows
    """
    # Random Forest
    n_estimators:       int   = 200
    max_depth:          Optional[int] = None
    min_samples_leaf:   int   = 2
    max_features:       str   = "sqrt"
    class_weight:       str   = "balanced"
    # Evaluation
    cv_folds:           int   = 5
    test_size:          float = 0.25
    random_state:       int   = 42
    # Temporal label window
    train_window_days:  int   = 45
    predict_window_days: int  = 15


# ─────────────────────────────────────────────────────────────────────────────
# Label engineering  (temporal window approach — no feature leakage)
# ─────────────────────────────────────────────────────────────────────────────
def create_return_labels(
    raw_df: pd.DataFrame,
    config: ClassifierConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Derive binary will_return labels using a temporal-window approach.

    The raw log is split at ``train_cutoff``:

        Feature window  : [min_date,    train_cutoff]   → compute ML features here
        Prediction window: (train_cutoff, predict_cutoff] → observe who returns here

    A user is labelled **will_return = 1** if and only if they appear in BOTH
    the feature window AND the prediction window.  Computing features only from
    the feature window guarantees there is no temporal leakage.

    Parameters
    ----------
    raw_df : pd.DataFrame
        Full raw access log (all 60 days).
    config : ClassifierConfig

    Returns
    -------
    labels        : DataFrame[user_id, will_return]
    train_df      : Rows from the feature window only — feed this to the preprocessor
    train_cutoff  : pd.Timestamp separating the two windows
    """
    df = raw_df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])

    min_ts         = df["timestamp"].min()
    train_cutoff   = min_ts + pd.Timedelta(days=config.train_window_days)
    predict_cutoff = train_cutoff + pd.Timedelta(days=config.predict_window_days)

    train_df   = df[df["timestamp"] <= train_cutoff].copy()
    predict_df = df[
        (df["timestamp"] > train_cutoff) & (df["timestamp"] <= predict_cutoff)
    ]

    train_users   = set(train_df["ip_address"].unique())
    predict_users = set(predict_df["ip_address"].unique())

    labels = pd.DataFrame({"user_id": sorted(train_users)})
    labels[TARGET_COL] = labels["user_id"].isin(predict_users).astype(int)

    pos = int(labels[TARGET_COL].sum())
    neg = len(labels) - pos
    log.info(
        "Labels created | %d users | will_return=1: %d (%.1f%%) | will_return=0: %d (%.1f%%)",
        len(labels), pos, 100 * pos / len(labels), neg, 100 * neg / len(labels),
    )
    return labels, train_df, train_cutoff


# ─────────────────────────────────────────────────────────────────────────────
# Classifier
# ─────────────────────────────────────────────────────────────────────────────
class ReturnUserClassifier:
    """Random Forest pipeline for predicting user return visits.

    Internal sklearn pipeline:  StandardScaler → RandomForestClassifier.

    (StandardScaler does not improve RF prediction quality but makes the
    pipeline trivially swappable with a distance-based or linear model
    without changing the serving code.)

    Parameters
    ----------
    config : ClassifierConfig, optional
        Hyperparameter settings.  Defaults to ``ClassifierConfig()`` if omitted.

    Examples
    --------
    Train from scratch via the CLI entry-point::

        python train_return_classifier.py

    Load a saved model and run inference::

        clf = ReturnUserClassifier.load("return_user_classifier.pkl")
        proba = clf.predict_proba(features_df)   # → [0.12, 0.87, …]
    """

    def __init__(self, config: Optional[ClassifierConfig] = None) -> None:
        self.cfg                = config or ClassifierConfig()
        self.pipeline: Optional[Pipeline] = None
        self.cv_results:        Dict = {}
        self.test_metrics:      Dict = {}
        self.feature_importances: Optional[pd.Series] = None

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _build_pipeline(self) -> Pipeline:
        """Construct a fresh (unfitted) sklearn Pipeline."""
        return Pipeline([
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(
                n_estimators    = self.cfg.n_estimators,
                max_depth       = self.cfg.max_depth,
                min_samples_leaf= self.cfg.min_samples_leaf,
                max_features    = self.cfg.max_features,
                class_weight    = self.cfg.class_weight,
                random_state    = self.cfg.random_state,
                n_jobs          = -1,
            )),
        ])

    # ── Cross-validation ──────────────────────────────────────────────────────
    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, np.ndarray]:
        """Run StratifiedKFold cross-validation on the training split.

        Scoring metrics: accuracy, precision (macro), recall (macro),
        F1 (macro), ROC-AUC.

        Parameters
        ----------
        X : pd.DataFrame  — training features  (FEATURE_COLS only)
        y : pd.Series     — training labels     (0 / 1)

        Returns
        -------
        dict mapping metric name → array of per-fold scores
        """
        cv = StratifiedKFold(
            n_splits    = self.cfg.cv_folds,
            shuffle     = True,
            random_state= self.cfg.random_state,
        )
        scoring = {
            "accuracy":  "accuracy",
            "precision": "precision_macro",
            "recall":    "recall_macro",
            "f1":        "f1_macro",
            "roc_auc":   "roc_auc",
        }
        log.info(
            "Running %d-fold StratifiedKFold cross-validation …",
            self.cfg.cv_folds,
        )
        raw = cross_validate(
            self._build_pipeline(), X, y,
            cv             = cv,
            scoring        = scoring,
            return_train_score = False,
            n_jobs         = -1,
        )
        self.cv_results = {
            k.replace("test_", ""): v
            for k, v in raw.items()
            if k.startswith("test_")
        }
        log.info(
            "CV complete | Acc %.4f ± %.4f | F1 %.4f ± %.4f | AUC %.4f ± %.4f",
            self.cv_results["accuracy"].mean(),  self.cv_results["accuracy"].std(),
            self.cv_results["f1"].mean(),         self.cv_results["f1"].std(),
            self.cv_results["roc_auc"].mean(),    self.cv_results["roc_auc"].std(),
        )
        return self.cv_results

    # ── Train ─────────────────────────────────────────────────────────────────
    def train(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit the full sklearn pipeline on the training set.

        Also extracts and stores feature importance scores (MDI) from
        the fitted Random Forest for later reporting.
        """
        log.info("Training Random Forest on %d samples …", len(X_train))
        self.pipeline = self._build_pipeline()
        self.pipeline.fit(X_train[FEATURE_COLS], y_train)

        rf = self.pipeline.named_steps["rf"]
        self.feature_importances = pd.Series(
            rf.feature_importances_,
            index=FEATURE_COLS,
        ).sort_values(ascending=False)
        log.info(
            "Training complete | top feature: %s (%.4f)",
            self.feature_importances.index[0],
            self.feature_importances.iloc[0],
        )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> Dict:
        """Compute the full metric suite on the hold-out test set.

        Metrics computed
        ----------------
        accuracy, precision (macro + weighted), recall (macro + weighted),
        F1-score (macro + weighted), ROC-AUC, confusion matrix,
        full sklearn classification report.

        Returns
        -------
        dict with all metric values (also stored in self.test_metrics).
        """
        if self.pipeline is None:
            raise RuntimeError("Call train() before evaluate().")

        y_pred = self.pipeline.predict(X_test[FEATURE_COLS])
        y_prob = self.pipeline.predict_proba(X_test[FEATURE_COLS])[:, 1]

        self.test_metrics = {
            "accuracy":             round(float(accuracy_score(y_test, y_pred)), 4),
            "precision_macro":      round(float(precision_score(y_test, y_pred, average="macro",    zero_division=0)), 4),
            "recall_macro":         round(float(recall_score(   y_test, y_pred, average="macro",    zero_division=0)), 4),
            "f1_macro":             round(float(f1_score(       y_test, y_pred, average="macro",    zero_division=0)), 4),
            "precision_weighted":   round(float(precision_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
            "recall_weighted":      round(float(recall_score(   y_test, y_pred, average="weighted", zero_division=0)), 4),
            "f1_weighted":          round(float(f1_score(       y_test, y_pred, average="weighted", zero_division=0)), 4),
            "roc_auc":              round(float(roc_auc_score(y_test, y_prob)), 4),
            "confusion_matrix":     confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=["No Return (0)", "Will Return (1)"],
                zero_division=0,
            ),
        }
        log.info(
            "Test evaluation | Acc %.4f | F1 %.4f | AUC %.4f",
            self.test_metrics["accuracy"],
            self.test_metrics["f1_macro"],
            self.test_metrics["roc_auc"],
        )
        return self.test_metrics

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        """Return binary predictions (0 = won't return, 1 = will return)."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.pipeline.predict(features[FEATURE_COLS])

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return return-probability score for each user  (0.0 – 1.0)."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained or loaded.")
        return self.pipeline.predict_proba(features[FEATURE_COLS])[:, 1]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save(self, path: Optional[Path] = None) -> Path:
        """Serialise the full model bundle to a .pkl file with joblib.

        Bundle contains: pipeline, feature names, config, CV results,
        test metrics, feature importances, training timestamp, version.
        """
        if self.pipeline is None:
            raise RuntimeError("Nothing to save — train the model first.")
        out = Path(path) if path else MODEL_PATH
        out.parent.mkdir(parents=True, exist_ok=True)

        bundle = {
            "pipeline":             self.pipeline,
            "feature_cols":         FEATURE_COLS,
            "target_col":           TARGET_COL,
            "config":               self.cfg,
            "cv_results":           self.cv_results,
            "test_metrics":         self.test_metrics,
            "feature_importances":  self.feature_importances,
            "trained_at":           datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "model_version":        MODEL_VERSION,
        }
        joblib.dump(bundle, out, compress=3)
        size_kb = out.stat().st_size / 1024
        log.info("Model saved → %s  (%.1f KB)", out, size_kb)
        return out

    @classmethod
    def load(cls, path: Path) -> "ReturnUserClassifier":
        """Load a previously saved model bundle.

        Parameters
        ----------
        path : Path  — path to the .pkl file produced by save().
        """
        bundle = joblib.load(path)
        obj = cls(config=bundle["config"])
        obj.pipeline              = bundle["pipeline"]
        obj.cv_results            = bundle["cv_results"]
        obj.test_metrics          = bundle["test_metrics"]
        obj.feature_importances   = bundle["feature_importances"]
        log.info(
            "Model loaded ← %s  (v%s, trained %s)",
            path, bundle.get("model_version", "?"), bundle.get("trained_at", "?"),
        )
        return obj

    # ── Console report ────────────────────────────────────────────────────────
    def print_report(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_test: pd.Series,
    ) -> None:
        """Print a formatted evaluation report to stdout.

        Sections: dataset summary · CV per-fold table · hold-out metrics ·
        classification report · confusion matrix (raw + normalised) ·
        feature importances with ASCII bar chart.
        """

        W   = 68
        SEP = "═" * W
        DIV = "─" * W

        def hdr(title: str) -> None:
            print(f"\n  {title}")
            print(f"  {DIV}")

        # ── Header ────────────────────────────────────────────────────────────
        print(f"\n{SEP}")
        print(f"{'  RETURN-USER CLASSIFIER  ·  Evaluation Report':^{W}}")
        print(f"{SEP}")

        # ── Dataset ───────────────────────────────────────────────────────────
        total = len(y)
        pos   = int(y.sum())
        neg   = total - pos
        hdr("Dataset")
        print(f"  Total users (labelled)  :  {total}")
        print(f"  will_return = 1  (pos)  :  {pos:4d}   ({100*pos/total:.1f}%)")
        print(f"  will_return = 0  (neg)  :  {neg:4d}   ({100*neg/total:.1f}%)")
        print(f"  Training set            :  {len(X_train)} users")
        print(f"  Test set                :  {len(X_test)}  users")

        # ── Cross-validation per-fold table ───────────────────────────────────
        cv = self.cv_results
        if cv:
            hdr(f"Cross-Validation  ·  {self.cfg.cv_folds}-fold StratifiedKFold  (on training set)")
            col_w = 9
            print(
                f"  {'Fold':>4}  "
                f"{'Accuracy':>{col_w}}  "
                f"{'Precision':>{col_w}}  "
                f"{'Recall':>{col_w}}  "
                f"{'F1-Score':>{col_w}}  "
                f"{'ROC-AUC':>{col_w}}"
            )
            print(f"  {'─'*4}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}")
            n_folds = len(cv["accuracy"])
            for i in range(n_folds):
                print(
                    f"  {i+1:>4}  "
                    f"{cv['accuracy'][i]:>{col_w}.4f}  "
                    f"{cv['precision'][i]:>{col_w}.4f}  "
                    f"{cv['recall'][i]:>{col_w}.4f}  "
                    f"{cv['f1'][i]:>{col_w}.4f}  "
                    f"{cv['roc_auc'][i]:>{col_w}.4f}"
                )
            print(f"  {'─'*4}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}")
            print(
                f"  {'Mean':>4}  "
                f"{cv['accuracy'].mean():>{col_w}.4f}  "
                f"{cv['precision'].mean():>{col_w}.4f}  "
                f"{cv['recall'].mean():>{col_w}.4f}  "
                f"{cv['f1'].mean():>{col_w}.4f}  "
                f"{cv['roc_auc'].mean():>{col_w}.4f}"
            )
            print(
                f"  {'±Std':>4}  "
                f"{cv['accuracy'].std():>{col_w}.4f}  "
                f"{cv['precision'].std():>{col_w}.4f}  "
                f"{cv['recall'].std():>{col_w}.4f}  "
                f"{cv['f1'].std():>{col_w}.4f}  "
                f"{cv['roc_auc'].std():>{col_w}.4f}"
            )

        # ── Hold-out test metrics ─────────────────────────────────────────────
        m = self.test_metrics
        hdr("Hold-out Test Set Metrics")
        print(f"  Accuracy              :  {m['accuracy']:.4f}")
        print(f"  Precision  (macro)    :  {m['precision_macro']:.4f}")
        print(f"  Recall     (macro)    :  {m['recall_macro']:.4f}")
        print(f"  F1-Score   (macro)    :  {m['f1_macro']:.4f}")
        print(f"  Precision  (weighted) :  {m['precision_weighted']:.4f}")
        print(f"  Recall     (weighted) :  {m['recall_weighted']:.4f}")
        print(f"  F1-Score   (weighted) :  {m['f1_weighted']:.4f}")
        print(f"  ROC-AUC               :  {m['roc_auc']:.4f}")

        # ── Per-class classification report ───────────────────────────────────
        hdr("Per-class Classification Report")
        # Indent each line
        for line in m["classification_report"].splitlines():
            print(f"  {line}")

        # ── Confusion matrix ──────────────────────────────────────────────────
        cm   = np.array(m["confusion_matrix"])
        tn, fp, fn, tp = cm.ravel()
        hdr("Confusion Matrix  (rows = Actual · columns = Predicted)")
        print(f"                       Pred 0     Pred 1")
        print(f"  Actual 0  (neg)  →  [{tn:6d}     {fp:6d} ]   TN | FP")
        print(f"  Actual 1  (pos)  →  [{fn:6d}     {tp:6d} ]   FN | TP")
        print()
        # Normalised (row-wise)
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        print(f"  Normalised  (row-share of true class):")
        print(f"                       Pred 0     Pred 1")
        print(f"  Actual 0         →  [ {cm_norm[0,0]:.3f}      {cm_norm[0,1]:.3f} ]")
        print(f"  Actual 1         →  [ {cm_norm[1,0]:.3f}      {cm_norm[1,1]:.3f} ]")

        # ── Feature importances (ASCII bar chart) ─────────────────────────────
        if self.feature_importances is not None:
            fi = self.feature_importances
            hdr("Feature Importances  (Mean Decrease in Impurity — MDI)")
            BAR_MAX = 28
            print(f"  {'#':>2}   {'Feature':<22}   {'Score':>7}   Bar")
            print(f"  {'─'*2}   {'─'*22}   {'─'*7}   {'─'*BAR_MAX}")
            for rank, (feat, score) in enumerate(fi.items(), start=1):
                bar_len = max(1, int(round(score * BAR_MAX / fi.max())))
                bar     = "█" * bar_len
                print(f"  {rank:>2}.  {feat:<22}   {score:.4f}   {bar}")

        # ── Footer ────────────────────────────────────────────────────────────
        print(f"\n{SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Orchestrated entry-point
# ─────────────────────────────────────────────────────────────────────────────
def run_training_pipeline(
    csv_path: Optional[str | Path] = None,
    save_path: Optional[Path]      = None,
    config: Optional[ClassifierConfig] = None,
) -> ReturnUserClassifier:
    """Full training pipeline: data → labels → features → train → evaluate → save.

    Parameters
    ----------
    csv_path  : path to a raw access-log CSV.  If None, synthetic data is generated.
    save_path : where to write the .pkl bundle.  Defaults to MODEL_PATH.
    config    : ClassifierConfig to use.        Defaults to ClassifierConfig().

    Returns
    -------
    Fitted ReturnUserClassifier instance.
    """
    cfg = config or ClassifierConfig()

    # ── 1. Source data ────────────────────────────────────────────────────────
    log.info("╔══ Training pipeline start ════════════════════════════════╗")
    if csv_path is None:
        log.info("  [1/6] Generating synthetic 60-day access log …")
        raw_df = generate_sample_log(n_days=60, n_users=200, seed=cfg.random_state)
    else:
        log.info("  [1/6] Loading access log from '%s' …", csv_path)
        raw_df = pd.read_csv(csv_path, low_memory=False)
    log.info("        Raw log: %d rows", len(raw_df))

    # ── 2. Temporal labels ────────────────────────────────────────────────────
    log.info("  [2/6] Creating temporal will_return labels …")
    labels, train_window_df, cutoff = create_return_labels(raw_df, cfg)
    log.info("        Feature window  :  up to %s", cutoff.date())
    log.info("        Predict window  :  next %d days", cfg.predict_window_days)

    # ── 3. Preprocess feature window ──────────────────────────────────────────
    log.info("  [3/6] Running preprocessing pipeline (feature window only) …")
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False, prefix="train_window_"
    ) as f:
        train_window_df.to_csv(f, index=False)
        tmp_path = f.name

    preprocessor = WebLogPreprocessor(PipelineConfig())
    features = preprocessor.run(tmp_path)
    os.unlink(tmp_path)
    log.info("        Features: %d users × %d columns", *features.shape)

    # ── 4. Merge features + labels ────────────────────────────────────────────
    log.info("  [4/6] Merging features with labels …")
    dataset = (
        features
        .merge(labels, on="user_id", how="inner")
        .dropna(subset=FEATURE_COLS + [TARGET_COL])
        .reset_index(drop=True)
    )
    log.info("        Final labelled dataset: %d users", len(dataset))

    X = dataset[FEATURE_COLS]
    y = dataset[TARGET_COL].astype(int)

    # ── 5. Stratified train / test split ──────────────────────────────────────
    log.info(
        "  [5/6] Stratified train/test split  (%.0f%% / %.0f%%) …",
        (1 - cfg.test_size) * 100, cfg.test_size * 100,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size    = cfg.test_size,
        stratify     = y,
        random_state = cfg.random_state,
    )
    log.info(
        "        Train: %d  |  Test: %d  |  Pos-rate train: %.1f%%  test: %.1f%%",
        len(X_train), len(X_test),
        100 * y_train.mean(), 100 * y_test.mean(),
    )

    # ── 6. Cross-validate → train → evaluate → save ───────────────────────────
    log.info("  [6/6] Cross-validating, training, evaluating …")
    clf = ReturnUserClassifier(cfg)
    clf.cross_validate(X_train, y_train)
    clf.train(X_train, y_train)
    clf.evaluate(X_test, y_test)

    # Print the full report
    clf.print_report(X, y, X_train, X_test, y_test)

    # Persist
    saved = clf.save(save_path)
    try:
        rel = saved.relative_to(Path.cwd())
    except ValueError:
        rel = saved
    print(f"  ✓  Model saved  →  {rel}")
    print(f"  ✓  Keys in bundle: pipeline · feature_cols · config · "
          f"cv_results · test_metrics · feature_importances · trained_at\n")

    log.info("╚══ Training pipeline complete ═════════════════════════════╝")
    return clf


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Train the return-user Random Forest classifier."
    )
    parser.add_argument(
        "--csv", default=None,
        help="Path to a raw access-log CSV.  Omit to use synthetic data.",
    )
    parser.add_argument(
        "--out", default=None,
        help=f"Output path for the .pkl bundle.  Default: {MODEL_PATH}",
    )
    parser.add_argument(
        "--n-estimators", type=int, default=200,
        help="Number of trees in the Random Forest  (default 200).",
    )
    parser.add_argument(
        "--cv-folds", type=int, default=5,
        help="Number of StratifiedKFold folds  (default 5).",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.25,
        help="Fraction of users for hold-out test  (default 0.25).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed  (default 42).",
    )
    args = parser.parse_args()

    cfg = ClassifierConfig(
        n_estimators = args.n_estimators,
        cv_folds     = args.cv_folds,
        test_size    = args.test_size,
        random_state = args.seed,
    )
    run_training_pipeline(
        csv_path  = args.csv,
        save_path = Path(args.out) if args.out else None,
        config    = cfg,
    )


if __name__ == "__main__":
    main()
