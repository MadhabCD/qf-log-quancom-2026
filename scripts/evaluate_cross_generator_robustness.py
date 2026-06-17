#!/usr/bin/env python3
"""
Cross-generator robustness evaluation for the simulation-based QF-LOG study.

Scientific purpose
------------------
Apply the classifiers trained on Generator-A to Generator-B without any
retraining, tuning, calibration, or threshold adjustment using Generator-B.
This measures performance degradation under shifted simulated operational
conditions while keeping the same four observable QF-LOG features:
    qber, photon_count, latency_ms, abort_flag
and the same four class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Recommended final-run mode
--------------------------
The default mode loads the exact models and transparent-baseline rules saved
by the completed Generator-A holdout experiment. This preserves direct
comparability between the Generator-A holdout results and Generator-B results.

Optional recovery mode
----------------------
If the saved Generator-A models are not available, use --model-source retrain.
This recreates the same 75/25 stratified Generator-A partition (seed 42), fits
all methods on the Generator-A training partition only, evaluates the internal
holdout, and then applies the models to Generator-B.

Important limitation
--------------------
Generator-B is synthetic. This experiment measures cross-generator robustness
under shifted simulated conditions, not real-QKD hardware generalisation.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - dependent on user system
    raise ImportError(
        "XGBoost is required. Install it with: python -m pip install xgboost"
    ) from exc


FEATURES = ["qber", "photon_count", "latency_ms", "abort_flag"]
CLASS_LABELS = [
    "normal",
    "partial_intercept_resend",
    "detector_blind",
    "fiber_tap",
]
CLASS_TO_ID = {label: index for index, label in enumerate(CLASS_LABELS)}
ID_TO_CLASS = {index: label for label, index in CLASS_TO_ID.items()}
MODELS = [
    "Transparent_Threshold_Baseline",
    "Logistic_Regression",
    "Random_Forest",
    "XGBoost",
]
DISPLAY_NAMES = {
    "Transparent_Threshold_Baseline": "Transparent Threshold Baseline",
    "Logistic_Regression": "Logistic Regression",
    "Random_Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


@dataclass(frozen=True)
class EvaluationConfig:
    observable_features: tuple[str, ...] = tuple(FEATURES)
    class_labels: tuple[str, ...] = tuple(CLASS_LABELS)
    training_generator: str = "Generator-A"
    external_test_generator: str = "Generator-B"
    generator_a_training_fraction: float = 0.75
    generator_a_holdout_fraction: float = 0.25
    generator_a_partition_seed: int = 42
    external_test_policy: str = (
        "Generator-B is used only for external testing; no fitting or tuning is performed on it."
    )
    logistic_max_iter: int = 1000
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int = 14
    xgboost_n_estimators: int = 400
    xgboost_max_depth: int = 10
    xgboost_learning_rate: float = 0.08
    xgboost_subsample: float = 0.90
    xgboost_colsample_bytree: float = 0.90


class TransparentThresholdBaseline:
    """Training-derived transparent QF-LOG baseline used in the A experiment."""

    def __init__(self) -> None:
        self.thresholds_: Dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransparentThresholdBaseline":
        qber_idx = FEATURES.index("qber")
        photon_idx = FEATURES.index("photon_count")

        def median_for(label: str, feature_index: int) -> float:
            values = X[y == CLASS_TO_ID[label], feature_index]
            if values.size == 0:
                raise ValueError(f"Training data contain no samples for class: {label}")
            return float(np.median(values))

        normal_qber = median_for("normal", qber_idx)
        ir_qber = median_for("partial_intercept_resend", qber_idx)
        normal_photons = median_for("normal", photon_idx)
        tap_photons = median_for("fiber_tap", photon_idx)
        blind_photons = median_for("detector_blind", photon_idx)
        self.thresholds_ = {
            "detector_blind_photon_threshold": (blind_photons + tap_photons) / 2.0,
            "partial_intercept_resend_qber_threshold": (normal_qber + ir_qber) / 2.0,
            "fiber_tap_photon_threshold": (tap_photons + normal_photons) / 2.0,
            "normal_qber_training_median": normal_qber,
            "partial_intercept_resend_qber_training_median": ir_qber,
            "normal_photon_training_median": normal_photons,
            "fiber_tap_photon_training_median": tap_photons,
            "detector_blind_photon_training_median": blind_photons,
        }
        return self

    @classmethod
    def from_rules_file(cls, rules_path: Path) -> "TransparentThresholdBaseline":
        if not rules_path.exists():
            raise FileNotFoundError(f"Baseline rules file not found: {rules_path}")
        rules = json.loads(rules_path.read_text(encoding="utf-8"))
        required = {
            "detector_blind_photon_threshold",
            "partial_intercept_resend_qber_threshold",
            "fiber_tap_photon_threshold",
        }
        if not required.issubset(rules):
            raise ValueError("Saved baseline rules file is missing required thresholds.")
        model = cls()
        model.thresholds_ = {key: float(value) for key, value in rules.items()}
        return model

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Transparent threshold baseline has not been fitted.")
        qber = X[:, FEATURES.index("qber")]
        photons = X[:, FEATURES.index("photon_count")]
        pred = np.full(X.shape[0], CLASS_TO_ID["normal"], dtype=np.int8)

        is_blind = photons <= self.thresholds_["detector_blind_photon_threshold"]
        pred[is_blind] = CLASS_TO_ID["detector_blind"]

        unresolved = ~is_blind
        is_ir = unresolved & (
            qber >= self.thresholds_["partial_intercept_resend_qber_threshold"]
        )
        pred[is_ir] = CLASS_TO_ID["partial_intercept_resend"]

        unresolved = unresolved & ~is_ir
        is_tap = unresolved & (
            photons <= self.thresholds_["fiber_tap_photon_threshold"]
        )
        pred[is_tap] = CLASS_TO_ID["fiber_tap"]
        return pred


def parse_args() -> argparse.Namespace:
    default_root = r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"
    parser = argparse.ArgumentParser(
        description="Evaluate Generator-A-trained QF-LOG classifiers on Generator-B."
    )
    parser.add_argument("--root", type=Path, default=Path(default_root))
    parser.add_argument("--generator-a", type=Path, default=None)
    parser.add_argument("--generator-b", type=Path, default=None)
    parser.add_argument("--generator-b-metadata", type=Path, default=None)
    parser.add_argument(
        "--model-source",
        choices=("saved", "retrain"),
        default="saved",
        help=(
            "Default 'saved' applies the exact models from the completed A holdout "
            "experiment. Use 'retrain' only if those saved models are unavailable."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--skip-metadata-analysis",
        action="store_true",
        help="Skip Generator-B hardware-profile and temporal-window analysis.",
    )
    parser.add_argument(
        "--run-pca",
        action="store_true",
        help="Generate an optional standardised A-to-B PCA domain-shift projection.",
    )
    parser.add_argument(
        "--pca-samples-per-class",
        type=int,
        default=3000,
        help="Samples per class per generator used for optional PCA.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save 5,000,000 Generator-B predictions for each method as gzip CSV.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing cross_generator_evaluation output folder.",
    )
    return parser.parse_args()


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("qf_log_cross_generator")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def prepare_dirs(root: Path, overwrite: bool) -> Dict[str, Path]:
    out = root / "results" / "cross_generator_evaluation"
    if out.exists() and any(out.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {out}\n"
                "Use --overwrite only when you intentionally want to replace these outputs."
            )
        shutil.rmtree(out)
    dirs = {
        "root": out,
        "metrics": out / "metrics",
        "confusion": out / "confusion_matrices",
        "figures": out / "figures",
        "diagnostics": out / "diagnostics",
        "metadata": out / "metadata_analysis",
        "predictions": out / "predictions",
        "models": out / "models",
        "logs": out / "logs",
    }
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    return dirs


def load_qf_log(path: Path, name: str, logger: logging.Logger) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{name} dataset not found: {path}")
    logger.info("Loading %s dataset: %s", name, path)
    dtypes = {
        "qber": "float32",
        "photon_count": "int32",
        "latency_ms": "float32",
        "abort_flag": "int8",
        "label": "category",
    }
    df = pd.read_csv(path, usecols=FEATURES + ["label"], dtype=dtypes)
    if df.empty or df.isnull().any().any():
        raise ValueError(f"{name} dataset is empty or contains missing values.")
    labels = set(df["label"].astype(str).unique())
    if labels != set(CLASS_LABELS):
        raise ValueError(f"{name} labels do not match required labels: {sorted(labels)}")
    if not df["qber"].between(0.0, 0.5).all():
        raise ValueError(f"{name} QBER contains values outside [0, 0.5].")
    if not (df["photon_count"] >= 0).all():
        raise ValueError(f"{name} photon_count contains negative values.")
    if not (df["latency_ms"] >= 0).all():
        raise ValueError(f"{name} latency_ms contains negative values.")
    if not df["abort_flag"].isin([0, 1]).all():
        raise ValueError(f"{name} abort_flag is not binary.")
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].astype(str).map(CLASS_TO_ID).to_numpy(dtype=np.int8, copy=True)
    counts = df["label"].astype(str).value_counts().reindex(CLASS_LABELS, fill_value=0)
    logger.info("%s loaded: %s rows. Class counts: %s", name, f"{len(df):,}", counts.to_dict())
    return df, X, y


def create_models(seed: int, n_jobs: int, xgb_device: str) -> Dict[str, BaseEstimator]:
    xgb_kwargs: Dict[str, Any] = {
        "n_estimators": 400,
        "max_depth": 10,
        "learning_rate": 0.08,
        "subsample": 0.90,
        "colsample_bytree": 0.90,
        "objective": "multi:softprob",
        "num_class": len(CLASS_LABELS),
        "eval_metric": "mlogloss",
        "tree_method": "hist",
        "random_state": seed,
        "n_jobs": n_jobs,
    }
    if xgb_device == "cuda":
        xgb_kwargs["device"] = "cuda"
    return {
        "Logistic_Regression": Pipeline(
            [("scale", StandardScaler()),
             ("classifier", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed))]
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, max_depth=14, criterion="gini", bootstrap=True,
            n_jobs=n_jobs, random_state=seed,
        ),
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


def probability_metrics(y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    targets = label_binarize(y, classes=np.arange(len(CLASS_LABELS)))
    return {
        "macro_roc_auc_ovr": float(roc_auc_score(targets, probs, average="macro", multi_class="ovr")),
        "macro_pr_auc_ovr": float(average_precision_score(targets, probs, average="macro")),
    }


def metrics(y: np.ndarray, pred: np.ndarray, probs: np.ndarray | None = None) -> Dict[str, float]:
    result = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }
    if probs is not None:
        result.update(probability_metrics(y, probs))
    else:
        result.update({"macro_roc_auc_ovr": math.nan, "macro_pr_auc_ovr": math.nan})
    return result


def save_confusion_and_report(
    model_name: str, y: np.ndarray, pred: np.ndarray, dirs: Mapping[str, Path]
) -> None:
    cm = confusion_matrix(y, pred, labels=np.arange(len(CLASS_LABELS)))
    cm_df = pd.DataFrame(cm, index=CLASS_LABELS, columns=CLASS_LABELS)
    cm_df.to_csv(dirs["confusion"] / f"generator_b_{model_name.lower()}_confusion_matrix.csv")
    report = classification_report(
        y, pred, labels=np.arange(len(CLASS_LABELS)), target_names=CLASS_LABELS,
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).transpose().to_csv(
        dirs["metrics"] / f"generator_b_{model_name.lower()}_classification_report.csv"
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ConfusionMatrixDisplay(cm, display_labels=CLASS_LABELS).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False, xticks_rotation=35
    )
    ax.set_title(f"Generator-B Cross-Generator Confusion Matrix: {DISPLAY_NAMES[model_name]}")
    fig.tight_layout()
    fig.savefig(dirs["figures"] / f"generator_b_{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close(fig)


def load_saved_models(root: Path, logger: logging.Logger) -> Dict[str, Any]:
    result_root = root / "results" / "model_evaluation"
    model_dir = result_root / "models"
    baseline_rules = result_root / "metrics" / "transparent_threshold_baseline_rules.json"
    required_paths = [
        baseline_rules,
        model_dir / "logistic_regression_model.joblib",
        model_dir / "random_forest_model.joblib",
        model_dir / "xgboost_model.json",
    ]
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Saved Generator-A model outputs required for --model-source saved were not found:\n"
            + "\n".join(missing)
            + "\nRun with --model-source retrain only if you intentionally need to recreate them."
        )
    loaded: Dict[str, Any] = {
        "Transparent_Threshold_Baseline": TransparentThresholdBaseline.from_rules_file(baseline_rules),
        "Logistic_Regression": joblib.load(model_dir / "logistic_regression_model.joblib"),
        "Random_Forest": joblib.load(model_dir / "random_forest_model.joblib"),
    }
    xgb = XGBClassifier()
    xgb.load_model(model_dir / "xgboost_model.json")
    loaded["XGBoost"] = xgb
    logger.info("Loaded saved Generator-A methods from completed fixed-holdout evaluation.")
    return loaded


def retrain_models(
    X_a: np.ndarray, y_a: np.ndarray, args: argparse.Namespace,
    dirs: Mapping[str, Path], logger: logging.Logger
) -> tuple[Dict[str, Any], pd.DataFrame]:
    logger.warning(
        "Retrain mode selected. Methods will be trained on the Generator-A training partition only."
    )
    X_train, X_holdout, y_train, y_holdout = train_test_split(
        X_a, y_a, test_size=args.test_size, random_state=args.seed, stratify=y_a
    )
    methods: Dict[str, Any] = {
        "Transparent_Threshold_Baseline": TransparentThresholdBaseline().fit(X_train, y_train)
    }
    methods.update(create_models(args.seed, args.n_jobs, args.xgb_device))
    rows: list[Dict[str, Any]] = []
    for name, method in methods.items():
        if name != "Transparent_Threshold_Baseline":
            logger.info("Training %s on Generator-A training partition.", DISPLAY_NAMES[name])
            method.fit(X_train, y_train)
        pred = method.predict(X_holdout).astype(np.int8)
        probs = None if name == "Transparent_Threshold_Baseline" else method.predict_proba(X_holdout)
        rows.append({"model": name, **metrics(y_holdout, pred, probs)})
    holdout = pd.DataFrame(rows)
    holdout.to_csv(dirs["metrics"] / "generator_a_holdout_reference_metrics_retrained.csv", index=False)
    return methods, holdout


def load_reference_holdout(root: Path) -> pd.DataFrame:
    path = root / "results" / "model_evaluation" / "metrics" / "holdout_overall_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Completed Generator-A holdout metrics were not found: {path}\n"
            "Use --model-source retrain if the saved evaluation is unavailable."
        )
    df = pd.read_csv(path)
    required = {"model", "accuracy", "balanced_accuracy", "macro_f1", "mcc"}
    if not required.issubset(df.columns):
        raise ValueError("Generator-A holdout metrics file lacks required columns.")
    return df


def evaluate_generator_b(
    methods: Mapping[str, Any], X_b: np.ndarray, y_b: np.ndarray,
    b_df: pd.DataFrame, dirs: Mapping[str, Path], logger: logging.Logger,
    save_predictions: bool,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    rows: list[Dict[str, Any]] = []
    predictions: Dict[str, np.ndarray] = {}
    for name in MODELS:
        method = methods[name]
        logger.info("Applying Generator-A-trained %s to Generator-B.", DISPLAY_NAMES[name])
        started = time.perf_counter()
        pred = method.predict(X_b).astype(np.int8)
        probs = None if name == "Transparent_Threshold_Baseline" else method.predict_proba(X_b)
        elapsed = time.perf_counter() - started
        row = {"model": name, "test_dataset": "Generator-B", **metrics(y_b, pred, probs), "predict_seconds": elapsed}
        rows.append(row)
        predictions[name] = pred
        save_confusion_and_report(name, y_b, pred, dirs)
        if save_predictions:
            out = pd.DataFrame({"actual_label": b_df["label"].astype(str), "predicted_label": [ID_TO_CLASS[int(x)] for x in pred]})
            if probs is not None:
                for idx, label in enumerate(CLASS_LABELS):
                    out[f"probability_{label}"] = probs[:, idx]
            out.to_csv(
                dirs["predictions"] / f"generator_b_{name.lower()}_predictions.csv.gz",
                index=False, compression="gzip",
            )
        logger.info(
            "%s on Generator-B: Accuracy=%.6f Macro-F1=%.6f MCC=%.6f.",
            DISPLAY_NAMES[name], row["accuracy"], row["macro_f1"], row["mcc"]
        )
        del probs
    external = pd.DataFrame(rows)
    external.to_csv(dirs["metrics"] / "generator_b_external_performance.csv", index=False)
    return external, predictions


def degradation_table(reference_a: pd.DataFrame, external_b: pd.DataFrame, path: Path) -> pd.DataFrame:
    metric_columns = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
        "weighted_f1", "mcc", "macro_roc_auc_ovr", "macro_pr_auc_ovr",
    ]
    a = reference_a[["model"] + [m for m in metric_columns if m in reference_a.columns]].copy()
    b = external_b[["model"] + [m for m in metric_columns if m in external_b.columns]].copy()
    merged = a.merge(b, on="model", suffixes=("_generator_a_holdout", "_generator_b_external"))
    for metric in metric_columns:
        a_col = f"{metric}_generator_a_holdout"
        b_col = f"{metric}_generator_b_external"
        if a_col in merged.columns and b_col in merged.columns:
            merged[f"{metric}_absolute_change"] = merged[b_col] - merged[a_col]
            merged[f"{metric}_relative_change"] = np.where(
                merged[a_col].abs() > 1e-12,
                (merged[b_col] - merged[a_col]) / merged[a_col],
                np.nan,
            )
    merged.to_csv(path, index=False)
    return merged


def feature_shift_table(a: pd.DataFrame, b: pd.DataFrame, path: Path) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    groups = [("overall", a, b)] + [
        (label, a[a["label"].astype(str) == label], b[b["label"].astype(str) == label])
        for label in CLASS_LABELS
    ]
    for group_name, a_group, b_group in groups:
        for feature in FEATURES:
            mean_a = float(a_group[feature].mean())
            mean_b = float(b_group[feature].mean())
            sd_a = float(a_group[feature].std(ddof=0))
            sd_b = float(b_group[feature].std(ddof=0))
            pooled_sd = math.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0)
            rows.append({
                "class_or_scope": group_name,
                "feature": feature,
                "generator_a_mean": mean_a,
                "generator_a_std": sd_a,
                "generator_b_mean": mean_b,
                "generator_b_std": sd_b,
                "mean_difference_b_minus_a": mean_b - mean_a,
                "standardized_mean_difference": (mean_b - mean_a) / pooled_sd if pooled_sd > 0 else math.nan,
            })
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def mcnemar(y: np.ndarray, a: np.ndarray, b: np.ndarray, a_name: str, b_name: str) -> Dict[str, Any]:
    a_correct = a == y
    b_correct = b == y
    a_only = int(np.sum(a_correct & ~b_correct))
    b_only = int(np.sum(~a_correct & b_correct))
    discordant = a_only + b_only
    if discordant == 0:
        test, statistic, p_value = "No discordant predictions", 0.0, 1.0
    elif discordant <= 25:
        test, statistic = "Exact binomial McNemar", math.nan
        p_value = float(binomtest(a_only, discordant, 0.5, alternative="two-sided").pvalue)
    else:
        test = "McNemar chi-square with continuity correction"
        statistic = float((abs(a_only - b_only) - 1.0) ** 2 / discordant)
        p_value = float(chi2.sf(statistic, df=1))
    return {
        "model_a": a_name, "model_b": b_name,
        "model_a_only_correct": a_only, "model_b_only_correct": b_only,
        "discordant_predictions": discordant, "test": test,
        "statistic": statistic, "p_value": p_value,
    }


def external_mcnemar_tests(y_b: np.ndarray, pred: Mapping[str, np.ndarray], path: Path) -> pd.DataFrame:
    comparisons = []
    baseline = "Transparent_Threshold_Baseline"
    for model in ("Logistic_Regression", "Random_Forest", "XGBoost"):
        comparisons.append(mcnemar(y_b, pred[model], pred[baseline], model, baseline))
    comparisons.append(mcnemar(y_b, pred["Random_Forest"], pred["XGBoost"], "Random_Forest", "XGBoost"))
    out = pd.DataFrame(comparisons)
    out.to_csv(path, index=False)
    return out


def metadata_subgroup_evaluation(
    metadata_path: Path, b_df: pd.DataFrame, y_b: np.ndarray,
    predictions: Mapping[str, np.ndarray], dirs: Mapping[str, Path], logger: logging.Logger
) -> None:
    if not metadata_path.exists():
        logger.warning("Generator-B metadata not found; skipping subgroup analysis: %s", metadata_path)
        return
    logger.info("Loading Generator-B metadata for profile and temporal-window diagnostics: %s", metadata_path)
    meta = pd.read_csv(
        metadata_path,
        usecols=["session_id", "time_step", "hardware_profile", "label"],
        dtype={"session_id": "int64", "time_step": "int32", "hardware_profile": "category", "label": "category"},
        compression="gzip",
    )
    if len(meta) != len(b_df):
        raise ValueError("Generator-B metadata and ML dataset row counts do not match.")
    if not np.array_equal(meta["label"].astype(str).to_numpy(), b_df["label"].astype(str).to_numpy()):
        raise ValueError("Generator-B metadata and ML dataset labels are not aligned by row order.")
    meta["time_window"] = pd.cut(
        meta["time_step"], bins=[-1, 99, 199, 299, 399, 499],
        labels=["frames_000_099", "frames_100_199", "frames_200_299", "frames_300_399", "frames_400_499"],
    )
    for grouping, filename in (("hardware_profile", "performance_by_hardware_profile.csv"), ("time_window", "performance_by_temporal_window.csv")):
        rows: list[Dict[str, Any]] = []
        for group_value, indexes in meta.groupby(grouping, observed=True).groups.items():
            idx = np.asarray(list(indexes), dtype=np.int64)
            for name in MODELS:
                rows.append({
                    grouping: str(group_value), "model": name, "samples": int(len(idx)),
                    **metrics(y_b[idx], predictions[name][idx]),
                })
        pd.DataFrame(rows).to_csv(dirs["metadata"] / filename, index=False)
    logger.info("Saved Generator-B hardware-profile and temporal-window diagnostics.")


def optional_pca(
    a: pd.DataFrame, b: pd.DataFrame, dirs: Mapping[str, Path], samples_per_class: int, seed: int
) -> None:
    rng = np.random.default_rng(seed)
    a_parts, b_parts = [], []
    for label in CLASS_LABELS:
        a_sub = a[a["label"].astype(str) == label]
        b_sub = b[b["label"].astype(str) == label]
        n_a, n_b = min(samples_per_class, len(a_sub)), min(samples_per_class, len(b_sub))
        a_parts.append(a_sub.iloc[rng.choice(len(a_sub), n_a, replace=False)].copy())
        b_parts.append(b_sub.iloc[rng.choice(len(b_sub), n_b, replace=False)].copy())
    a_s = pd.concat(a_parts, ignore_index=True)
    b_s = pd.concat(b_parts, ignore_index=True)
    scaler = StandardScaler().fit(a_s[FEATURES])
    pca = PCA(n_components=2, random_state=seed).fit(scaler.transform(a_s[FEATURES]))
    a_pc = pca.transform(scaler.transform(a_s[FEATURES]))
    b_pc = pca.transform(scaler.transform(b_s[FEATURES]))
    loadings = pd.DataFrame(pca.components_.T, index=FEATURES, columns=["PC1", "PC2"])
    loadings.to_csv(dirs["diagnostics"] / "pca_standardised_generator_a_loadings.csv")
    pd.DataFrame({"component": ["PC1", "PC2"], "explained_variance_ratio": pca.explained_variance_ratio_}).to_csv(
        dirs["diagnostics"] / "pca_standardised_explained_variance.csv", index=False
    )
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(a_pc[:, 0], a_pc[:, 1], s=4, alpha=0.18, label="Generator-A")
    ax.scatter(b_pc[:, 0], b_pc[:, 1], s=4, alpha=0.18, label="Generator-B")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%})")
    ax.set_title("Standardised PCA Projection of Generator-A and Generator-B")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dirs["figures"] / "standardised_pca_generator_a_vs_generator_b.png", dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    root = args.root
    a_path = args.generator_a or (root / "data" / "qf_log_dataset.csv")
    b_path = args.generator_b or (root / "data" / "generator_b_qf_log_dataset.csv")
    b_metadata = args.generator_b_metadata or (root / "data" / "generator_b_full_metadata.csv.gz")
    dirs = prepare_dirs(root, args.overwrite)
    logger = setup_logger(dirs["logs"] / "cross_generator_evaluation_log.txt")
    config = EvaluationConfig(generator_a_partition_seed=args.seed, generator_a_holdout_fraction=args.test_size, generator_a_training_fraction=1.0 - args.test_size)
    (dirs["metrics"] / "cross_generator_evaluation_configuration.json").write_text(
        json.dumps({**asdict(config), "model_source": args.model_source}, indent=2), encoding="utf-8"
    )

    logger.info("Cross-generator protocol: train/fitted on Generator-A only; test on Generator-B only.")
    a_df, X_a, y_a = load_qf_log(a_path, "Generator-A", logger)
    b_df, X_b, y_b = load_qf_log(b_path, "Generator-B", logger)

    shift = feature_shift_table(a_df, b_df, dirs["diagnostics"] / "generator_a_vs_b_observable_feature_shift.csv")
    logger.info("Saved observable feature-distribution shift table.")

    if args.model_source == "saved":
        methods = load_saved_models(root, logger)
        reference_a = load_reference_holdout(root)
    else:
        methods, reference_a = retrain_models(X_a, y_a, args, dirs, logger)

    external_b, pred = evaluate_generator_b(methods, X_b, y_b, b_df, dirs, logger, args.save_predictions)
    degradation = degradation_table(
        reference_a, external_b, dirs["metrics"] / "generator_a_holdout_vs_generator_b_degradation.csv"
    )
    mcnemar_results = external_mcnemar_tests(
        y_b, pred, dirs["metrics"] / "generator_b_mcnemar_pairwise_tests.csv"
    )

    if not args.skip_metadata_analysis:
        metadata_subgroup_evaluation(b_metadata, b_df, y_b, pred, dirs, logger)
    if args.run_pca:
        optional_pca(a_df, b_df, dirs, args.pca_samples_per_class, args.seed)
        logger.info("Saved optional standardised PCA diagnostics.")

    logger.info("Generator-B external performance summary:\n%s", external_b.to_string(index=False))
    display_cols = ["model", "accuracy_generator_a_holdout", "accuracy_generator_b_external", "accuracy_absolute_change", "macro_f1_generator_a_holdout", "macro_f1_generator_b_external", "macro_f1_absolute_change"]
    existing_display_cols = [col for col in display_cols if col in degradation.columns]
    logger.info("A-holdout versus B-external degradation summary:\n%s", degradation[existing_display_cols].to_string(index=False))
    logger.info("Generator-B McNemar comparisons:\n%s", mcnemar_results.to_string(index=False))
    logger.info("Cross-generator evaluation completed. Outputs: %s", dirs["root"])

    print("\nCross-generator evaluation completed. Generator-B external performance:")
    print(external_b.to_string(index=False))
    print("\nGenerator-A holdout versus Generator-B degradation:")
    print(degradation[existing_display_cols].to_string(index=False))
    print(f"\nResults saved to: {dirs['root']}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
