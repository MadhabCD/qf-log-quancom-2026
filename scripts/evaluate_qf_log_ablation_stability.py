#!/usr/bin/env python3
"""
Extended validation for the simulation-based, physics-inspired QF-LOG dataset.

This script completes the analyses not included in the first fixed-holdout run:
    1) leave-one-feature-out ablation for every trained ML model;
    2) single-feature evaluation for every trained ML model; and
    3) multi-seed stratified 5-fold validation for the threshold baseline and
       every trained ML model.

Observable ML features only:
    qber, photon_count, latency_ms, abort_flag

Class labels only:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Methodological note:
    Feature ablation and single-feature evaluation apply to the trained ML
    classifiers. The transparent threshold baseline is evaluated in repeated
    validation, but it is not included in ablation because removing qber or
    photon_count changes its explicitly defined forensic rule set rather than
    ablating a learned model input.

By default, analyses use all 5,000,000 records. Optional balanced sampling
arguments are provided only for script testing or separately reported
computational diagnostics; do not use sampled results as full-dataset final
results unless explicitly stated in the manuscript.
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import shutil
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "XGBoost is required. Install it with: python -m pip install xgboost"
    ) from exc


FEATURES = ["qber", "photon_count", "latency_ms", "abort_flag"]
CLASS_LABELS = ["normal", "partial_intercept_resend", "detector_blind", "fiber_tap"]
CLASS_TO_ID = {label: idx for idx, label in enumerate(CLASS_LABELS)}
ML_MODELS = ["Logistic_Regression", "Random_Forest", "XGBoost"]
ALL_METHODS = ["Transparent_Threshold_Baseline", *ML_MODELS]
DEFAULT_SEEDS = [1, 7, 21, 42, 99]


@dataclass(frozen=True)
class ExtendedValidationConfiguration:
    observable_features: tuple[str, ...] = tuple(FEATURES)
    class_labels: tuple[str, ...] = tuple(CLASS_LABELS)
    fixed_holdout_seed: int = 42
    fixed_holdout_test_fraction: float = 0.25
    validation_seeds: tuple[int, ...] = tuple(DEFAULT_SEEDS)
    validation_folds: int = 5
    logistic_max_iter: int = 1000
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int = 14
    xgboost_n_estimators: int = 400
    xgboost_max_depth: int = 10
    xgboost_learning_rate: float = 0.08
    xgboost_subsample: float = 0.90
    xgboost_colsample_bytree: float = 0.90


class TransparentThresholdBaseline:
    """Threshold baseline whose thresholds are fitted from training data only."""

    def __init__(self) -> None:
        self.thresholds_: dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransparentThresholdBaseline":
        q_idx = FEATURES.index("qber")
        p_idx = FEATURES.index("photon_count")

        def class_median(class_label: str, column: int) -> float:
            values = X[y == CLASS_TO_ID[class_label], column]
            if values.size == 0:
                raise ValueError(f"Training data contain no samples for {class_label}.")
            return float(np.median(values))

        normal_q = class_median("normal", q_idx)
        ir_q = class_median("partial_intercept_resend", q_idx)
        normal_p = class_median("normal", p_idx)
        tap_p = class_median("fiber_tap", p_idx)
        blind_p = class_median("detector_blind", p_idx)
        self.thresholds_ = {
            "detector_blind_photon_threshold": (blind_p + tap_p) / 2.0,
            "partial_intercept_resend_qber_threshold": (normal_q + ir_q) / 2.0,
            "fiber_tap_photon_threshold": (tap_p + normal_p) / 2.0,
        }
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Threshold baseline has not been fitted.")
        qber = X[:, FEATURES.index("qber")]
        photons = X[:, FEATURES.index("photon_count")]
        pred = np.full(X.shape[0], CLASS_TO_ID["normal"], dtype=np.int8)

        is_blind = photons <= self.thresholds_["detector_blind_photon_threshold"]
        pred[is_blind] = CLASS_TO_ID["detector_blind"]
        is_ir = (~is_blind) & (
            qber >= self.thresholds_["partial_intercept_resend_qber_threshold"]
        )
        pred[is_ir] = CLASS_TO_ID["partial_intercept_resend"]
        is_tap = (~is_blind) & (~is_ir) & (
            photons <= self.thresholds_["fiber_tap_photon_threshold"]
        )
        pred[is_tap] = CLASS_TO_ID["fiber_tap"]
        return pred


def parse_args() -> argparse.Namespace:
    default_root = r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"
    parser = argparse.ArgumentParser(
        description="Run all-model ablation, single-feature testing, and multi-seed validation."
    )
    parser.add_argument("--root", type=Path, default=Path(default_root))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Default: <root>/data/qf_log_dataset.csv",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=("ablation", "single-feature", "stability"),
        default=["ablation", "single-feature", "stability"],
        help="Analyses to execute. Existing completed rows are skipped unless --overwrite is used.",
    )
    parser.add_argument("--holdout-seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--xgb-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--diagnostic-samples-per-class",
        type=int,
        default=None,
        help="Optional balanced diagnostic subset for ablation/single-feature stages. Omit for final full-dataset results.",
    )
    parser.add_argument(
        "--cv-samples-per-class",
        type=int,
        default=None,
        help="Optional balanced subset for stability stage. Omit for final full-dataset results.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete prior extended-validation outputs and restart all requested stages.",
    )
    return parser.parse_args()


def setup_paths(root: Path, overwrite: bool) -> dict[str, Path]:
    out = root / "results" / "extended_validation"
    if overwrite and out.exists():
        shutil.rmtree(out)
    paths = {
        "root": out,
        "diagnostics": out / "diagnostics",
        "validation": out / "validation",
        "logs": out / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def setup_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger("qf_log_extended_validation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(path, mode="a", encoding="utf-8")
    fh.setFormatter(formatter)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def load_dataset(dataset_path: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    logger.info("Loading dataset: %s", dataset_path)
    dtypes = {
        "qber": "float32",
        "photon_count": "int32",
        "latency_ms": "float32",
        "abort_flag": "int8",
        "label": "category",
    }
    data = pd.read_csv(dataset_path, usecols=FEATURES + ["label"], dtype=dtypes)
    if data.empty or data.isna().any().any():
        raise ValueError("Dataset is empty or contains missing values.")
    labels_found = set(data["label"].astype(str).unique())
    if labels_found != set(CLASS_LABELS):
        raise ValueError(f"Unexpected labels: {sorted(labels_found)}")
    X = data[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = data["label"].astype(str).map(CLASS_TO_ID).to_numpy(dtype=np.int8, copy=True)
    counts = pd.Series(y).value_counts().sort_index().to_dict()
    logger.info("Dataset loaded: %s rows. Class counts: %s", f"{len(y):,}", counts)
    return X, y


def balanced_subset(
    X: np.ndarray,
    y: np.ndarray,
    samples_per_class: int | None,
    seed: int,
    logger: logging.Logger,
    purpose: str,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class is None:
        logger.info("%s will use the complete dataset: %s rows.", purpose, f"{len(y):,}")
        return X, y
    if samples_per_class <= 0:
        raise ValueError("Sample-per-class values must be positive integers.")
    rng = np.random.default_rng(seed)
    selected_parts: list[np.ndarray] = []
    for class_id in range(len(CLASS_LABELS)):
        candidates = np.flatnonzero(y == class_id)
        if samples_per_class > len(candidates):
            raise ValueError(
                f"Requested {samples_per_class:,} records for class {CLASS_LABELS[class_id]}, "
                f"but only {len(candidates):,} are available."
            )
        selected_parts.append(rng.choice(candidates, size=samples_per_class, replace=False))
    indices = np.concatenate(selected_parts)
    rng.shuffle(indices)
    logger.info(
        "%s will use a fixed balanced subset: %s rows (%s per class).",
        purpose,
        f"{len(indices):,}",
        f"{samples_per_class:,}",
    )
    return X[indices], y[indices]


def make_model(model_name: str, seed: int, n_jobs: int, xgb_device: str) -> BaseEstimator:
    if model_name == "Logistic_Regression":
        return Pipeline(
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed),
                ),
            ]
        )
    if model_name == "Random_Forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            criterion="gini",
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
    if model_name == "XGBoost":
        params: dict[str, Any] = {
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
            params["device"] = "cuda"
        return XGBClassifier(**params)
    raise ValueError(f"Unknown ML model: {model_name}")


def metric_values(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if probabilities is None:
        metrics["macro_roc_auc_ovr"] = math.nan
        metrics["macro_pr_auc_ovr"] = math.nan
    else:
        y_binary = label_binarize(y_true, classes=np.arange(len(CLASS_LABELS)))
        metrics["macro_roc_auc_ovr"] = float(
            roc_auc_score(y_binary, probabilities, average="macro", multi_class="ovr")
        )
        metrics["macro_pr_auc_ovr"] = float(
            average_precision_score(y_binary, probabilities, average="macro")
        )
    return metrics


def completed_keys(path: Path, key_columns: list[str]) -> set[tuple[str, ...]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    existing = pd.read_csv(path, dtype=str)
    if not set(key_columns).issubset(existing.columns):
        return set()
    return {tuple(row) for row in existing[key_columns].itertuples(index=False, name=None)}


def append_result(path: Path, row: dict[str, Any]) -> None:
    pd.DataFrame([row]).to_csv(path, mode="a", header=not path.exists(), index=False)


def fit_and_score(
    model_name: str,
    selected_indices: list[int],
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_jobs: int,
    xgb_device: str,
) -> dict[str, Any]:
    estimator = make_model(model_name, seed, n_jobs, xgb_device)
    started = time.perf_counter()
    estimator.fit(X_train[:, selected_indices], y_train)
    pred = estimator.predict(X_test[:, selected_indices]).astype(np.int8)
    prob = estimator.predict_proba(X_test[:, selected_indices])
    elapsed = time.perf_counter() - started
    output = metric_values(y_test, pred, prob)
    output["fit_predict_seconds"] = elapsed
    del estimator, pred, prob
    gc.collect()
    return output


def maybe_import_full_feature_holdout(
    root: Path,
    output_path: Path,
    logger: logging.Logger,
) -> None:
    """Import already completed four-feature ML rows if they exist and are not already saved."""
    source = root / "results" / "model_evaluation" / "metrics" / "holdout_overall_metrics.csv"
    if not source.exists():
        logger.info("No earlier holdout-metric file found; full-feature rows will be fitted when needed.")
        return
    existing_keys = completed_keys(output_path, ["model", "feature_condition"])
    holdout = pd.read_csv(source)
    for model_name in ML_MODELS:
        key = (model_name, "all_features")
        if key in existing_keys:
            continue
        rows = holdout[holdout["model"] == model_name]
        if rows.empty:
            continue
        original = rows.iloc[0].to_dict()
        row = {
            "model": model_name,
            "feature_condition": "all_features",
            "removed_feature": "none",
            "selected_features": ", ".join(FEATURES),
            "source": "imported_from_completed_holdout_evaluation",
            **{k: original.get(k, math.nan) for k in [
                "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
                "macro_f1", "weighted_f1", "mcc", "macro_roc_auc_ovr",
                "macro_pr_auc_ovr", "fit_predict_seconds",
            ]},
        }
        append_result(output_path, row)
        logger.info("Imported completed full-feature holdout row for %s.", model_name)


def run_ablation(
    root: Path,
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    paths: dict[str, Path],
    logger: logging.Logger,
) -> None:
    output = paths["diagnostics"] / "all_model_leave_one_feature_out_ablation.csv"
    maybe_import_full_feature_holdout(root, output, logger)
    done = completed_keys(output, ["model", "feature_condition"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.holdout_seed
    )
    for model_name in ML_MODELS:
        all_key = (model_name, "all_features")
        if all_key not in done:
            logger.info("Ablation table requires full-feature row; training %s with all features.", model_name)
            result = fit_and_score(
                model_name, list(range(len(FEATURES))), X_train, X_test, y_train, y_test,
                args.holdout_seed, args.n_jobs, args.xgb_device,
            )
            append_result(output, {
                "model": model_name,
                "feature_condition": "all_features",
                "removed_feature": "none",
                "selected_features": ", ".join(FEATURES),
                "source": "computed_by_extended_validation",
                **result,
            })
            done.add(all_key)
        for removed in FEATURES:
            condition = f"without_{removed}"
            key = (model_name, condition)
            if key in done:
                logger.info("Skipping completed ablation: %s | %s", model_name, condition)
                continue
            selected = [name for name in FEATURES if name != removed]
            indices = [FEATURES.index(name) for name in selected]
            logger.info("Running ablation: %s | without %s", model_name, removed)
            result = fit_and_score(
                model_name, indices, X_train, X_test, y_train, y_test,
                args.holdout_seed, args.n_jobs, args.xgb_device,
            )
            append_result(output, {
                "model": model_name,
                "feature_condition": condition,
                "removed_feature": removed,
                "selected_features": ", ".join(selected),
                "source": "computed_by_extended_validation",
                **result,
            })
            done.add(key)
            logger.info("Completed: %s | %s | macro_f1=%.6f", model_name, condition, result["macro_f1"])
    logger.info("All-model leave-one-feature-out ablation saved: %s", output)


def run_single_feature(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    paths: dict[str, Path],
    logger: logging.Logger,
) -> None:
    output = paths["diagnostics"] / "all_model_single_feature_evaluation.csv"
    done = completed_keys(output, ["model", "feature_used"])
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.holdout_seed
    )
    for model_name in ML_MODELS:
        for feature_name in FEATURES:
            key = (model_name, feature_name)
            if key in done:
                logger.info("Skipping completed single-feature evaluation: %s | %s", model_name, feature_name)
                continue
            logger.info("Running single-feature evaluation: %s | only %s", model_name, feature_name)
            result = fit_and_score(
                model_name, [FEATURES.index(feature_name)], X_train, X_test, y_train, y_test,
                args.holdout_seed, args.n_jobs, args.xgb_device,
            )
            append_result(output, {
                "model": model_name,
                "feature_used": feature_name,
                "selected_features": feature_name,
                **result,
            })
            done.add(key)
            logger.info("Completed: %s | only %s | macro_f1=%.6f", model_name, feature_name, result["macro_f1"])
    logger.info("All-model single-feature evaluation saved: %s", output)


def summarise_stability(run_path: Path, summary_path: Path) -> pd.DataFrame:
    df = pd.read_csv(run_path)
    metric_cols = [
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "mcc", "macro_roc_auc_ovr", "macro_pr_auc_ovr",
    ]
    records: list[dict[str, Any]] = []
    for model_name, group in df.groupby("model", sort=False):
        record: dict[str, Any] = {"model": model_name, "completed_runs": int(len(group))}
        for metric in metric_cols:
            record[f"{metric}_mean"] = float(group[metric].mean())
            record[f"{metric}_std"] = float(group[metric].std(ddof=1)) if len(group) > 1 else math.nan
            record[f"{metric}_min"] = float(group[metric].min())
            record[f"{metric}_max"] = float(group[metric].max())
        records.append(record)
    summary = pd.DataFrame(records)
    summary.to_csv(summary_path, index=False)
    return summary


def run_stability(
    X: np.ndarray,
    y: np.ndarray,
    args: argparse.Namespace,
    paths: dict[str, Path],
    logger: logging.Logger,
) -> None:
    run_path = paths["validation"] / "multiseed_5fold_run_metrics.csv"
    summary_path = paths["validation"] / "multiseed_5fold_summary.csv"
    done = completed_keys(run_path, ["model", "seed", "fold"])
    for seed in args.validation_seeds:
        splitter = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, valid_idx) in enumerate(splitter.split(X, y), start=1):
            X_train, X_valid = X[train_idx], X[valid_idx]
            y_train, y_valid = y[train_idx], y[valid_idx]

            baseline_key = ("Transparent_Threshold_Baseline", str(seed), str(fold))
            if baseline_key not in done:
                logger.info("Stability: threshold baseline | seed=%d fold=%d/%d", seed, fold, args.cv_folds)
                start = time.perf_counter()
                baseline = TransparentThresholdBaseline().fit(X_train, y_train)
                pred = baseline.predict(X_valid)
                elapsed = time.perf_counter() - start
                append_result(run_path, {
                    "model": "Transparent_Threshold_Baseline",
                    "seed": seed,
                    "fold": fold,
                    **metric_values(y_valid, pred),
                    "fit_predict_seconds": elapsed,
                })
                done.add(baseline_key)
                del baseline, pred

            for model_name in ML_MODELS:
                key = (model_name, str(seed), str(fold))
                if key in done:
                    logger.info("Skipping completed stability run: %s | seed=%d fold=%d", model_name, seed, fold)
                    continue
                logger.info("Stability: %s | seed=%d fold=%d/%d", model_name, seed, fold, args.cv_folds)
                result = fit_and_score(
                    model_name, list(range(len(FEATURES))),
                    X_train, X_valid, y_train, y_valid,
                    seed, args.n_jobs, args.xgb_device,
                )
                append_result(run_path, {
                    "model": model_name,
                    "seed": seed,
                    "fold": fold,
                    **result,
                })
                done.add(key)
                logger.info("Completed stability: %s | seed=%d fold=%d | macro_f1=%.6f", model_name, seed, fold, result["macro_f1"])
            del X_train, X_valid, y_train, y_valid
            gc.collect()
            summarise_stability(run_path, summary_path)
    summary = summarise_stability(run_path, summary_path)
    logger.info("Multi-seed validation summary:\n%s", summary.to_string(index=False))


def main() -> int:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")
    if len(set(args.validation_seeds)) != len(args.validation_seeds):
        raise ValueError("--validation-seeds must not contain duplicate values.")

    root = args.root.expanduser()
    dataset_path = args.dataset if args.dataset is not None else root / "data" / "qf_log_dataset.csv"
    paths = setup_paths(root, args.overwrite)
    logger = setup_logger(paths["logs"] / "extended_validation_log.txt")

    config = ExtendedValidationConfiguration(
        fixed_holdout_seed=args.holdout_seed,
        fixed_holdout_test_fraction=args.test_size,
        validation_seeds=tuple(args.validation_seeds),
        validation_folds=args.cv_folds,
    )
    configuration = {
        **asdict(config),
        "root": str(root),
        "dataset": str(dataset_path),
        "requested_stages": args.stages,
        "diagnostic_samples_per_class": args.diagnostic_samples_per_class,
        "cv_samples_per_class": args.cv_samples_per_class,
        "n_jobs": args.n_jobs,
        "xgb_device": args.xgb_device,
        "scope_note": (
            "Ablation and single-feature testing apply to Logistic Regression, Random Forest, "
            "and XGBoost. The threshold baseline is included in multi-seed validation only, "
            "because removal of its required rule variables would define a different baseline."
        ),
    }
    (paths["root"] / "extended_validation_configuration.json").write_text(
        json.dumps(configuration, indent=2), encoding="utf-8"
    )
    (paths["diagnostics"] / "ablation_scope_note.txt").write_text(
        configuration["scope_note"] + "\n", encoding="utf-8"
    )

    X, y = load_dataset(dataset_path, logger)

    if "ablation" in args.stages or "single-feature" in args.stages:
        X_diag, y_diag = balanced_subset(
            X, y, args.diagnostic_samples_per_class, args.holdout_seed, logger,
            "Ablation and single-feature evaluation",
        )
        if "ablation" in args.stages:
            run_ablation(root, X_diag, y_diag, args, paths, logger)
        if "single-feature" in args.stages:
            run_single_feature(X_diag, y_diag, args, paths, logger)
        del X_diag, y_diag
        gc.collect()

    if "stability" in args.stages:
        X_cv, y_cv = balanced_subset(
            X, y, args.cv_samples_per_class, args.holdout_seed, logger,
            "Multi-seed stratified cross-validation",
        )
        run_stability(X_cv, y_cv, args, paths, logger)
        del X_cv, y_cv
        gc.collect()

    logger.info("Requested extended validation stages completed. Outputs: %s", paths["root"])
    print("\nExtended validation execution completed for requested stages.")
    print(f"Outputs saved to: {paths['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
