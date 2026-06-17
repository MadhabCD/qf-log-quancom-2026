#!/usr/bin/env python3
"""
Model evaluation pipeline for the equation-driven QF-LOG dataset.

This script evaluates only the four observable features retained in the study:
    qber, photon_count, latency_ms, abort_flag

Class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Primary evaluation:
    - Transparent threshold baseline fitted on the training partition only
    - Logistic Regression
    - Random Forest
    - XGBoost
    - 75/25 stratified holdout evaluation
    - Confusion matrices, per-class metrics, probability-based metrics
    - McNemar comparison against the threshold baseline
    - Single-feature shortcut diagnostic using shallow decision trees

Optional final analyses:
    --run-ablation          Random Forest feature-ablation experiment
    --run-cross-validation  Stratified K-fold stability evaluation

Important methodological note:
    The generated dataset currently contains independent frame records rather
    than session identifiers. Therefore this script reproduces a stratified
    frame-level evaluation. It does not claim session-grouped validation.
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
from typing import Any, Dict, Iterable, Mapping

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.base import BaseEstimator, clone
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
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.tree import DecisionTreeClassifier, export_text

try:
    from xgboost import XGBClassifier
except ImportError as exc:  # pragma: no cover - user environment dependent
    raise ImportError(
        "XGBoost is required for this experiment. Install it with: "
        "python -m pip install xgboost"
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
MODEL_DISPLAY_NAMES = {
    "Transparent_Threshold_Baseline": "Transparent Threshold Baseline",
    "Logistic_Regression": "Logistic Regression",
    "Random_Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


@dataclass(frozen=True)
class EvaluationConfiguration:
    """Experiment values recorded with the generated evaluation outputs."""

    observable_features: tuple[str, ...] = tuple(FEATURES)
    class_labels: tuple[str, ...] = tuple(CLASS_LABELS)
    train_fraction: float = 0.75
    test_fraction: float = 0.25
    random_seed: int = 42
    logistic_max_iter: int = 1000
    random_forest_n_estimators: int = 300
    random_forest_max_depth: int = 14
    xgboost_n_estimators: int = 400
    xgboost_max_depth: int = 10
    xgboost_learning_rate: float = 0.08
    xgboost_subsample: float = 0.90
    xgboost_colsample_bytree: float = 0.90


class TransparentThresholdBaseline:
    """
    Transparent, low-complexity baseline using training-partition medians.

    Thresholds are derived from the training partition only, preventing use of
    test labels. The baseline reflects the primary expected signatures in the
    equation-driven generator:
        - detector_blind: strong photon-count suppression
        - partial_intercept_resend: raised QBER
        - fiber_tap: moderate photon-count reduction
        - otherwise: normal
    """

    def __init__(self) -> None:
        self.thresholds_: Dict[str, float] | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TransparentThresholdBaseline":
        if X.ndim != 2 or X.shape[1] != len(FEATURES):
            raise ValueError("Threshold baseline received an invalid feature matrix.")

        qber_index = FEATURES.index("qber")
        photon_index = FEATURES.index("photon_count")

        def median_for(label: str, feature_index: int) -> float:
            values = X[y == CLASS_TO_ID[label], feature_index]
            if values.size == 0:
                raise ValueError(f"Training partition has no rows for class: {label}")
            return float(np.median(values))

        normal_qber = median_for("normal", qber_index)
        ir_qber = median_for("partial_intercept_resend", qber_index)
        normal_photons = median_for("normal", photon_index)
        tap_photons = median_for("fiber_tap", photon_index)
        blind_photons = median_for("detector_blind", photon_index)

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

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.thresholds_ is None:
            raise RuntimeError("Threshold baseline must be fitted before prediction.")

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


def parse_arguments() -> argparse.Namespace:
    default_root = r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"
    parser = argparse.ArgumentParser(
        description="Evaluate ML models on the formula-based QF-LOG dataset."
    )
    parser.add_argument("--root", type=Path, default=Path(default_root))
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset path. Default: <root>/data/qf_log_dataset.csv",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument(
        "--xgb-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Use 'cuda' only when CUDA-enabled XGBoost is available.",
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Run Random Forest feature-ablation evaluation.",
    )
    parser.add_argument(
        "--run-cross-validation",
        action="store_true",
        help="Run stratified K-fold stability evaluation for all primary models.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--skip-prediction-files",
        action="store_true",
        help="Do not save test-set prediction files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing model_evaluation output folder.",
    )
    return parser.parse_args()


def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("qf_log_evaluation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def prepare_directories(root: Path, overwrite: bool) -> Dict[str, Path]:
    result_root = root / "results" / "model_evaluation"
    if result_root.exists() and any(result_root.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Evaluation output already exists: {result_root}\n"
                "Use --overwrite only when you intend to replace it."
            )
        shutil.rmtree(result_root)

    directories = {
        "root": result_root,
        "metrics": result_root / "metrics",
        "confusion": result_root / "confusion_matrices",
        "figures": result_root / "figures",
        "models": result_root / "models",
        "predictions": result_root / "predictions",
        "diagnostics": result_root / "diagnostics",
        "logs": result_root / "logs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories


def load_dataset(dataset_path: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
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

    if data.empty:
        raise ValueError("Dataset is empty.")
    if data.isnull().any().any():
        raise ValueError("Dataset contains missing values.")
    labels_found = set(data["label"].astype(str).unique())
    if labels_found != set(CLASS_LABELS):
        raise ValueError(
            "Dataset class labels do not match the required four classes. "
            f"Found: {sorted(labels_found)}"
        )
    if not data["qber"].between(0.0, 0.5).all():
        raise ValueError("QBER contains values outside [0, 0.5].")
    if not (data["photon_count"] >= 0).all():
        raise ValueError("Photon count contains negative values.")
    if not (data["latency_ms"] >= 0).all():
        raise ValueError("Latency contains negative values.")
    if not data["abort_flag"].isin([0, 1]).all():
        raise ValueError("Abort flag contains values other than 0 and 1.")

    X = data[FEATURES].to_numpy(dtype=np.float32, copy=True)
    label_text = data["label"].astype(str)
    y = label_text.map(CLASS_TO_ID).to_numpy(dtype=np.int8, copy=True)
    label_counts = (
        label_text.value_counts().reindex(CLASS_LABELS, fill_value=0).rename("samples").reset_index()
    )
    label_counts.columns = ["label", "samples"]
    logger.info("Dataset loaded: %s rows and %s observable features.", f"{len(data):,}", len(FEATURES))
    logger.info("Class distribution:\n%s", label_counts.to_string(index=False))
    return X, y, label_counts


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
            steps=[
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        solver="lbfgs",
                        max_iter=1000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            criterion="gini",
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(**xgb_kwargs),
    }


def probability_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> Dict[str, float]:
    binary_targets = label_binarize(y_true, classes=np.arange(len(CLASS_LABELS)))
    return {
        "macro_roc_auc_ovr": float(
            roc_auc_score(binary_targets, probabilities, average="macro", multi_class="ovr")
        ),
        "macro_pr_auc_ovr": float(
            average_precision_score(binary_targets, probabilities, average="macro")
        ),
    }


def aggregate_metrics(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> Dict[str, float]:
    output = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "macro_precision": float(
            precision_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_recall": float(
            recall_score(y_true, predictions, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, predictions, average="weighted", zero_division=0)
        ),
        "mcc": float(matthews_corrcoef(y_true, predictions)),
    }
    if probabilities is not None:
        output.update(probability_metrics(y_true, probabilities))
    else:
        output.update({"macro_roc_auc_ovr": math.nan, "macro_pr_auc_ovr": math.nan})
    return output


def save_confusion_outputs(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    paths: Mapping[str, Path],
) -> None:
    matrix = confusion_matrix(y_true, predictions, labels=np.arange(len(CLASS_LABELS)))
    matrix_df = pd.DataFrame(matrix, index=CLASS_LABELS, columns=CLASS_LABELS)
    matrix_df.to_csv(paths["confusion"] / f"{model_name.lower()}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(8.5, 7.0))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=CLASS_LABELS)
    display.plot(ax=ax, values_format=",d", colorbar=False)
    ax.set_title(MODEL_DISPLAY_NAMES.get(model_name, model_name))
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(paths["figures"] / f"{model_name.lower()}_confusion_matrix.png", dpi=300)
    plt.close(fig)


def save_classification_report(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    paths: Mapping[str, Path],
) -> None:
    report = classification_report(
        y_true,
        predictions,
        labels=np.arange(len(CLASS_LABELS)),
        target_names=CLASS_LABELS,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(paths["metrics"] / f"{model_name.lower()}_classification_report.csv")


def save_predictions(
    model_name: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
    paths: Mapping[str, Path],
) -> None:
    output = pd.DataFrame(
        {
            "true_label": [ID_TO_CLASS[int(value)] for value in y_true],
            "predicted_label": [ID_TO_CLASS[int(value)] for value in predictions],
            "correct": y_true == predictions,
        }
    )
    if probabilities is not None:
        for class_id, label in enumerate(CLASS_LABELS):
            output[f"probability_{label}"] = probabilities[:, class_id]
    output.to_csv(
        paths["predictions"] / f"{model_name.lower()}_test_predictions.csv.gz",
        index=False,
        compression="gzip",
    )


def fit_primary_models(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    models: Mapping[str, BaseEstimator],
    paths: Mapping[str, Path],
    logger: logging.Logger,
    skip_prediction_files: bool,
) -> tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    results: list[Dict[str, Any]] = []
    prediction_map: Dict[str, np.ndarray] = {}

    baseline = TransparentThresholdBaseline().fit(X_train, y_train)
    if baseline.thresholds_ is None:
        raise RuntimeError("Baseline threshold construction failed.")
    (paths["metrics"] / "transparent_threshold_baseline_rules.json").write_text(
        json.dumps(baseline.thresholds_, indent=2), encoding="utf-8"
    )
    started = time.perf_counter()
    baseline_predictions = baseline.predict(X_test)
    seconds = time.perf_counter() - started
    metrics = aggregate_metrics(y_test, baseline_predictions)
    results.append({"model": "Transparent_Threshold_Baseline", **metrics, "fit_predict_seconds": seconds})
    prediction_map["Transparent_Threshold_Baseline"] = baseline_predictions
    save_confusion_outputs("Transparent_Threshold_Baseline", y_test, baseline_predictions, paths)
    save_classification_report("Transparent_Threshold_Baseline", y_test, baseline_predictions, paths)
    if not skip_prediction_files:
        save_predictions("Transparent_Threshold_Baseline", y_test, baseline_predictions, None, paths)
    logger.info("Transparent threshold baseline completed. Accuracy=%.6f Macro-F1=%.6f", metrics["accuracy"], metrics["macro_f1"])

    for model_name, estimator in models.items():
        logger.info("Training %s.", MODEL_DISPLAY_NAMES.get(model_name, model_name))
        started = time.perf_counter()
        estimator.fit(X_train, y_train)
        predictions = estimator.predict(X_test).astype(np.int8)
        probabilities = estimator.predict_proba(X_test)
        seconds = time.perf_counter() - started
        metrics = aggregate_metrics(y_test, predictions, probabilities)
        results.append({"model": model_name, **metrics, "fit_predict_seconds": seconds})
        prediction_map[model_name] = predictions

        save_confusion_outputs(model_name, y_test, predictions, paths)
        save_classification_report(model_name, y_test, predictions, paths)
        if not skip_prediction_files:
            save_predictions(model_name, y_test, predictions, probabilities, paths)

        if model_name == "XGBoost":
            estimator.save_model(paths["models"] / "xgboost_model.json")
        else:
            joblib.dump(estimator, paths["models"] / f"{model_name.lower()}_model.joblib", compress=3)
        logger.info(
            "%s completed. Accuracy=%.6f Macro-F1=%.6f MCC=%.6f.",
            MODEL_DISPLAY_NAMES.get(model_name, model_name),
            metrics["accuracy"],
            metrics["macro_f1"],
            metrics["mcc"],
        )

    results_df = pd.DataFrame(results)
    results_df.to_csv(paths["metrics"] / "holdout_overall_metrics.csv", index=False)
    return results_df, prediction_map


def mcnemar_comparison(
    y_true: np.ndarray,
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> Dict[str, Any]:
    correct_a = predictions_a == y_true
    correct_b = predictions_b == y_true
    a_only_correct = int(np.sum(correct_a & ~correct_b))
    b_only_correct = int(np.sum(~correct_a & correct_b))
    discordant = a_only_correct + b_only_correct

    if discordant == 0:
        test_name = "No discordant predictions"
        statistic = 0.0
        p_value = 1.0
    elif discordant <= 25:
        test_name = "Exact binomial McNemar"
        statistic = math.nan
        p_value = float(binomtest(a_only_correct, discordant, 0.5, alternative="two-sided").pvalue)
    else:
        test_name = "McNemar chi-square with continuity correction"
        statistic = float((abs(a_only_correct - b_only_correct) - 1.0) ** 2 / discordant)
        p_value = float(chi2.sf(statistic, df=1))

    return {
        "model_a": name_a,
        "model_b": name_b,
        "model_a_only_correct": a_only_correct,
        "model_b_only_correct": b_only_correct,
        "discordant_predictions": discordant,
        "test": test_name,
        "statistic": statistic,
        "p_value": p_value,
    }


def run_mcnemar_tests(
    y_test: np.ndarray,
    prediction_map: Mapping[str, np.ndarray],
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    baseline_name = "Transparent_Threshold_Baseline"
    comparisons = []
    for model_name in ("Logistic_Regression", "Random_Forest", "XGBoost"):
        comparisons.append(
            mcnemar_comparison(
                y_test,
                prediction_map[model_name],
                prediction_map[baseline_name],
                model_name,
                baseline_name,
            )
        )
    comparisons.append(
        mcnemar_comparison(
            y_test,
            prediction_map["Random_Forest"],
            prediction_map["XGBoost"],
            "Random_Forest",
            "XGBoost",
        )
    )
    result = pd.DataFrame(comparisons)
    result.to_csv(paths["metrics"] / "mcnemar_pairwise_tests.csv", index=False)
    return result


def run_single_feature_shortcut_audit(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    """Use low-depth trees to test whether simple single-feature rules solve the task."""
    audit_rows: list[Dict[str, Any]] = []
    rule_texts: list[str] = []
    for feature_index, feature_name in enumerate(FEATURES):
        estimator = DecisionTreeClassifier(
            max_depth=4,
            min_samples_leaf=1000,
            random_state=seed,
        )
        estimator.fit(X_train[:, [feature_index]], y_train)
        predictions = estimator.predict(X_test[:, [feature_index]]).astype(np.int8)
        metrics = aggregate_metrics(y_test, predictions)
        audit_rows.append({"feature": feature_name, **metrics})
        rule_texts.append(
            f"FEATURE: {feature_name}\n"
            + export_text(estimator, feature_names=[feature_name])
            + "\n"
        )
        logger.info(
            "Single-feature audit: %s | accuracy=%.6f macro_f1=%.6f",
            feature_name,
            metrics["accuracy"],
            metrics["macro_f1"],
        )
    audit_df = pd.DataFrame(audit_rows)
    audit_df.to_csv(paths["diagnostics"] / "single_feature_shortcut_audit.csv", index=False)
    (paths["diagnostics"] / "single_feature_tree_rules.txt").write_text(
        "\n".join(rule_texts), encoding="utf-8"
    )
    return audit_df


def run_random_forest_ablation(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    n_jobs: int,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for removed_feature in FEATURES:
        selected = [feature for feature in FEATURES if feature != removed_feature]
        indices = [FEATURES.index(feature) for feature in selected]
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            criterion="gini",
            bootstrap=True,
            n_jobs=n_jobs,
            random_state=seed,
        )
        logger.info("Running Random Forest ablation without feature: %s", removed_feature)
        started = time.perf_counter()
        model.fit(X_train[:, indices], y_train)
        predictions = model.predict(X_test[:, indices]).astype(np.int8)
        seconds = time.perf_counter() - started
        metrics = aggregate_metrics(y_test, predictions)
        rows.append(
            {
                "removed_feature": removed_feature,
                "retained_features": ", ".join(selected),
                **metrics,
                "fit_predict_seconds": seconds,
            }
        )
    ablation_df = pd.DataFrame(rows)
    ablation_df.to_csv(paths["diagnostics"] / "random_forest_feature_ablation.csv", index=False)
    return ablation_df


def run_cross_validation(
    X: np.ndarray,
    y: np.ndarray,
    models: Mapping[str, BaseEstimator],
    folds: int,
    seed: int,
    paths: Mapping[str, Path],
    logger: logging.Logger,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_rows: list[Dict[str, Any]] = []

    for fold_number, (train_index, valid_index) in enumerate(splitter.split(X, y), start=1):
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]

        baseline = TransparentThresholdBaseline().fit(X_train, y_train)
        baseline_predictions = baseline.predict(X_valid)
        baseline_metrics = aggregate_metrics(y_valid, baseline_predictions)
        fold_rows.append({"model": "Transparent_Threshold_Baseline", "fold": fold_number, **baseline_metrics})
        logger.info("Cross-validation fold %d/%d baseline completed.", fold_number, folds)

        for model_name, model in models.items():
            logger.info("Cross-validation fold %d/%d training %s.", fold_number, folds, model_name)
            estimator = clone(model)
            estimator.fit(X_train, y_train)
            predictions = estimator.predict(X_valid).astype(np.int8)
            probabilities = estimator.predict_proba(X_valid)
            metrics = aggregate_metrics(y_valid, predictions, probabilities)
            fold_rows.append({"model": model_name, "fold": fold_number, **metrics})

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(paths["metrics"] / "cross_validation_fold_metrics.csv", index=False)

    metric_columns = [
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "mcc",
        "macro_roc_auc_ovr",
        "macro_pr_auc_ovr",
    ]
    aggregate_df = (
        fold_df.groupby("model", sort=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    aggregate_df.columns = [
        column if isinstance(column, str) else "_".join(part for part in column if part)
        for column in aggregate_df.columns
    ]
    aggregate_df.to_csv(paths["metrics"] / "cross_validation_summary.csv", index=False)
    return fold_df, aggregate_df


def save_split_summary(y_train: np.ndarray, y_test: np.ndarray, paths: Mapping[str, Path]) -> None:
    rows = []
    for partition, values in (("training", y_train), ("test", y_test)):
        counts = np.bincount(values, minlength=len(CLASS_LABELS))
        for class_id, count in enumerate(counts):
            rows.append({"partition": partition, "label": ID_TO_CLASS[class_id], "samples": int(count)})
    pd.DataFrame(rows).to_csv(paths["metrics"] / "train_test_split_summary.csv", index=False)


def main() -> int:
    args = parse_arguments()
    if not (0.0 < args.test_size < 1.0):
        raise ValueError("--test-size must be between 0 and 1.")
    if args.cv_folds < 2:
        raise ValueError("--cv-folds must be at least 2.")

    root = args.root.expanduser()
    dataset_path = args.dataset if args.dataset is not None else root / "data" / "qf_log_dataset.csv"
    paths = prepare_directories(root, args.overwrite)
    logger = setup_logger(paths["logs"] / "model_evaluation_log.txt")

    config = EvaluationConfiguration(
        train_fraction=1.0 - args.test_size,
        test_fraction=args.test_size,
        random_seed=args.seed,
    )
    (paths["metrics"] / "evaluation_configuration.json").write_text(
        json.dumps(
            {
                **asdict(config),
                "dataset_path": str(dataset_path),
                "run_ablation": bool(args.run_ablation),
                "run_cross_validation": bool(args.run_cross_validation),
                "cv_folds": args.cv_folds if args.run_cross_validation else None,
                "xgb_device": args.xgb_device,
                "n_jobs": args.n_jobs,
                "evaluation_note": (
                    "All classifiers use only the four observable QF-LOG features. "
                    "The generated dataset does not contain session identifiers; "
                    "the evaluation is stratified at frame level."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    X, y, class_distribution = load_dataset(dataset_path, logger)
    class_distribution.to_csv(paths["metrics"] / "dataset_class_distribution.csv", index=False)

    logger.info("Creating stratified 75/25 holdout partition with random seed %d.", args.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        stratify=y,
        random_state=args.seed,
    )
    save_split_summary(y_train, y_test, paths)
    logger.info("Training rows: %s | Test rows: %s", f"{len(y_train):,}", f"{len(y_test):,}")

    models = create_models(args.seed, args.n_jobs, args.xgb_device)
    holdout_results, prediction_map = fit_primary_models(
        X_train,
        X_test,
        y_train,
        y_test,
        models,
        paths,
        logger,
        args.skip_prediction_files,
    )
    mcnemar_results = run_mcnemar_tests(y_test, prediction_map, paths)
    shortcut_results = run_single_feature_shortcut_audit(
        X_train, X_test, y_train, y_test, args.seed, paths, logger
    )

    if args.run_ablation:
        run_random_forest_ablation(
            X_train, X_test, y_train, y_test, args.seed, args.n_jobs, paths, logger
        )

    if args.run_cross_validation:
        run_cross_validation(X, y, models, args.cv_folds, args.seed, paths, logger)

    logger.info("Holdout evaluation summary:\n%s", holdout_results.to_string(index=False))
    logger.info("McNemar comparison summary:\n%s", mcnemar_results.to_string(index=False))
    logger.info("Single-feature shortcut diagnostic:\n%s", shortcut_results.to_string(index=False))
    logger.info("Model evaluation completed. Results folder: %s", paths["root"])

    print("\nModel evaluation completed. Holdout summary:")
    print(holdout_results.to_string(index=False))
    print("\nSingle-feature shortcut diagnostic:")
    print(shortcut_results.to_string(index=False))
    print(f"\nAll outputs saved to: {paths['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
