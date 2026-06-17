#!/usr/bin/env python3
"""
Feature contribution analysis for the fresh QF-LOG Generator-A experiment.

Purpose
-------
This script uses the exact trained models from the completed fixed-holdout
Generator-A experiment and evaluates feature contribution on a stratified
subset of the unseen Generator-A holdout partition.

Primary analysis:
    Held-out permutation importance measured as the decrease in macro F1.
    This is computed for Logistic Regression, Random Forest, and XGBoost on
    the same diagnostic subset, making the result comparable across models.

Supplementary native analyses:
    - Logistic Regression: mean absolute standardized coefficient magnitude.
    - Random Forest: impurity-based feature importance returned by the model.
    - XGBoost: normalized gain importance returned by the trained booster.

The transparent threshold baseline is not assigned a learned feature
importance because its decision rules are explicit rather than fitted feature
importance scores.

Expected existing files
-----------------------
<data root>/data/qf_log_dataset.csv
<data root>/results/model_evaluation/models/logistic_regression_model.joblib
<data root>/results/model_evaluation/models/random_forest_model.joblib
<data root>/results/model_evaluation/models/xgboost_model.json

Default data root on Windows:
C:\\Users\\madha\\OneDrive\\Desktop\\QKD QF-Log Forensic
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise SystemExit(
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
MODEL_NAMES = ["Logistic_Regression", "Random_Forest", "XGBoost"]
MODEL_DISPLAY = {
    "Logistic_Regression": "Logistic Regression",
    "Random_Forest": "Random Forest",
    "XGBoost": "XGBoost",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate held-out permutation and native feature-importance results "
            "for trained QF-LOG Generator-A models."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"),
        help="Project root directory.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Dataset CSV path. Default: <root>/data/qf_log_dataset.csv.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used in the completed Generator-A fixed holdout evaluation.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Holdout proportion used in the completed Generator-A evaluation.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=25000,
        help=(
            "Number of unseen holdout records per class used for permutation "
            "importance. Default gives a 100,000-record diagnostic sample."
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=10,
        help="Permutation repetitions per feature and model.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for permutation importance. Default 1 avoids nested model parallelism on large data.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing feature-importance results folder.",
    )
    return parser.parse_args()


def prepare_directories(root: Path, overwrite: bool) -> Dict[str, Path]:
    result_root = root / "results" / "feature_importance"
    if result_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Results folder already exists: {result_root}\n"
                "Use --overwrite only when you intend to replace its contents."
            )
        shutil.rmtree(result_root)

    dirs = {
        "root": result_root,
        "tables": result_root / "tables",
        "figures": result_root / "figures",
        "logs": result_root / "logs",
        "metadata": result_root / "metadata",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def configure_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("qf_log_feature_importance")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def load_dataset(dataset_path: Path, logger: logging.Logger) -> tuple[np.ndarray, np.ndarray]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    logger.info("Loading Generator-A dataset: %s", dataset_path)
    dtypes = {
        "qber": "float32",
        "photon_count": "float32",
        "latency_ms": "float32",
        "abort_flag": "int8",
        "label": "category",
    }
    df = pd.read_csv(dataset_path, usecols=FEATURES + ["label"], dtype=dtypes)
    observed = set(df["label"].astype(str).unique())
    required = set(CLASS_LABELS)
    if observed != required:
        raise ValueError(
            f"Class labels do not match the expected labels. Found={observed}; expected={required}"
        )
    if not df["abort_flag"].isin([0, 1]).all():
        raise ValueError("abort_flag contains values other than 0 and 1.")
    X = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = df["label"].astype(str).map(CLASS_TO_ID).to_numpy(dtype=np.int8, copy=True)
    counts = df["label"].astype(str).value_counts().reindex(CLASS_LABELS)
    logger.info("Loaded %s records. Class counts: %s", f"{len(df):,}", counts.to_dict())
    return X, y


def reproduce_holdout_indices(y: np.ndarray, test_size: float, seed: int) -> np.ndarray:
    indices = np.arange(len(y), dtype=np.int64)
    _, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return np.asarray(test_idx, dtype=np.int64)


def stratified_subsample(
    holdout_idx: np.ndarray,
    y: np.ndarray,
    samples_per_class: int,
    seed: int,
) -> np.ndarray:
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive.")
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    holdout_y = y[holdout_idx]
    for class_id, label in enumerate(CLASS_LABELS):
        candidate_idx = holdout_idx[holdout_y == class_id]
        if len(candidate_idx) < samples_per_class:
            raise ValueError(
                f"Holdout contains only {len(candidate_idx):,} records for {label}; "
                f"cannot sample {samples_per_class:,}."
            )
        selected.append(rng.choice(candidate_idx, size=samples_per_class, replace=False))
    combined = np.concatenate(selected)
    rng.shuffle(combined)
    return combined


def load_models(root: Path, logger: logging.Logger) -> Dict[str, Any]:
    model_dir = root / "results" / "model_evaluation" / "models"
    expected = {
        "Logistic_Regression": model_dir / "logistic_regression_model.joblib",
        "Random_Forest": model_dir / "random_forest_model.joblib",
        "XGBoost": model_dir / "xgboost_model.json",
    }
    missing = [str(path) for path in expected.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Required saved Generator-A models were not found:\n" + "\n".join(missing)
        )
    models: Dict[str, Any] = {
        "Logistic_Regression": joblib.load(expected["Logistic_Regression"]),
        "Random_Forest": joblib.load(expected["Random_Forest"]),
    }
    xgb = XGBClassifier()
    xgb.load_model(expected["XGBoost"])
    models["XGBoost"] = xgb
    logger.info("Loaded the three saved Generator-A trained models.")
    return models


def evaluate_sample_reference(
    models: Dict[str, Any], X_sample: np.ndarray, y_sample: np.ndarray
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for name in MODEL_NAMES:
        pred = models[name].predict(X_sample).astype(np.int8)
        rows.append(
            {
                "model": name,
                "diagnostic_samples": int(len(y_sample)),
                "accuracy": float(accuracy_score(y_sample, pred)),
                "macro_f1": float(f1_score(y_sample, pred, average="macro")),
                "mcc": float(matthews_corrcoef(y_sample, pred)),
            }
        )
    return pd.DataFrame(rows)


def run_permutation_importance(
    models: Dict[str, Any],
    X_sample: np.ndarray,
    y_sample: np.ndarray,
    repeats: int,
    seed: int,
    n_jobs: int,
    logger: logging.Logger,
) -> pd.DataFrame:
    rows: list[Dict[str, Any]] = []
    for name in MODEL_NAMES:
        display = MODEL_DISPLAY[name]
        logger.info(
            "Computing held-out permutation importance: %s | %s records | %d repeats.",
            display,
            f"{len(y_sample):,}",
            repeats,
        )
        started = time.perf_counter()
        result = permutation_importance(
            models[name],
            X_sample,
            y_sample,
            scoring="f1_macro",
            n_repeats=repeats,
            random_state=seed,
            n_jobs=n_jobs,
        )
        elapsed = time.perf_counter() - started
        for feature, mean_value, std_value in zip(
            FEATURES, result.importances_mean, result.importances_std
        ):
            rows.append(
                {
                    "model": name,
                    "feature": feature,
                    "importance_method": "heldout_permutation_macro_f1_decrease",
                    "importance_mean": float(mean_value),
                    "importance_std": float(std_value),
                    "diagnostic_samples": int(len(y_sample)),
                    "n_repeats": int(repeats),
                    "runtime_seconds": float(elapsed),
                }
            )
        logger.info("Permutation importance completed: %s | %.2f seconds.", display, elapsed)
    return pd.DataFrame(rows)


def logistic_native_importance(model: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not hasattr(model, "named_steps") or "classifier" not in model.named_steps:
        raise ValueError("Expected Logistic Regression to be a fitted sklearn Pipeline.")
    classifier = model.named_steps["classifier"]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.shape != (len(CLASS_LABELS), len(FEATURES)):
        raise ValueError(f"Unexpected Logistic Regression coefficient shape: {coefficients.shape}")
    per_class_rows: list[Dict[str, Any]] = []
    for class_id, label in enumerate(CLASS_LABELS):
        for feature_id, feature in enumerate(FEATURES):
            per_class_rows.append(
                {
                    "model": "Logistic_Regression",
                    "class_label": label,
                    "feature": feature,
                    "standardized_coefficient": float(coefficients[class_id, feature_id]),
                    "absolute_standardized_coefficient": float(abs(coefficients[class_id, feature_id])),
                }
            )
    per_class = pd.DataFrame(per_class_rows)
    aggregate = (
        per_class.groupby("feature", as_index=False)["absolute_standardized_coefficient"]
        .mean()
        .rename(columns={"absolute_standardized_coefficient": "native_importance"})
    )
    aggregate["model"] = "Logistic_Regression"
    aggregate["importance_method"] = "mean_absolute_standardized_coefficient"
    total = aggregate["native_importance"].sum()
    aggregate["native_importance_normalized"] = aggregate["native_importance"] / total
    return aggregate[["model", "feature", "importance_method", "native_importance", "native_importance_normalized"]], per_class


def random_forest_native_importance(model: Any) -> pd.DataFrame:
    values = np.asarray(model.feature_importances_, dtype=float)
    if len(values) != len(FEATURES):
        raise ValueError("Unexpected Random Forest feature_importances_ length.")
    return pd.DataFrame(
        {
            "model": "Random_Forest",
            "feature": FEATURES,
            "importance_method": "impurity_based_importance",
            "native_importance": values,
            "native_importance_normalized": values / values.sum(),
        }
    )


def xgboost_native_importance(model: Any) -> pd.DataFrame:
    gain_map = model.get_booster().get_score(importance_type="gain")
    values = np.asarray([float(gain_map.get(f"f{i}", 0.0)) for i in range(len(FEATURES))])
    total = values.sum()
    normalized = values / total if total > 0 else np.zeros_like(values)
    return pd.DataFrame(
        {
            "model": "XGBoost",
            "feature": FEATURES,
            "importance_method": "gain_importance",
            "native_importance": values,
            "native_importance_normalized": normalized,
        }
    )


def create_permutation_plot(df: pd.DataFrame, output: Path) -> None:
    features_display = {
        "qber": "QBER",
        "photon_count": "Photon count",
        "latency_ms": "Latency",
        "abort_flag": "Abort flag",
    }
    models_display = [MODEL_DISPLAY[m] for m in MODEL_NAMES]
    temp = df.copy()
    temp["Feature"] = temp["feature"].map(features_display)
    temp["Model"] = temp["model"].map(MODEL_DISPLAY)
    pivot = temp.pivot(index="Feature", columns="Model", values="importance_mean")
    pivot = pivot.reindex(["QBER", "Photon count", "Latency", "Abort flag"])
    pivot = pivot.reindex(columns=models_display)
    ax = pivot.plot(kind="bar", figsize=(9, 5.2), rot=0)
    ax.set_ylabel("Decrease in macro F1 after permutation")
    ax.set_xlabel("Observable QF-LOG feature")
    ax.set_title("Held-out Permutation Importance on Generator-A")
    ax.legend(title="Model", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def create_native_plot(df: pd.DataFrame, output: Path) -> None:
    features_display = {
        "qber": "QBER",
        "photon_count": "Photon count",
        "latency_ms": "Latency",
        "abort_flag": "Abort flag",
    }
    models_display = [MODEL_DISPLAY[m] for m in MODEL_NAMES]
    temp = df.copy()
    temp["Feature"] = temp["feature"].map(features_display)
    temp["Model"] = temp["model"].map(MODEL_DISPLAY)
    pivot = temp.pivot(index="Feature", columns="Model", values="native_importance_normalized")
    pivot = pivot.reindex(["QBER", "Photon count", "Latency", "Abort flag"])
    pivot = pivot.reindex(columns=models_display)
    ax = pivot.plot(kind="bar", figsize=(9, 5.2), rot=0)
    ax.set_ylabel("Normalized within-model importance")
    ax.set_xlabel("Observable QF-LOG feature")
    ax.set_title("Supplementary Native Model Importance")
    ax.legend(title="Model", frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(ax.figure)


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    dataset_path = args.dataset.resolve() if args.dataset else root / "data" / "qf_log_dataset.csv"
    dirs = prepare_directories(root, args.overwrite)
    logger = configure_logger(dirs["logs"] / "feature_importance_log.txt")

    config = {
        "dataset": str(dataset_path),
        "features": FEATURES,
        "class_labels": CLASS_LABELS,
        "holdout_seed": args.seed,
        "test_size": args.test_size,
        "diagnostic_samples_per_class": args.samples_per_class,
        "permutation_repeats": args.n_repeats,
        "primary_importance_method": "heldout_permutation_macro_f1_decrease",
        "native_importance_is_supplementary": True,
    }
    (dirs["metadata"] / "feature_importance_configuration.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    X, y = load_dataset(dataset_path, logger)
    holdout_idx = reproduce_holdout_indices(y, args.test_size, args.seed)
    selected_idx = stratified_subsample(holdout_idx, y, args.samples_per_class, args.seed)
    X_sample = X[selected_idx]
    y_sample = y[selected_idx]
    sample_counts = pd.Series(y_sample).map({v: k for k, v in CLASS_TO_ID.items()}).value_counts().reindex(CLASS_LABELS)
    sample_counts.rename("samples").to_csv(dirs["metadata"] / "diagnostic_sample_class_counts.csv")
    logger.info(
        "Using a stratified unseen holdout diagnostic sample of %s records: %s",
        f"{len(y_sample):,}",
        sample_counts.to_dict(),
    )

    models = load_models(root, logger)
    reference = evaluate_sample_reference(models, X_sample, y_sample)
    reference.to_csv(dirs["tables"] / "diagnostic_sample_reference_performance.csv", index=False)

    permutation_df = run_permutation_importance(
        models,
        X_sample,
        y_sample,
        args.n_repeats,
        args.seed,
        args.n_jobs,
        logger,
    )
    permutation_df["rank_within_model"] = permutation_df.groupby("model")["importance_mean"].rank(
        ascending=False, method="dense"
    ).astype(int)
    permutation_df = permutation_df.sort_values(["model", "rank_within_model", "feature"])
    permutation_df.to_csv(dirs["tables"] / "heldout_permutation_importance_macro_f1.csv", index=False)

    lr_importance, lr_coefficients = logistic_native_importance(models["Logistic_Regression"])
    native_df = pd.concat(
        [
            lr_importance,
            random_forest_native_importance(models["Random_Forest"]),
            xgboost_native_importance(models["XGBoost"]),
        ],
        ignore_index=True,
    )
    native_df["rank_within_model"] = native_df.groupby("model")["native_importance_normalized"].rank(
        ascending=False, method="dense"
    ).astype(int)
    native_df = native_df.sort_values(["model", "rank_within_model", "feature"])
    native_df.to_csv(dirs["tables"] / "supplementary_native_model_importance.csv", index=False)
    lr_coefficients.to_csv(
        dirs["tables"] / "logistic_regression_standardized_coefficients_by_class.csv", index=False
    )

    create_permutation_plot(
        permutation_df,
        dirs["figures"] / "heldout_permutation_importance_macro_f1.png",
    )
    create_native_plot(
        native_df,
        dirs["figures"] / "supplementary_native_model_importance.png",
    )

    note = (
        "Primary interpretation should use heldout_permutation_importance_macro_f1.csv.\n"
        "This table reports the decrease in macro F1 when one feature is permuted on\n"
        "the same unseen holdout diagnostic sample for all three trained ML models.\n\n"
        "supplementary_native_model_importance.csv is supplementary only because\n"
        "coefficient magnitude, Random Forest impurity importance, and XGBoost gain\n"
        "are model-specific quantities and should not be compared as the same scale.\n\n"
        "The transparent threshold baseline is excluded from learned feature importance\n"
        "because its feature use is explicitly defined by its decision rules.\n"
    )
    (dirs["metadata"] / "interpretation_note.txt").write_text(note, encoding="utf-8")

    logger.info("Diagnostic sample reference performance:\n%s", reference.to_string(index=False))
    logger.info(
        "Held-out permutation importance:\n%s",
        permutation_df[["model", "feature", "importance_mean", "importance_std", "rank_within_model"]].to_string(index=False),
    )
    logger.info("Feature contribution analysis completed. Outputs: %s", dirs["root"])

    print("\nFeature contribution analysis completed. Primary output:")
    print(permutation_df[["model", "feature", "importance_mean", "importance_std", "rank_within_model"]].to_string(index=False))
    print(f"\nResults saved to: {dirs['root']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
