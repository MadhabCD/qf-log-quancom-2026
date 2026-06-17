#!/usr/bin/env python3
"""
QF-LOG formula-based dataset generator.

Purpose
-------
Generate a balanced, physics-informed synthetic QKD forensic log dataset with
the four observable features retained in the study:
    qber, photon_count, latency_ms, abort_flag

and the four class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Design principles
-----------------
1. Observable features are calculated from explicit equations, not drawn
   directly from class-specific feature ranges.
2. photon_count is the detected click/count statistic per QKD frame.
3. qber is obtained from generated error events among detected counts.
4. latency_ms and abort_flag are explicit controller-level operational models;
   they are not claimed as quantum-mechanical laws or hardware measurements.
5. fiber_tap represents added optical attenuation/signal diversion, not
   copying an unknown quantum state.
6. detector_blind is modelled as a monitored detector-response suppression
   condition; it is a simulation proxy for an observable forensic signature,
   not a universal description of every bright-illumination attack.

The generator is reproducible: its random seed, parameter specification,
equations, summary statistics, and latent generation metadata are saved.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, TextIO

import numpy as np
import pandas as pd


CLASSES = (
    "normal",
    "partial_intercept_resend",
    "detector_blind",
    "fiber_tap",
)


@dataclass(frozen=True)
class SimulationParameters:
    """Configured parameters used by the equation-driven simulator."""

    # Dataset design
    total_samples: int = 5_000_000
    random_seed: int = 42
    chunk_size: int = 100_000
    pulses_per_frame: int = 16_000

    # Source/channel/detector baseline parameters
    mean_photon_number_mean: float = 0.50
    mean_photon_number_sd: float = 0.04
    mean_photon_number_low: float = 0.35
    mean_photon_number_high: float = 0.65

    fiber_length_km_mean: float = 10.0
    fiber_length_km_sd: float = 2.5
    fiber_length_km_low: float = 3.0
    fiber_length_km_high: float = 20.0

    attenuation_db_per_km_mean: float = 0.20
    attenuation_db_per_km_sd: float = 0.01
    attenuation_db_per_km_low: float = 0.17
    attenuation_db_per_km_high: float = 0.23

    detector_efficiency_mean: float = 0.20
    detector_efficiency_sd: float = 0.018
    detector_efficiency_low: float = 0.14
    detector_efficiency_high: float = 0.26

    background_yield_mean: float = 5.0e-5
    background_yield_sd: float = 1.0e-5
    background_yield_low: float = 1.0e-5
    background_yield_high: float = 9.0e-5

    misalignment_error_mean: float = 0.018
    misalignment_error_sd: float = 0.006
    misalignment_error_low: float = 0.005
    misalignment_error_high: float = 0.040

    # Partial intercept-resend attack intensity:
    # f_IR = lower + scale * Beta(alpha, beta)
    intercept_fraction_lower: float = 0.03
    intercept_fraction_scale: float = 0.42
    intercept_fraction_beta_alpha: float = 1.5
    intercept_fraction_beta_beta: float = 2.8
    intercept_resend_error_on_attacked_fraction: float = 0.25

    # Fiber-tap attack parameter: added optical attenuation in dB
    tap_loss_db_lower: float = 0.10
    tap_loss_db_scale: float = 2.50
    tap_loss_beta_alpha: float = 1.5
    tap_loss_beta_beta: float = 2.2

    # Detector-blind monitoring proxy: response suppression and modest
    # response-instability error component.
    blind_response_lower: float = 0.05
    blind_response_scale: float = 0.90
    blind_response_beta_alpha: float = 1.4
    blind_response_beta_beta: float = 2.2
    blind_excess_error_scale: float = 0.025
    blind_error_beta_alpha: float = 1.2
    blind_error_beta_beta: float = 3.0

    # Controller-level processing latency model
    base_latency_ms_mean: float = 4.55
    base_latency_ms_sd: float = 0.20
    base_latency_ms_low: float = 4.00
    base_latency_ms_high: float = 5.10
    error_correction_latency_weight_ms: float = 5.50
    loss_handling_latency_weight_ms: float = 2.80
    latency_jitter_ms_sd: float = 0.15

    # Operational abort model
    spontaneous_abort_probability: float = 0.0005
    abort_qber_center: float = 0.105
    abort_qber_slope: float = 90.0
    abort_loss_center: float = 0.55
    abort_loss_slope: float = 18.0
    abort_latency_center_ms: float = 8.00
    abort_latency_slope: float = 3.00


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    sd: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Generate a bounded normal draw for configured simulator parameters."""
    values = rng.normal(mean, sd, size=size)
    return np.clip(values, low, high)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binary_entropy(q: np.ndarray) -> np.ndarray:
    """Binary entropy H2(q), with safe clipping for log calculations."""
    q_safe = np.clip(q, 1e-12, 1.0 - 1e-12)
    return -(q_safe * np.log2(q_safe) + (1.0 - q_safe) * np.log2(1.0 - q_safe))


def balanced_shuffled_labels(rng: np.random.Generator, size: int) -> np.ndarray:
    """Return an equal count of each class, shuffled within the chunk."""
    if size % len(CLASSES) != 0:
        raise ValueError("Every chunk size must be divisible by the number of classes (4).")
    labels = np.repeat(np.asarray(CLASSES, dtype=object), size // len(CLASSES))
    rng.shuffle(labels)
    return labels


def generate_chunk(
    rng: np.random.Generator,
    params: SimulationParameters,
    start_sample_id: int,
    size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one chunk of ML rows and full generation metadata."""

    labels = balanced_shuffled_labels(rng, size)

    # ---- Baseline hidden operational parameters --------------------------
    mu = clipped_normal(
        rng,
        params.mean_photon_number_mean,
        params.mean_photon_number_sd,
        params.mean_photon_number_low,
        params.mean_photon_number_high,
        size,
    )
    fiber_length = clipped_normal(
        rng,
        params.fiber_length_km_mean,
        params.fiber_length_km_sd,
        params.fiber_length_km_low,
        params.fiber_length_km_high,
        size,
    )
    attenuation = clipped_normal(
        rng,
        params.attenuation_db_per_km_mean,
        params.attenuation_db_per_km_sd,
        params.attenuation_db_per_km_low,
        params.attenuation_db_per_km_high,
        size,
    )
    detector_efficiency = clipped_normal(
        rng,
        params.detector_efficiency_mean,
        params.detector_efficiency_sd,
        params.detector_efficiency_low,
        params.detector_efficiency_high,
        size,
    )
    background_yield = clipped_normal(
        rng,
        params.background_yield_mean,
        params.background_yield_sd,
        params.background_yield_low,
        params.background_yield_high,
        size,
    )
    misalignment_error = clipped_normal(
        rng,
        params.misalignment_error_mean,
        params.misalignment_error_sd,
        params.misalignment_error_low,
        params.misalignment_error_high,
        size,
    )

    # ---- Attack intervention parameters ----------------------------------
    intercept_fraction = np.zeros(size, dtype=np.float64)
    tap_loss_db = np.zeros(size, dtype=np.float64)
    detector_response_factor = np.ones(size, dtype=np.float64)
    blind_excess_error = np.zeros(size, dtype=np.float64)

    ir_mask = labels == "partial_intercept_resend"
    tap_mask = labels == "fiber_tap"
    blind_mask = labels == "detector_blind"

    n_ir = int(ir_mask.sum())
    n_tap = int(tap_mask.sum())
    n_blind = int(blind_mask.sum())

    intercept_fraction[ir_mask] = (
        params.intercept_fraction_lower
        + params.intercept_fraction_scale
        * rng.beta(params.intercept_fraction_beta_alpha, params.intercept_fraction_beta_beta, n_ir)
    )

    tap_loss_db[tap_mask] = (
        params.tap_loss_db_lower
        + params.tap_loss_db_scale
        * rng.beta(params.tap_loss_beta_alpha, params.tap_loss_beta_beta, n_tap)
    )

    detector_response_factor[blind_mask] = (
        params.blind_response_lower
        + params.blind_response_scale
        * rng.beta(params.blind_response_beta_alpha, params.blind_response_beta_beta, n_blind)
    )
    blind_excess_error[blind_mask] = (
        params.blind_excess_error_scale
        * rng.beta(params.blind_error_beta_alpha, params.blind_error_beta_beta, n_blind)
    )

    # ---- Photon count equation -------------------------------------------
    # Baseline channel transmission:
    #   eta_ch,0 = 10^[-(alpha * L) / 10]
    # Attacked channel transmission for tap:
    #   eta_ch = 10^[-(alpha * L + a_tap) / 10]
    baseline_channel_transmission = np.power(10.0, -(attenuation * fiber_length) / 10.0)
    attacked_channel_transmission = np.power(
        10.0, -(attenuation * fiber_length + tap_loss_db) / 10.0
    )

    reference_total_efficiency = detector_efficiency * baseline_channel_transmission
    attacked_total_efficiency = (
        detector_efficiency * attacked_channel_transmission * detector_response_factor
    )

    # Signal detection/click probability for weak coherent pulses:
    #   p_sig = 1 - exp(-mu * eta)
    # Background-adjusted observed click probability:
    #   p_click = 1 - (1 - Y0) exp(-mu * eta)
    reference_signal_probability = 1.0 - np.exp(-mu * reference_total_efficiency)
    attacked_signal_probability = 1.0 - np.exp(-mu * attacked_total_efficiency)
    reference_click_probability = 1.0 - (1.0 - background_yield) * np.exp(
        -mu * reference_total_efficiency
    )
    observed_click_probability = 1.0 - (1.0 - background_yield) * np.exp(
        -mu * attacked_total_efficiency
    )

    expected_reference_count = params.pulses_per_frame * reference_click_probability
    photon_count = rng.binomial(params.pulses_per_frame, observed_click_probability).astype(np.int32)

    # ---- QBER equation ----------------------------------------------------
    # Baseline optical error probability:
    #   q_base = [0.5*Y0 + e_d*(1-exp(-mu*eta))] / p_click
    q_base = (
        0.5 * background_yield + misalignment_error * attacked_signal_probability
    ) / np.maximum(observed_click_probability, 1e-12)

    # For partial BB84 intercept-resend:
    #   q_true = (1-f_IR)q_base + f_IR*0.25
    q_true = q_base.copy()
    q_true[ir_mask] = (
        (1.0 - intercept_fraction[ir_mask]) * q_base[ir_mask]
        + intercept_fraction[ir_mask] * params.intercept_resend_error_on_attacked_fraction
    )

    # For fiber tap, no arbitrary QBER rise is inserted. Its changed q_base
    # follows the lower signal-to-background relation after attenuation.
    # For detector-blind, this simulator adds an explicitly configured
    # monitored response-instability component.
    q_true[blind_mask] = q_base[blind_mask] + blind_excess_error[blind_mask]
    q_true = np.clip(q_true, 0.0, 0.5)

    error_count = rng.binomial(photon_count, q_true).astype(np.int32)
    qber = np.divide(
        error_count,
        photon_count,
        out=np.zeros(size, dtype=np.float64),
        where=photon_count > 0,
    )

    # ---- Controller latency equation -------------------------------------
    # This is an explicit operational controller model, not optical
    # propagation delay:
    #   T = T0 + k_EC H2(QBER) + k_loss R_loss + epsilon_T
    # where R_loss is photon-count deficiency relative to the same-frame
    # no-attack expected count.
    observed_loss_fraction = np.clip(
        1.0 - photon_count / np.maximum(expected_reference_count, 1.0), 0.0, 1.0
    )
    base_latency = clipped_normal(
        rng,
        params.base_latency_ms_mean,
        params.base_latency_ms_sd,
        params.base_latency_ms_low,
        params.base_latency_ms_high,
        size,
    )
    latency_jitter = rng.normal(0.0, params.latency_jitter_ms_sd, size=size)
    latency_ms = (
        base_latency
        + params.error_correction_latency_weight_ms * binary_entropy(qber)
        + params.loss_handling_latency_weight_ms * observed_loss_fraction
        + latency_jitter
    )
    latency_ms = np.maximum(latency_ms, 0.0)

    # ---- Abort flag equation ---------------------------------------------
    # Abort causes are modelled as independent controller hazards:
    #   p_Q = sigmoid(s_Q * (QBER - tau_Q))
    #   p_L = sigmoid(s_L * (R_loss - tau_L))
    #   p_T = sigmoid(s_T * (T - tau_T))
    #   P(abort=1) = 1 - (1-p0)(1-p_Q)(1-p_L)(1-p_T)
    p_abort_qber = sigmoid(
        params.abort_qber_slope * (qber - params.abort_qber_center)
    )
    p_abort_loss = sigmoid(
        params.abort_loss_slope * (observed_loss_fraction - params.abort_loss_center)
    )
    p_abort_latency = sigmoid(
        params.abort_latency_slope * (latency_ms - params.abort_latency_center_ms)
    )
    abort_probability = 1.0 - (
        (1.0 - params.spontaneous_abort_probability)
        * (1.0 - p_abort_qber)
        * (1.0 - p_abort_loss)
        * (1.0 - p_abort_latency)
    )
    abort_probability = np.clip(abort_probability, 0.0, 1.0)
    abort_flag = rng.binomial(1, abort_probability).astype(np.int8)

    sample_id = np.arange(start_sample_id, start_sample_id + size, dtype=np.int64)

    ml_data = pd.DataFrame(
        {
            "qber": np.round(qber, 8),
            "photon_count": photon_count,
            "latency_ms": np.round(latency_ms, 6),
            "abort_flag": abort_flag,
            "label": labels,
        }
    )

    metadata = pd.DataFrame(
        {
            "sample_id": sample_id,
            "label": labels,
            "qber": np.round(qber, 8),
            "photon_count": photon_count,
            "latency_ms": np.round(latency_ms, 6),
            "abort_flag": abort_flag,
            "error_count": error_count,
            "pulses_per_frame": params.pulses_per_frame,
            "mean_photon_number": np.round(mu, 8),
            "fiber_length_km": np.round(fiber_length, 6),
            "attenuation_db_per_km": np.round(attenuation, 8),
            "detector_efficiency": np.round(detector_efficiency, 8),
            "background_yield": np.round(background_yield, 10),
            "misalignment_error": np.round(misalignment_error, 8),
            "intercept_fraction": np.round(intercept_fraction, 8),
            "tap_loss_db": np.round(tap_loss_db, 8),
            "detector_response_factor": np.round(detector_response_factor, 8),
            "blind_excess_error": np.round(blind_excess_error, 8),
            "reference_click_probability": np.round(reference_click_probability, 10),
            "observed_click_probability": np.round(observed_click_probability, 10),
            "expected_reference_count": np.round(expected_reference_count, 6),
            "observed_loss_fraction": np.round(observed_loss_fraction, 8),
            "qber_probability": np.round(q_true, 8),
            "abort_probability": np.round(abort_probability, 8),
        }
    )
    return ml_data, metadata


def validate_chunk(data: pd.DataFrame, params: SimulationParameters) -> None:
    """Fail immediately if the equation generator produces invalid outputs."""
    if data.isnull().any().any():
        raise ValueError("Generated data contain missing values.")
    if not data["qber"].between(0.0, 0.5).all():
        raise ValueError("QBER values fall outside [0, 0.5].")
    if not data["photon_count"].between(0, params.pulses_per_frame).all():
        raise ValueError("Photon-count values fall outside permitted frame count.")
    if not (data["latency_ms"] >= 0.0).all():
        raise ValueError("Latency values must be nonnegative.")
    if not data["abort_flag"].isin([0, 1]).all():
        raise ValueError("Abort flag is not binary.")
    if set(data["label"].unique()) != set(CLASSES):
        raise ValueError("A generated chunk does not contain all four required classes.")


def initialise_stats() -> Dict[str, Dict[str, float]]:
    return {
        label: {
            "n": 0.0,
            "qber_sum": 0.0,
            "qber_sq_sum": 0.0,
            "qber_min": math.inf,
            "qber_max": -math.inf,
            "photon_count_sum": 0.0,
            "photon_count_sq_sum": 0.0,
            "photon_count_min": math.inf,
            "photon_count_max": -math.inf,
            "latency_ms_sum": 0.0,
            "latency_ms_sq_sum": 0.0,
            "latency_ms_min": math.inf,
            "latency_ms_max": -math.inf,
            "abort_flag_sum": 0.0,
        }
        for label in CLASSES
    }


def update_stats(stats: Dict[str, Dict[str, float]], data: pd.DataFrame) -> None:
    for label, group in data.groupby("label", sort=False):
        entry = stats[str(label)]
        n = len(group)
        entry["n"] += n
        for col in ("qber", "photon_count", "latency_ms"):
            values = group[col].to_numpy(dtype=np.float64)
            entry[f"{col}_sum"] += float(values.sum())
            entry[f"{col}_sq_sum"] += float(np.square(values).sum())
            entry[f"{col}_min"] = min(entry[f"{col}_min"], float(values.min()))
            entry[f"{col}_max"] = max(entry[f"{col}_max"], float(values.max()))
        entry["abort_flag_sum"] += float(group["abort_flag"].sum())


def finalise_summary(stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rows = []
    for label in CLASSES:
        entry = stats[label]
        n = int(entry["n"])
        row: Dict[str, object] = {"label": label, "samples": n}
        for col in ("qber", "photon_count", "latency_ms"):
            mean = entry[f"{col}_sum"] / n
            variance = max(entry[f"{col}_sq_sum"] / n - mean * mean, 0.0)
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = math.sqrt(variance)
            row[f"{col}_min"] = entry[f"{col}_min"]
            row[f"{col}_max"] = entry[f"{col}_max"]
        row["abort_rate"] = entry["abort_flag_sum"] / n
        rows.append(row)
    return pd.DataFrame(rows)


def equation_text(params: SimulationParameters) -> str:
    return f"""QF-LOG Formula-Based Dataset Generation Model
================================================

Observable ML features:
    qber, photon_count, latency_ms, abort_flag

Class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Important interpretation:
    photon_count is the detected photon/click count per generated QKD frame.
    latency_ms is a controller-level processing/buffering delay model; it is
    not optical time-of-flight.
    abort_flag is generated from an operational probability model; it is not
    assigned directly from the class label.
    fiber_tap models added optical attenuation/signal diversion, not copying
    an unknown qubit.
    detector_blind models a monitored response-suppression condition and is
    an explicit simulator assumption rather than a universal hardware trace.

1. Channel and detected-count model
-----------------------------------
Let:
    N      = pulses per frame = {params.pulses_per_frame}
    mu     = mean photon number
    alpha  = attenuation in dB/km
    L      = fiber length in km
    a_tap  = added tap attenuation in dB
    eta_d  = detector efficiency
    r_DB   = detector response factor (1 outside detector_blind)
    Y0     = background yield probability

Baseline channel transmission:
    eta_ch,0 = 10^[-(alpha * L) / 10]

Attack-adjusted channel transmission:
    eta_ch = 10^[-(alpha * L + a_tap) / 10]

Attack-adjusted total detection efficiency:
    eta = eta_d * eta_ch * r_DB

Observed click probability:
    p_click = 1 - (1 - Y0) * exp(-mu * eta)

Detected photon/count feature:
    photon_count ~ Binomial(N, p_click)

2. QBER model
-------------
Let e_d be the optical misalignment/error parameter.

Signal click probability:
    p_sig = 1 - exp(-mu * eta)

Baseline error probability:
    q_base = [0.5 * Y0 + e_d * p_sig] / p_click

Normal:
    q_true = q_base

Partial intercept-resend:
    q_true = (1 - f_IR) * q_base + f_IR * 0.25

Fiber tap:
    q_true = q_base
    No arbitrary QBER increase is imposed; attenuation changes signal and
    background composition through the detection equation.

Detector blind monitoring proxy:
    q_true = q_base + delta_DB

Observed error events:
    error_count ~ Binomial(photon_count, q_true)

Observed QBER feature:
    qber = error_count / photon_count, when photon_count > 0

3. Controller latency model
---------------------------
Binary entropy:
    H2(q) = -q*log2(q) - (1-q)*log2(1-q)

Reference count is the expected detected count without attack effect in the
same baseline frame configuration:
    C_ref = N * [1 - (1-Y0)*exp(-mu*eta_d*eta_ch,0)]

Observed loss fraction:
    R_loss = max(0, 1 - photon_count / C_ref)

Controller-level latency:
    latency_ms = T0 + k_EC*H2(qber) + k_loss*R_loss + epsilon_T

where epsilon_T is normally distributed controller jitter.

4. Abort-flag model
-------------------
Abort probability uses independent controller hazard terms:

    p_Q = sigmoid(s_Q * (qber - tau_Q))
    p_L = sigmoid(s_L * (R_loss - tau_L))
    p_T = sigmoid(s_T * (latency_ms - tau_T))

    P(abort_flag = 1)
      = 1 - (1-p0)*(1-p_Q)*(1-p_L)*(1-p_T)

    abort_flag ~ Bernoulli(P(abort_flag = 1))

5. Source basis and modelling limits
------------------------------------
- The weak-coherent-pulse loss/detection and QBER observation structure is
  based on practical decoy-state QKD modelling:
  Ma, Qi, Zhao, and Lo, Practical Decoy State for Quantum Key Distribution,
  Physical Review A, 72, 012326, 2005.
- The 0.25 attacked-pulse error contribution is the standard BB84
  intercept-resend disturbance under random basis choice.
- The detector_blind class is motivated by practical detector-control attacks
  using bright illumination:
  Lydersen et al., Hacking commercial quantum cryptography systems by
  tailored bright illumination, Nature Photonics, 4, 686-689, 2010.
- The detector-response factor, controller latency equation, and abort
  probability rule are explicitly configured simulation assumptions. They
  must not be described as directly measured hardware laws unless validated
  with experimental QKD logs.
"""


def configure_logging(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("qf_log_generator")
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


def parse_arguments() -> argparse.Namespace:
    default_root = r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"
    parser = argparse.ArgumentParser(
        description="Generate the equation-based QF-LOG forensic dataset."
    )
    parser.add_argument("--root", type=Path, default=Path(default_root))
    parser.add_argument("--n-samples", type=int, default=5_000_000)
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacement of existing output files.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.n_samples <= 0 or args.n_samples % len(CLASSES) != 0:
        raise ValueError("--n-samples must be positive and divisible by 4.")
    if args.chunk_size <= 0 or args.chunk_size % len(CLASSES) != 0:
        raise ValueError("--chunk-size must be positive and divisible by 4.")

    root = args.root.expanduser()
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    logs_dir = root / "logs"
    for directory in (data_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ml_path = data_dir / "qf_log_dataset.csv"
    full_metadata_path = data_dir / "qf_log_full_metadata.csv.gz"
    parameters_path = metadata_dir / "qf_log_parameters.json"
    summary_path = metadata_dir / "qf_log_generation_summary.csv"
    equations_path = metadata_dir / "qf_log_equation_model.txt"
    log_path = logs_dir / "qf_log_generation_log.txt"

    outputs = (
        ml_path,
        full_metadata_path,
        parameters_path,
        summary_path,
        equations_path,
        log_path,
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Output files already exist. Use --overwrite only when you intend "
            "to replace them:\n" + "\n".join(existing)
        )

    params = SimulationParameters(
        total_samples=args.n_samples,
        random_seed=args.seed,
        chunk_size=args.chunk_size,
    )

    logger = configure_logging(log_path)
    logger.info("Starting QF-LOG formula-based data generation.")
    logger.info("Root folder: %s", root)
    logger.info("Total samples: %s", f"{params.total_samples:,}")
    logger.info("Samples per class: %s", f"{params.total_samples // 4:,}")
    logger.info("Chunk size: %s", f"{params.chunk_size:,}")
    logger.info("Random seed: %d", params.random_seed)

    with parameters_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generator_name": "QF-LOG formula-based dataset generator",
                "observable_ml_features": [
                    "qber",
                    "photon_count",
                    "latency_ms",
                    "abort_flag",
                ],
                "class_labels": list(CLASSES),
                "parameters": asdict(params),
                "generation_note": (
                    "All observable features are computed from documented equations. "
                    "Latency and abort are explicit controller-level simulation models."
                ),
            },
            file,
            indent=2,
        )

    equations_path.write_text(equation_text(params), encoding="utf-8")

    rng = np.random.default_rng(params.random_seed)
    stats = initialise_stats()
    generated = 0
    first_chunk = True
    started = time.perf_counter()

    with gzip.open(full_metadata_path, mode="wt", encoding="utf-8", newline="") as metadata_file:
        while generated < params.total_samples:
            size = min(params.chunk_size, params.total_samples - generated)
            # This also protects exact balance in the last chunk.
            if size % len(CLASSES) != 0:
                raise ValueError(
                    "The final chunk is not divisible by 4; use compatible n-samples "
                    "and chunk-size values."
                )

            ml_chunk, metadata_chunk = generate_chunk(
                rng=rng,
                params=params,
                start_sample_id=generated + 1,
                size=size,
            )
            validate_chunk(ml_chunk, params)
            update_stats(stats, ml_chunk)

            ml_chunk.to_csv(
                ml_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            metadata_chunk.to_csv(
                metadata_file,
                header=first_chunk,
                index=False,
            )

            generated += size
            first_chunk = False
            logger.info(
                "Generated %s / %s rows (%.1f%%).",
                f"{generated:,}",
                f"{params.total_samples:,}",
                100.0 * generated / params.total_samples,
            )

    summary = finalise_summary(stats)
    summary.to_csv(summary_path, index=False)

    elapsed = time.perf_counter() - started
    logger.info("Data generation completed in %.2f seconds.", elapsed)
    logger.info("ML dataset: %s", ml_path)
    logger.info("Full metadata dataset: %s", full_metadata_path)
    logger.info("Generation summary:\n%s", summary.to_string(index=False))

    print("\nGeneration complete. Class-wise summary:")
    print(summary.to_string(index=False))
    print(f"\nML dataset saved to: {ml_path}")
    print(f"Full metadata saved to: {full_metadata_path}")
    print(f"Equation model saved to: {equations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
