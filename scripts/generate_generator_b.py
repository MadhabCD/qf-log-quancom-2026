#!/usr/bin/env python3
"""
Generator-B: physics-inspired simulation-based QKD forensic log generator.

Purpose
-------
Generate a fresh cross-generator evaluation dataset for the four observable
QF-LOG features used in the study:
    qber, photon_count, latency_ms, abort_flag

and the four class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Methodological constraint
-------------------------
Generator-B uses the same observable feature equations as Generator-A:
    1. photon_count from channel transmission and detector-click probability;
    2. qber from generated error events among observed clicks;
    3. latency_ms from controller-level QBER/loss workload and jitter;
    4. abort_flag from a probabilistic controller abort rule.

Generator-B changes only hidden simulated operating conditions: hardware
profiles, fiber/channel conditions, detector calibration drift, controller
jitter, temporal instability, and attack-intensity distributions. It is
intended for cross-generator robustness testing, not for model training.

Interpretation limits
---------------------
The dataset is synthetic. The detector_blind class is a monitored response-
suppression proxy for a detector-control/blinding condition; it is not a
universal hardware trace. Fiber_tap represents added optical attenuation or
signal diversion, not copying an unknown quantum state.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import pandas as pd


CLASSES = (
    "normal",
    "partial_intercept_resend",
    "detector_blind",
    "fiber_tap",
)

HARDWARE_PROFILES = (
    "shifted_standard",
    "low_efficiency_detector",
    "longer_fiber_link",
)

PROFILE_PROBABILITIES = np.asarray([0.45, 0.30, 0.25], dtype=np.float64)


@dataclass(frozen=True)
class GeneratorBParameters:
    """Configured parameters for the fresh Generator-B simulator."""

    # Dataset design
    total_samples: int = 5_000_000
    random_seed: int = 84
    frames_per_session: int = 500
    sessions_per_chunk: int = 200
    pulses_per_frame: int = 16_000

    # Generator-B hardware profile distributions. Hardware profiles are
    # sampled independently of attack labels.
    standard_mu_mean: float = 0.49
    standard_length_km_mean: float = 12.0
    standard_attenuation_mean: float = 0.205
    standard_efficiency_mean: float = 0.190
    standard_background_yield_mean: float = 6.0e-5
    standard_misalignment_mean: float = 0.022
    standard_base_latency_mean: float = 4.72

    low_eff_mu_mean: float = 0.48
    low_eff_length_km_mean: float = 12.5
    low_eff_attenuation_mean: float = 0.208
    low_eff_efficiency_mean: float = 0.166
    low_eff_background_yield_mean: float = 7.5e-5
    low_eff_misalignment_mean: float = 0.026
    low_eff_base_latency_mean: float = 4.88

    long_fiber_mu_mean: float = 0.50
    long_fiber_length_km_mean: float = 18.0
    long_fiber_attenuation_mean: float = 0.210
    long_fiber_efficiency_mean: float = 0.185
    long_fiber_background_yield_mean: float = 7.0e-5
    long_fiber_misalignment_mean: float = 0.028
    long_fiber_base_latency_mean: float = 5.00

    # Within-profile bounded variability
    mean_photon_number_sd: float = 0.050
    mean_photon_number_low: float = 0.30
    mean_photon_number_high: float = 0.68

    fiber_length_km_sd: float = 2.50
    fiber_length_km_low: float = 4.0
    fiber_length_km_high: float = 28.0

    attenuation_db_per_km_sd: float = 0.014
    attenuation_db_per_km_low: float = 0.17
    attenuation_db_per_km_high: float = 0.25

    detector_efficiency_sd: float = 0.020
    detector_efficiency_low: float = 0.10
    detector_efficiency_high: float = 0.25

    background_yield_sd: float = 2.0e-5
    background_yield_low: float = 1.0e-5
    background_yield_high: float = 1.50e-4

    misalignment_error_sd: float = 0.009
    misalignment_error_low: float = 0.005
    misalignment_error_high: float = 0.070

    base_latency_ms_sd: float = 0.24
    base_latency_ms_low: float = 4.00
    base_latency_ms_high: float = 5.70

    # Temporal drift: d_t = rho*d_(t-1) + epsilon_t
    temporal_drift_rho: float = 0.96
    channel_loss_drift_initial_sd_db: float = 0.08
    channel_loss_drift_innovation_sd_db: float = 0.025
    detector_calibration_drift_initial_sd: float = 0.025
    detector_calibration_drift_innovation_sd: float = 0.008
    alignment_drift_initial_sd: float = 0.0020
    alignment_drift_innovation_sd: float = 0.0008
    controller_drift_initial_sd_ms: float = 0.08
    controller_drift_innovation_sd_ms: float = 0.025

    # Shifted attack intensity distributions. These make weak attack
    # conditions more frequent and create more overlap than Generator-A.
    intercept_fraction_lower: float = 0.015
    intercept_fraction_scale: float = 0.34
    intercept_fraction_beta_alpha: float = 1.4
    intercept_fraction_beta_beta: float = 3.0
    intercept_frame_jitter_sd: float = 0.010
    intercept_resend_error_on_attacked_fraction: float = 0.25

    tap_loss_db_lower: float = 0.04
    tap_loss_db_scale: float = 1.80
    tap_loss_beta_alpha: float = 1.4
    tap_loss_beta_beta: float = 2.8
    tap_loss_frame_jitter_sd: float = 0.06

    blind_response_lower: float = 0.18
    blind_response_scale: float = 0.80
    blind_response_beta_alpha: float = 1.7
    blind_response_beta_beta: float = 1.9
    blind_response_frame_jitter_sd: float = 0.025
    blind_excess_error_scale: float = 0.020
    blind_error_beta_alpha: float = 1.2
    blind_error_beta_beta: float = 3.2
    blind_error_frame_jitter_sd: float = 0.0015

    # Controller-level latency model. Equation is unchanged; parameter
    # values represent a shifted controller environment.
    error_correction_latency_weight_ms: float = 5.50
    loss_handling_latency_weight_ms: float = 2.80
    latency_jitter_ms_sd: float = 0.24

    # Operational abort model: kept identical to Generator-A so changes in
    # abort behaviour result from shifted observations rather than a changed
    # decision rule.
    spontaneous_abort_probability: float = 0.0005
    abort_qber_center: float = 0.105
    abort_qber_slope: float = 90.0
    abort_loss_center: float = 0.55
    abort_loss_slope: float = 18.0
    abort_latency_center_ms: float = 8.00
    abort_latency_slope: float = 3.00


def bounded_normal(
    rng: np.random.Generator,
    mean: np.ndarray | float,
    sd: float,
    low: float,
    high: float,
    size: int,
) -> np.ndarray:
    """Draw a normal random variable and restrict it to documented bounds."""
    return np.clip(rng.normal(mean, sd, size=size), low, high)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute a numerically stable logistic function."""
    out = np.empty_like(x, dtype=np.float64)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def binary_entropy(q: np.ndarray) -> np.ndarray:
    """Return binary entropy H2(q), with safe clipping for logarithms."""
    q_safe = np.clip(q, 1e-12, 1.0 - 1e-12)
    return -(q_safe * np.log2(q_safe) + (1.0 - q_safe) * np.log2(1.0 - q_safe))


def ar1_paths(
    rng: np.random.Generator,
    n_sessions: int,
    frames_per_session: int,
    rho: float,
    initial_sd: float,
    innovation_sd: float,
) -> np.ndarray:
    """
    Generate session-level temporal drift paths:
        d_(s,t) = rho*d_(s,t-1) + epsilon_(s,t).
    """
    paths = np.empty((n_sessions, frames_per_session), dtype=np.float64)
    paths[:, 0] = rng.normal(0.0, initial_sd, size=n_sessions)
    innovations = rng.normal(
        0.0, innovation_sd, size=(n_sessions, frames_per_session)
    )
    for time_step in range(1, frames_per_session):
        paths[:, time_step] = (
            rho * paths[:, time_step - 1] + innovations[:, time_step]
        )
    return paths


def balanced_session_labels(rng: np.random.Generator, n_sessions: int) -> np.ndarray:
    """Create balanced attack-class sessions and randomly reorder sessions."""
    if n_sessions % len(CLASSES) != 0:
        raise ValueError("The number of sessions must be divisible by four.")
    labels = np.repeat(np.asarray(CLASSES, dtype=object), n_sessions // len(CLASSES))
    rng.shuffle(labels)
    return labels


def profile_means(
    profile_names: np.ndarray, params: GeneratorBParameters
) -> Dict[str, np.ndarray]:
    """Map hardware profile names to per-session mean values."""
    definitions = {
        "shifted_standard": {
            "mu": params.standard_mu_mean,
            "length": params.standard_length_km_mean,
            "attenuation": params.standard_attenuation_mean,
            "efficiency": params.standard_efficiency_mean,
            "background": params.standard_background_yield_mean,
            "misalignment": params.standard_misalignment_mean,
            "latency": params.standard_base_latency_mean,
        },
        "low_efficiency_detector": {
            "mu": params.low_eff_mu_mean,
            "length": params.low_eff_length_km_mean,
            "attenuation": params.low_eff_attenuation_mean,
            "efficiency": params.low_eff_efficiency_mean,
            "background": params.low_eff_background_yield_mean,
            "misalignment": params.low_eff_misalignment_mean,
            "latency": params.low_eff_base_latency_mean,
        },
        "longer_fiber_link": {
            "mu": params.long_fiber_mu_mean,
            "length": params.long_fiber_length_km_mean,
            "attenuation": params.long_fiber_attenuation_mean,
            "efficiency": params.long_fiber_efficiency_mean,
            "background": params.long_fiber_background_yield_mean,
            "misalignment": params.long_fiber_misalignment_mean,
            "latency": params.long_fiber_base_latency_mean,
        },
    }
    result: Dict[str, np.ndarray] = {}
    for key in ("mu", "length", "attenuation", "efficiency", "background", "misalignment", "latency"):
        result[key] = np.asarray(
            [definitions[str(profile)][key] for profile in profile_names],
            dtype=np.float64,
        )
    return result


def generate_session_chunk(
    rng: np.random.Generator,
    params: GeneratorBParameters,
    session_ids: np.ndarray,
    session_labels: np.ndarray,
    hardware_profiles: np.ndarray,
    start_sample_id: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate one chunk of temporally structured Generator-B records."""
    n_sessions = len(session_ids)
    frames = params.frames_per_session
    size = n_sessions * frames

    means = profile_means(hardware_profiles, params)

    # Session-level baseline conditions.
    mu_session = bounded_normal(
        rng, means["mu"], params.mean_photon_number_sd,
        params.mean_photon_number_low, params.mean_photon_number_high, n_sessions
    )
    fiber_length_session = bounded_normal(
        rng, means["length"], params.fiber_length_km_sd,
        params.fiber_length_km_low, params.fiber_length_km_high, n_sessions
    )
    attenuation_session = bounded_normal(
        rng, means["attenuation"], params.attenuation_db_per_km_sd,
        params.attenuation_db_per_km_low, params.attenuation_db_per_km_high, n_sessions
    )
    detector_efficiency_session = bounded_normal(
        rng, means["efficiency"], params.detector_efficiency_sd,
        params.detector_efficiency_low, params.detector_efficiency_high, n_sessions
    )
    background_yield_session = bounded_normal(
        rng, means["background"], params.background_yield_sd,
        params.background_yield_low, params.background_yield_high, n_sessions
    )
    misalignment_session = bounded_normal(
        rng, means["misalignment"], params.misalignment_error_sd,
        params.misalignment_error_low, params.misalignment_error_high, n_sessions
    )
    latency_session = bounded_normal(
        rng, means["latency"], params.base_latency_ms_sd,
        params.base_latency_ms_low, params.base_latency_ms_high, n_sessions
    )

    # Repeated identifiers and session conditions for frame-level records.
    label = np.repeat(session_labels, frames)
    profile = np.repeat(hardware_profiles, frames)
    session_id = np.repeat(session_ids, frames)
    time_step = np.tile(np.arange(frames, dtype=np.int32), n_sessions)

    mu = np.repeat(mu_session, frames)
    fiber_length = np.repeat(fiber_length_session, frames)
    attenuation = np.repeat(attenuation_session, frames)
    background_yield = np.repeat(background_yield_session, frames)

    # Temporal operational drift, independent of the attack label.
    channel_loss_drift_db = ar1_paths(
        rng, n_sessions, frames, params.temporal_drift_rho,
        params.channel_loss_drift_initial_sd_db,
        params.channel_loss_drift_innovation_sd_db,
    ).ravel()
    detector_calibration_drift = ar1_paths(
        rng, n_sessions, frames, params.temporal_drift_rho,
        params.detector_calibration_drift_initial_sd,
        params.detector_calibration_drift_innovation_sd,
    ).ravel()
    alignment_drift = ar1_paths(
        rng, n_sessions, frames, params.temporal_drift_rho,
        params.alignment_drift_initial_sd,
        params.alignment_drift_innovation_sd,
    ).ravel()
    controller_drift_ms = ar1_paths(
        rng, n_sessions, frames, params.temporal_drift_rho,
        params.controller_drift_initial_sd_ms,
        params.controller_drift_innovation_sd_ms,
    ).ravel()

    detector_efficiency = np.clip(
        np.repeat(detector_efficiency_session, frames) * (1.0 + detector_calibration_drift),
        params.detector_efficiency_low,
        params.detector_efficiency_high,
    )
    misalignment_error = np.clip(
        np.repeat(misalignment_session, frames) + alignment_drift,
        params.misalignment_error_low,
        params.misalignment_error_high,
    )

    # Session-level attack intensities followed by frame-level variation.
    ir_session_mask = session_labels == "partial_intercept_resend"
    tap_session_mask = session_labels == "fiber_tap"
    blind_session_mask = session_labels == "detector_blind"

    intercept_session = np.zeros(n_sessions, dtype=np.float64)
    tap_loss_session = np.zeros(n_sessions, dtype=np.float64)
    blind_response_session = np.ones(n_sessions, dtype=np.float64)
    blind_error_session = np.zeros(n_sessions, dtype=np.float64)

    intercept_session[ir_session_mask] = (
        params.intercept_fraction_lower
        + params.intercept_fraction_scale
        * rng.beta(
            params.intercept_fraction_beta_alpha,
            params.intercept_fraction_beta_beta,
            int(ir_session_mask.sum()),
        )
    )
    tap_loss_session[tap_session_mask] = (
        params.tap_loss_db_lower
        + params.tap_loss_db_scale
        * rng.beta(
            params.tap_loss_beta_alpha,
            params.tap_loss_beta_beta,
            int(tap_session_mask.sum()),
        )
    )
    blind_response_session[blind_session_mask] = (
        params.blind_response_lower
        + params.blind_response_scale
        * rng.beta(
            params.blind_response_beta_alpha,
            params.blind_response_beta_beta,
            int(blind_session_mask.sum()),
        )
    )
    blind_error_session[blind_session_mask] = (
        params.blind_excess_error_scale
        * rng.beta(
            params.blind_error_beta_alpha,
            params.blind_error_beta_beta,
            int(blind_session_mask.sum()),
        )
    )

    ir_mask = label == "partial_intercept_resend"
    tap_mask = label == "fiber_tap"
    blind_mask = label == "detector_blind"

    intercept_fraction = np.zeros(size, dtype=np.float64)
    intercept_fraction[ir_mask] = np.clip(
        np.repeat(intercept_session, frames)[ir_mask]
        + rng.normal(0.0, params.intercept_frame_jitter_sd, int(ir_mask.sum())),
        0.0,
        0.50,
    )

    tap_loss_db = np.zeros(size, dtype=np.float64)
    tap_loss_db[tap_mask] = np.clip(
        np.repeat(tap_loss_session, frames)[tap_mask]
        + rng.normal(0.0, params.tap_loss_frame_jitter_sd, int(tap_mask.sum())),
        0.0,
        params.tap_loss_db_lower + params.tap_loss_db_scale,
    )

    detector_response_factor = np.ones(size, dtype=np.float64)
    detector_response_factor[blind_mask] = np.clip(
        np.repeat(blind_response_session, frames)[blind_mask]
        + rng.normal(0.0, params.blind_response_frame_jitter_sd, int(blind_mask.sum())),
        params.blind_response_lower,
        1.0,
    )

    blind_excess_error = np.zeros(size, dtype=np.float64)
    blind_excess_error[blind_mask] = np.clip(
        np.repeat(blind_error_session, frames)[blind_mask]
        + rng.normal(0.0, params.blind_error_frame_jitter_sd, int(blind_mask.sum())),
        0.0,
        params.blind_excess_error_scale,
    )

    # ---------------------------------------------------------------------
    # Observable feature equations: identical structure to Generator-A.
    # ---------------------------------------------------------------------
    baseline_loss_db = np.maximum(
        attenuation * fiber_length + channel_loss_drift_db, 0.0
    )
    baseline_channel_transmission = np.power(10.0, -baseline_loss_db / 10.0)
    attacked_channel_transmission = np.power(
        10.0, -(baseline_loss_db + tap_loss_db) / 10.0
    )

    reference_total_efficiency = detector_efficiency * baseline_channel_transmission
    attacked_total_efficiency = (
        detector_efficiency * attacked_channel_transmission * detector_response_factor
    )

    reference_click_probability = 1.0 - (1.0 - background_yield) * np.exp(
        -mu * reference_total_efficiency
    )
    attacked_signal_probability = 1.0 - np.exp(-mu * attacked_total_efficiency)
    observed_click_probability = 1.0 - (1.0 - background_yield) * np.exp(
        -mu * attacked_total_efficiency
    )

    expected_reference_count = params.pulses_per_frame * reference_click_probability
    photon_count = rng.binomial(
        params.pulses_per_frame, observed_click_probability
    ).astype(np.int32)

    q_base = (
        0.5 * background_yield + misalignment_error * attacked_signal_probability
    ) / np.maximum(observed_click_probability, 1e-12)

    q_true = q_base.copy()
    q_true[ir_mask] = (
        (1.0 - intercept_fraction[ir_mask]) * q_base[ir_mask]
        + intercept_fraction[ir_mask] * params.intercept_resend_error_on_attacked_fraction
    )
    q_true[blind_mask] = q_base[blind_mask] + blind_excess_error[blind_mask]
    q_true = np.clip(q_true, 0.0, 0.5)

    error_count = rng.binomial(photon_count, q_true).astype(np.int32)
    qber = np.divide(
        error_count, photon_count,
        out=np.zeros(size, dtype=np.float64),
        where=photon_count > 0,
    )

    observed_loss_fraction = np.clip(
        1.0 - photon_count / np.maximum(expected_reference_count, 1.0), 0.0, 1.0
    )
    base_latency = np.clip(
        np.repeat(latency_session, frames) + controller_drift_ms,
        params.base_latency_ms_low,
        params.base_latency_ms_high + 0.5,
    )
    latency_jitter = rng.normal(0.0, params.latency_jitter_ms_sd, size=size)
    latency_ms = (
        base_latency
        + params.error_correction_latency_weight_ms * binary_entropy(qber)
        + params.loss_handling_latency_weight_ms * observed_loss_fraction
        + latency_jitter
    )
    latency_ms = np.maximum(latency_ms, 0.0)

    p_abort_qber = sigmoid(params.abort_qber_slope * (qber - params.abort_qber_center))
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
            "label": label,
        }
    )

    metadata = pd.DataFrame(
        {
            "sample_id": sample_id,
            "session_id": session_id,
            "time_step": time_step,
            "hardware_profile": profile,
            "label": label,
            "qber": np.round(qber, 8),
            "photon_count": photon_count,
            "latency_ms": np.round(latency_ms, 6),
            "abort_flag": abort_flag,
            "error_count": error_count,
            "pulses_per_frame": params.pulses_per_frame,
            "mean_photon_number": np.round(mu, 8),
            "fiber_length_km": np.round(fiber_length, 6),
            "attenuation_db_per_km": np.round(attenuation, 8),
            "channel_loss_drift_db": np.round(channel_loss_drift_db, 8),
            "detector_efficiency": np.round(detector_efficiency, 8),
            "detector_calibration_drift": np.round(detector_calibration_drift, 8),
            "background_yield": np.round(background_yield, 10),
            "misalignment_error": np.round(misalignment_error, 8),
            "alignment_drift": np.round(alignment_drift, 8),
            "controller_drift_ms": np.round(controller_drift_ms, 8),
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


def validate_chunk(data: pd.DataFrame, params: GeneratorBParameters) -> None:
    """Validate generated observable outputs before writing them to disk."""
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


def initialise_stats() -> Dict[str, Dict[str, float]]:
    """Initialise running class-wise summary statistics."""
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
    """Update running class-wise summary statistics."""
    for label, group in data.groupby("label", sort=False):
        entry = stats[str(label)]
        n = len(group)
        entry["n"] += n
        for column in ("qber", "photon_count", "latency_ms"):
            values = group[column].to_numpy(dtype=np.float64)
            entry[f"{column}_sum"] += float(values.sum())
            entry[f"{column}_sq_sum"] += float(np.square(values).sum())
            entry[f"{column}_min"] = min(entry[f"{column}_min"], float(values.min()))
            entry[f"{column}_max"] = max(entry[f"{column}_max"], float(values.max()))
        entry["abort_flag_sum"] += float(group["abort_flag"].sum())


def finalise_summary(stats: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """Return class-wise mean, standard deviation, range, and abort rate."""
    rows: list[Dict[str, object]] = []
    for label in CLASSES:
        entry = stats[label]
        n = int(entry["n"])
        row: Dict[str, object] = {"label": label, "samples": n}
        for column in ("qber", "photon_count", "latency_ms"):
            mean = entry[f"{column}_sum"] / n
            variance = max(entry[f"{column}_sq_sum"] / n - mean * mean, 0.0)
            row[f"{column}_mean"] = mean
            row[f"{column}_std"] = math.sqrt(variance)
            row[f"{column}_min"] = entry[f"{column}_min"]
            row[f"{column}_max"] = entry[f"{column}_max"]
        row["abort_rate"] = entry["abort_flag_sum"] / n
        rows.append(row)
    return pd.DataFrame(rows)


def shift_specification() -> pd.DataFrame:
    """Document the intended operational shifts from Generator-A to Generator-B."""
    rows = [
        ("Feature equations", "Generator-A equations", "Same equations", "Only latent conditions are shifted"),
        ("Data structure", "Independent frames", "500-frame sessions", "Temporal robustness evaluation"),
        ("Fiber length (km)", "N(10.0,2.5), [3,20]", "Profile means 12.0, 12.5, 18.0; [4,28]", "Unseen link conditions"),
        ("Attenuation (dB/km)", "N(0.20,0.01), [0.17,0.23]", "Profile means 0.205-0.210; [0.17,0.25]", "Channel variability"),
        ("Detector efficiency", "N(0.20,0.018), [0.14,0.26]", "Profile means 0.166-0.190; [0.10,0.25]", "Hardware variability"),
        ("Background yield", "N(5e-5,1e-5), [1e-5,9e-5]", "Profile means 6e-5-7.5e-5; [1e-5,1.5e-4]", "Noise variability"),
        ("Misalignment error", "N(0.018,0.006), [0.005,0.040]", "Profile means 0.022-0.028; [0.005,0.070]", "Calibration variation"),
        ("Temporal drift", "Not applied", "AR(1) channel, detector, alignment and controller drift", "Time-dependent instability"),
        ("Partial intercept-resend", "0.03 + 0.42 Beta(1.5,2.8)", "0.015 + 0.34 Beta(1.4,3.0)", "Weaker and more overlapping activity"),
        ("Fiber-tap attenuation", "0.10 + 2.50 Beta(1.5,2.2)", "0.04 + 1.80 Beta(1.4,2.8)", "Subtler attenuation shift"),
        ("Detector-blind response", "0.05 + 0.90 Beta(1.4,2.2)", "0.18 + 0.80 Beta(1.7,1.9)", "Reduced separability"),
        ("Abort decision rule", "Configured controller rule", "Identical rule", "No label-specific policy shift"),
    ]
    return pd.DataFrame(rows, columns=["component", "generator_a", "generator_b", "purpose"])


def equation_text(params: GeneratorBParameters) -> str:
    """Return an auditable mathematical model summary saved with the dataset."""
    return f"""Generator-B Simulation and Cross-Generator Shift Model
====================================================

Observable ML features:
    qber, photon_count, latency_ms, abort_flag

Class labels:
    normal, partial_intercept_resend, detector_blind, fiber_tap

Dataset design:
    Records: {params.total_samples:,}
    Frames per session: {params.frames_per_session}
    Pulses per frame: {params.pulses_per_frame:,}

Core rule:
    Generator-B uses the same observable feature equations as Generator-A.
    Generator-B alters hidden operational parameter distributions and adds
    documented temporal drift; it does not directly shift feature values.

Temporal drift:
    d_(s,t) = rho * d_(s,t-1) + epsilon_(s,t)
    rho = {params.temporal_drift_rho}

Channel and photon-count model:
    baseline_loss_db = alpha * L + channel_loss_drift_db
    eta_ch,0 = 10^[-baseline_loss_db / 10]
    eta_ch = 10^[-(baseline_loss_db + a_tap) / 10]
    eta = eta_d * eta_ch * r_DB
    p_click = 1 - (1 - Y0) * exp(-mu * eta)
    photon_count ~ Binomial(N, p_click)

QBER model:
    p_sig = 1 - exp(-mu * eta)
    q_base = [0.5*Y0 + e_d*p_sig] / p_click
    normal and fiber_tap: q_true = q_base
    partial_intercept_resend:
        q_true = (1 - f_IR)*q_base + f_IR*0.25
    detector_blind proxy:
        q_true = q_base + delta_DB
    error_count ~ Binomial(photon_count, q_true)
    qber = error_count / photon_count when photon_count > 0

Controller latency model:
    R_loss = max(0, 1 - photon_count / expected_reference_count)
    H2(q) = -q*log2(q) - (1-q)*log2(1-q)
    latency_ms = T0 + controller_drift_ms
                 + k_EC*H2(qber) + k_loss*R_loss + epsilon_T
    k_EC = {params.error_correction_latency_weight_ms}
    k_loss = {params.loss_handling_latency_weight_ms}

Abort model:
    p_Q = sigmoid({params.abort_qber_slope} * (qber - {params.abort_qber_center}))
    p_L = sigmoid({params.abort_loss_slope} * (R_loss - {params.abort_loss_center}))
    p_T = sigmoid({params.abort_latency_slope} * (latency_ms - {params.abort_latency_center_ms}))
    P(abort=1) = 1 - (1-p0)(1-p_Q)(1-p_L)(1-p_T)
    abort_flag ~ Bernoulli(P(abort=1))

Interpretation:
    Latency and abort are controller-level simulation rules.
    Fiber tap represents optical attenuation/signal diversion.
    Detector blind represents a monitored response-suppression proxy.
    No latent metadata parameter is an ML input feature.
"""


def configure_logging(path: Path) -> logging.Logger:
    """Configure file and console logging."""
    logger = logging.getLogger("generator_b")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_root = r"C:\Users\madha\OneDrive\Desktop\QKD QF-Log Forensic"
    parser = argparse.ArgumentParser(
        description="Generate fresh Generator-B cross-generator QF-LOG data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(default_root),
        help="Project root folder.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5_000_000,
        help="Total frame records. Must equal frames-per-session times a session count divisible by four.",
    )
    parser.add_argument(
        "--frames-per-session",
        type=int,
        default=500,
        help="Number of temporal frames in each simulated session.",
    )
    parser.add_argument(
        "--sessions-per-chunk",
        type=int,
        default=200,
        help="Number of sessions generated in each write chunk.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=84,
        help="Random seed for Generator-B.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing Generator-B outputs.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()

    if args.n_samples <= 0 or args.frames_per_session <= 0:
        raise ValueError("--n-samples and --frames-per-session must be positive.")
    if args.n_samples % args.frames_per_session != 0:
        raise ValueError("--n-samples must be divisible by --frames-per-session.")

    total_sessions = args.n_samples // args.frames_per_session
    if total_sessions % len(CLASSES) != 0:
        raise ValueError("Total number of sessions must be divisible by four.")
    if args.sessions_per_chunk <= 0 or args.sessions_per_chunk % len(CLASSES) != 0:
        raise ValueError("--sessions-per-chunk must be positive and divisible by four.")

    root = args.root.expanduser()
    data_dir = root / "data"
    metadata_dir = root / "metadata"
    logs_dir = root / "logs"
    for directory in (data_dir, metadata_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    ml_path = data_dir / "generator_b_qf_log_dataset.csv"
    full_metadata_path = data_dir / "generator_b_full_metadata.csv.gz"
    parameters_path = metadata_dir / "generator_b_parameters.json"
    summary_path = metadata_dir / "generator_b_generation_summary.csv"
    shift_path = metadata_dir / "generator_b_shift_specification.csv"
    equations_path = metadata_dir / "generator_b_equation_model.txt"
    log_path = logs_dir / "generator_b_generation_log.txt"

    outputs = (
        ml_path, full_metadata_path, parameters_path, summary_path,
        shift_path, equations_path, log_path
    )
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Generator-B output files already exist. Use --overwrite only when "
            "you intentionally want to replace them:\n" + "\n".join(existing)
        )

    params = GeneratorBParameters(
        total_samples=args.n_samples,
        random_seed=args.seed,
        frames_per_session=args.frames_per_session,
        sessions_per_chunk=args.sessions_per_chunk,
    )

    logger = configure_logging(log_path)
    logger.info("Starting fresh Generator-B simulation data generation.")
    logger.info("Root folder: %s", root)
    logger.info("Total records: %s", f"{params.total_samples:,}")
    logger.info("Total sessions: %s", f"{total_sessions:,}")
    logger.info("Frames per session: %s", f"{params.frames_per_session:,}")
    logger.info("Random seed: %d", params.random_seed)

    with parameters_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generator_name": "Generator-B cross-generator simulation dataset",
                "purpose": "Cross-generator robustness testing only",
                "observable_ml_features": [
                    "qber", "photon_count", "latency_ms", "abort_flag"
                ],
                "class_labels": list(CLASSES),
                "hardware_profiles": list(HARDWARE_PROFILES),
                "hardware_profile_probabilities": PROFILE_PROBABILITIES.tolist(),
                "parameters": asdict(params),
                "method_note": (
                    "Generator-B uses the same observable-feature equations as "
                    "Generator-A and shifts only hidden simulated operating "
                    "conditions and temporal drift."
                ),
            },
            file,
            indent=2,
        )

    shift_specification().to_csv(shift_path, index=False)
    equations_path.write_text(equation_text(params), encoding="utf-8")

    rng = np.random.default_rng(params.random_seed)
    session_labels = balanced_session_labels(rng, total_sessions)
    profiles = rng.choice(
        np.asarray(HARDWARE_PROFILES, dtype=object),
        size=total_sessions,
        p=PROFILE_PROBABILITIES,
    )
    session_ids = np.arange(1, total_sessions + 1, dtype=np.int64)

    stats = initialise_stats()
    generated_records = 0
    generated_sessions = 0
    first_chunk = True
    started = time.perf_counter()

    with gzip.open(full_metadata_path, mode="wt", encoding="utf-8", newline="") as metadata_file:
        while generated_sessions < total_sessions:
            stop = min(
                generated_sessions + params.sessions_per_chunk, total_sessions
            )
            chunk_session_ids = session_ids[generated_sessions:stop]
            chunk_labels = session_labels[generated_sessions:stop]
            chunk_profiles = profiles[generated_sessions:stop]

            ml_chunk, metadata_chunk = generate_session_chunk(
                rng=rng,
                params=params,
                session_ids=chunk_session_ids,
                session_labels=chunk_labels,
                hardware_profiles=chunk_profiles,
                start_sample_id=generated_records + 1,
            )
            validate_chunk(ml_chunk, params)
            update_stats(stats, ml_chunk)

            ml_chunk.to_csv(
                ml_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            metadata_chunk.to_csv(metadata_file, header=first_chunk, index=False)

            generated_sessions = stop
            generated_records += len(ml_chunk)
            first_chunk = False
            logger.info(
                "Generated %s / %s records (%.1f%%).",
                f"{generated_records:,}",
                f"{params.total_samples:,}",
                100.0 * generated_records / params.total_samples,
            )

    summary = finalise_summary(stats)
    expected_per_class = params.total_samples // len(CLASSES)
    if not (summary["samples"] == expected_per_class).all():
        raise ValueError(
            "Final Generator-B dataset is not class-balanced as specified. "
            "No result files should be used."
        )
    summary.to_csv(summary_path, index=False)

    elapsed = time.perf_counter() - started
    logger.info("Generator-B data generation completed in %.2f seconds.", elapsed)
    logger.info("ML dataset: %s", ml_path)
    logger.info("Full metadata: %s", full_metadata_path)
    logger.info("Generation summary:\n%s", summary.to_string(index=False))

    print("\nGenerator-B generation complete. Class-wise summary:")
    print(summary.to_string(index=False))
    print(f"\nML dataset saved to: {ml_path}")
    print(f"Full metadata saved to: {full_metadata_path}")
    print(f"Shift specification saved to: {shift_path}")
    print(f"Equation model saved to: {equations_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
