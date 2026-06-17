# QF-LOG: Simulation-Based Quantum Forensics for QKD

This repository contains the code, metadata, and evaluation outputs for the paper:

**Simulation-Based Quantum Forensics for QKD: Machine Learning Attack Detection and Signature Extraction**

## Overview

QF-LOG is a simulation-based forensic framework for analyzing aggregated Quantum Key Distribution operational logs. The study uses four observable features:

- qber
- photon_count
- latency_ms
- abort_flag

The four simulated classes are:

- normal
- partial_intercept_resend
- detector_blind
- fiber_tap

## Scope

The datasets in this repository are synthetic and generated under controlled simulation settings. The results are not hardware validated and should not be interpreted as deployment-ready QKD attack detection.

## Repository Structure

- scripts/: dataset generation and evaluation scripts
- metadata/: Generator-A and Generator-B settings
- results/: evaluation metrics, figures, diagnostics, and logs
- docs/: reproduction steps and data dictionary

## Installation

pip install -r requirements.txt

## Reproduction

Run the scripts in this order:

python scripts/generate_qf_log_dataset.py
python scripts/generate_generator_b.py
python scripts/evaluate_qf_log_models.py
python scripts/evaluate_cross_generator_robustness.py
python scripts/evaluate_qf_log_ablation_stability.py
python scripts/evaluate_qf_log_feature_importance_clean.py

## Results

Main outputs are stored in:

- results/model_evaluation/
- results/cross_generator_evaluation/
- results/feature_importance/
- results/extended_validation/

## Citation

If you use this repository, please cite the related QUANCOM 2026 paper.
