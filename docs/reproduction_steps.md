# Reproduction Steps

This document describes the reproduction workflow for the QF-LOG simulation-based QKD forensic study.

## Environment

Python 3.11 or later is recommended.

Install dependencies:

`ash
pip install -r requirements.txt

`

## Workflow

1. Generate Generator-A.
2. Generate Generator-B.
3. Train the evaluated models.
4. Evaluate Generator-A.
5. Evaluate Generator-B.
6. Run feature diagnostics.

## Scope

The generated datasets are synthetic. The workflow is intended for controlled simulation-based forensic analysis and is not hardware validated.
