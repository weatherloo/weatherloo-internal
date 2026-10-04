#!/usr/bin/env python3
"""Compatibility entry point; remove after the pipeline path deprecation window."""

from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parents[3]
runpy.run_path(
    str(ROOT / "pipelines" / "benchmarking" / "persistence" / "compute_benchmark.py"),
    run_name="__main__",
)
