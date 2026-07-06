"""Smoke tests for the five-model ML architecture.

The full dependency stack (torch, torch-geometric, pyspark, ...) is too heavy
to install in CI, so instead of importing the modules these tests assert that
each model module exists and parses as valid Python via ast.parse.
"""

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_MODULES = [
    "models/disruption_risk_model.py",   # Model 1: XGBoost disruption scorer
    "models/demand_forecast_tft.py",     # Model 2: Temporal Fusion Transformer
    "models/anomaly_detector.py",        # Model 3: Isolation Forest + ECOD
    "graph/supplier_network_gnn.py",     # Model 4: GraphSAGE contagion model
    "simulation/monte_carlo_var.py",     # Model 5: Monte Carlo VaR simulator
]


@pytest.mark.parametrize("relative_path", MODEL_MODULES)
def test_model_module_parses(relative_path):
    source_path = REPO_ROOT / relative_path
    assert source_path.is_file(), f"Missing model module: {relative_path}"
    ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
