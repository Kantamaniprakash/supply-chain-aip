# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-05

### Added
- Medallion PySpark data pipeline (Bronze → Silver → Gold) with Foundry transforms for supplier, shipment, and geopolitical-risk ingestion and a Gold ML feature master table.
- Five-model ML risk engine: XGBoost disruption scorer (Optuna tuning + SHAP explainability), Temporal Fusion Transformer demand forecaster, Isolation Forest + ECOD anomaly detector, GraphSAGE supplier-network contagion GNN, and a Gaussian-copula Monte Carlo Value-at-Risk simulator.
- Model registry for versioning and deployment wrapping of the trained models.
- Palantir AIP agent layer connecting GPT-4o to live Foundry Ontology objects for root-cause analysis, action recommendations, and registered actions (purchase orders, supplier alerts, escalations, executive reports).
- Streamlit dashboard simulating a Palantir Foundry Workshop UI — command center, Monte Carlo VaR, anomaly detection, and the AIP intelligence agent chat.
- CI workflow (Python 3.10/3.11/3.12 matrix) with ruff lint and pytest smoke tests, plus Dependabot configuration, MIT license, and repository documentation.

[0.1.0]: https://github.com/Kantamaniprakash/supply-chain-aip/releases/tag/v0.1.0
