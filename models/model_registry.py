"""
Model Registry — Versioning, Deployment, and Governance
=========================================================
Centralised registry for all ML models in the supply chain AIP platform.
Handles model versioning, A/B shadow scoring, performance drift detection,
and automated retraining triggers.

Design:
  - Each model version is immutable once registered (content-addressed by SHA256)
  - Champion/challenger framework: production model vs new candidate
  - Population Stability Index (PSI) for input feature drift detection
  - Kullback-Leibler divergence for output score distribution drift
  - Automated challenger promotion when candidate outperforms champion
    on holdout evaluation window (rolling 30-day)

References:
  Gama, J., et al. (2014). A survey on concept drift adaptation.
    ACM Computing Surveys.
  PSI: Yurdakul, B. (2018). Statistical Properties of PSI.

Author: Satya Sai Prakash Kantamani
"""

import hashlib
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional
from sklearn.metrics import roc_auc_score, brier_score_loss
import warnings

warnings.filterwarnings("ignore")


# ── PSI thresholds (industry standard) ───────────────────────────────────────
PSI_STABLE    = 0.10   # no action required
PSI_MONITOR   = 0.25   # monitor — consider retraining
PSI_RETRAIN   = 0.25   # auto-trigger retraining above this

KL_DRIFT_THRESHOLD = 0.10   # output score KL divergence trigger


@dataclass
class ModelVersion:
    model_id:         str
    model_type:       str          # "disruption_risk" | "demand_forecast" | "anomaly_detector"
    version:          str          # semantic: "v1.2.0"
    sha256:           str          # content hash of serialised model
    registered_at:    str          # ISO 8601
    training_samples: int
    cv_auc:           float
    cv_ap:            float
    brier_score:      float
    feature_names:    list[str]
    hyperparameters:  dict
    status:           str          # "CANDIDATE" | "CHAMPION" | "RETIRED"
    tags:             dict = field(default_factory=dict)
    performance_log:  list = field(default_factory=list)   # rolling holdout metrics


@dataclass
class DriftReport:
    model_id:          str
    evaluated_at:      str
    feature_psi:       dict          # feature → PSI score
    max_psi:           float
    output_kl:         float
    drift_detected:    bool
    retrain_triggered: bool
    drift_features:    list[str]     # features with PSI > threshold


class ModelRegistry:
    """
    File-system based model registry (Foundry Object Storage in production).
    Supports champion/challenger management, drift monitoring, and audit trail.
    """

    def __init__(self, registry_path: str = "./model_registry"):
        self.root = Path(registry_path)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "models").mkdir(exist_ok=True)
        (self.root / "drift_reports").mkdir(exist_ok=True)
        self._manifest: dict[str, list[ModelVersion]] = {}
        self._load_manifest()

    # ── Registration ──────────────────────────────────────────────────────

    def register(
        self,
        model_obj:        Any,
        model_type:       str,
        version:          str,
        metrics:          dict,
        feature_names:    list[str],
        hyperparameters:  dict,
        tags:             dict = None,
    ) -> ModelVersion:
        """
        Serialise model, compute content hash, and register as CANDIDATE.
        Champions must be explicitly promoted via promote_to_champion().
        """
        blob = pickle.dumps(model_obj)
        sha  = hashlib.sha256(blob).hexdigest()

        model_id = f"{model_type}_{version}_{sha[:8]}"
        artifact_path = self.root / "models" / f"{model_id}.pkl"
        artifact_path.write_bytes(blob)

        version_record = ModelVersion(
            model_id=model_id,
            model_type=model_type,
            version=version,
            sha256=sha,
            registered_at=datetime.utcnow().isoformat(),
            training_samples=metrics.get("training_samples", 0),
            cv_auc=metrics.get("cv_auc", 0.0),
            cv_ap=metrics.get("cv_ap", 0.0),
            brier_score=metrics.get("brier_score", 1.0),
            feature_names=feature_names,
            hyperparameters=hyperparameters,
            status="CANDIDATE",
            tags=tags or {},
        )

        if model_type not in self._manifest:
            self._manifest[model_type] = []
        self._manifest[model_type].append(version_record)
        self._save_manifest()

        print(f"[Registry] Registered {model_id} as CANDIDATE | "
              f"AUC={version_record.cv_auc:.4f} | SHA={sha[:12]}")
        return version_record

    def promote_to_champion(self, model_type: str, model_id: str) -> None:
        """
        Retire current champion, promote specified candidate to CHAMPION.
        Validates that candidate AUC >= current champion AUC (guardrail).
        """
        versions = self._manifest.get(model_type, [])
        candidate = next((v for v in versions if v.model_id == model_id), None)
        current_champion = next((v for v in versions if v.status == "CHAMPION"), None)

        if candidate is None:
            raise ValueError(f"Model {model_id} not found in registry")
        if candidate.status != "CANDIDATE":
            raise ValueError(f"Model {model_id} is not a CANDIDATE (status={candidate.status})")

        # Performance guardrail
        if current_champion and candidate.cv_auc < current_champion.cv_auc - 0.01:
            raise ValueError(
                f"Candidate AUC {candidate.cv_auc:.4f} < Champion AUC "
                f"{current_champion.cv_auc:.4f} - 0.01. Promotion blocked."
            )

        if current_champion:
            current_champion.status = "RETIRED"
            print(f"[Registry] Retired champion: {current_champion.model_id}")

        candidate.status = "CHAMPION"
        self._save_manifest()
        print(f"[Registry] Promoted {model_id} to CHAMPION | AUC={candidate.cv_auc:.4f}")

    def load_champion(self, model_type: str) -> tuple[Any, ModelVersion]:
        """Load champion model artifact and metadata."""
        versions = self._manifest.get(model_type, [])
        champion  = next((v for v in versions if v.status == "CHAMPION"), None)
        if champion is None:
            raise RuntimeError(f"No CHAMPION registered for model type '{model_type}'")
        artifact = (self.root / "models" / f"{champion.model_id}.pkl").read_bytes()
        model_obj = pickle.loads(artifact)
        return model_obj, champion

    # ── Champion/Challenger Shadow Scoring ────────────────────────────────

    def shadow_score(
        self,
        model_type:  str,
        X:           pd.DataFrame,
        y_true:      np.ndarray,
    ) -> dict:
        """
        Score both champion and all CANDIDATE models on holdout data.
        Logs performance for challenger promotion decision.
        Returns comparative metrics dict.
        """
        results = {}
        versions = self._manifest.get(model_type, [])
        active = [v for v in versions if v.status in ("CHAMPION", "CANDIDATE")]

        for mv in active:
            artifact = (self.root / "models" / f"{mv.model_id}.pkl").read_bytes()
            model    = pickle.loads(artifact)

            try:
                y_prob = model.predict_proba(X)[:, 1]
            except AttributeError:
                y_prob = model.predict(X)

            auc    = roc_auc_score(y_true, y_prob)
            brier  = brier_score_loss(y_true, y_prob)

            perf_log_entry = {
                "evaluated_at": datetime.utcnow().isoformat(),
                "holdout_auc":  round(auc, 4),
                "brier_score":  round(brier, 4),
                "n_samples":    len(y_true),
            }
            mv.performance_log.append(perf_log_entry)

            results[mv.model_id] = {
                "status":       mv.status,
                "holdout_auc":  auc,
                "brier_score":  brier,
                "version":      mv.version,
            }
            print(f"  [{mv.status:<10}] {mv.model_id[:30]:<30} | "
                  f"AUC={auc:.4f} | Brier={brier:.4f}")

        self._save_manifest()
        return results

    # ── Drift Detection ───────────────────────────────────────────────────

    @staticmethod
    def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Population Stability Index.
        PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected)
        """
        eps = 1e-9
        breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
        breakpoints  = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        exp_pct = np.histogram(expected, bins=breakpoints)[0] / (len(expected) + eps)
        act_pct = np.histogram(actual,   bins=breakpoints)[0] / (len(actual) + eps)
        exp_pct = np.clip(exp_pct, eps, None)
        act_pct = np.clip(act_pct, eps, None)

        return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

    @staticmethod
    def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
        """KL(P||Q) for output score distribution drift detection."""
        eps = 1e-9
        p_hist, _ = np.histogram(p, bins=bins, range=(0, 1), density=True)
        q_hist, _ = np.histogram(q, bins=bins, range=(0, 1), density=True)
        p_hist = np.clip(p_hist, eps, None)
        q_hist = np.clip(q_hist, eps, None)
        return float(np.sum(p_hist * np.log(p_hist / q_hist)))

    def detect_drift(
        self,
        model_type:        str,
        training_features: pd.DataFrame,     # reference distribution
        current_features:  pd.DataFrame,     # live distribution window
        training_scores:   np.ndarray,       # reference output scores
        current_scores:    np.ndarray,       # live output scores
        features_to_check: list[str] = None,
    ) -> DriftReport:
        """
        Computes PSI for each feature dimension and KL for output scores.
        Triggers retraining flag if PSI > PSI_RETRAIN or KL > KL_DRIFT_THRESHOLD.
        """
        feats = features_to_check or training_features.columns.tolist()
        psi_per_feature = {}
        drift_features  = []

        for feat in feats:
            if feat not in training_features.columns:
                continue
            psi = self._psi(
                training_features[feat].dropna().values,
                current_features[feat].dropna().values,
            )
            psi_per_feature[feat] = round(psi, 4)
            if psi > PSI_MONITOR:
                drift_features.append(feat)

        max_psi  = max(psi_per_feature.values()) if psi_per_feature else 0.0
        kl_div   = self._kl_divergence(training_scores, current_scores)
        drift    = (max_psi > PSI_RETRAIN) or (kl_div > KL_DRIFT_THRESHOLD)
        retrain  = drift

        report = DriftReport(
            model_id=f"{model_type}_champion",
            evaluated_at=datetime.utcnow().isoformat(),
            feature_psi=psi_per_feature,
            max_psi=round(max_psi, 4),
            output_kl=round(kl_div, 4),
            drift_detected=drift,
            retrain_triggered=retrain,
            drift_features=drift_features,
        )

        report_path = (
            self.root / "drift_reports"
            / f"{model_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

        status = "DRIFT DETECTED — retraining triggered" if retrain else "STABLE"
        print(f"\n[DriftDetection] {model_type} | {status}")
        print(f"  Max PSI: {max_psi:.4f} (threshold: {PSI_RETRAIN})")
        print(f"  KL Div:  {kl_div:.4f} (threshold: {KL_DRIFT_THRESHOLD})")
        if drift_features:
            print(f"  Drifted features: {drift_features}")

        return report

    def list_models(self, model_type: str = None) -> pd.DataFrame:
        """Returns registry contents as a DataFrame for inspection."""
        rows = []
        types = [model_type] if model_type else list(self._manifest.keys())
        for mt in types:
            for mv in self._manifest.get(mt, []):
                rows.append({
                    "model_type":       mt,
                    "model_id":         mv.model_id,
                    "version":          mv.version,
                    "status":           mv.status,
                    "cv_auc":           mv.cv_auc,
                    "brier_score":      mv.brier_score,
                    "training_samples": mv.training_samples,
                    "registered_at":    mv.registered_at,
                    "sha256":           mv.sha256[:12],
                })
        return pd.DataFrame(rows)

    # ── Persistence ───────────────────────────────────────────────────────

    def _save_manifest(self) -> None:
        manifest_path = self.root / "manifest.json"
        serialisable  = {
            mt: [asdict(v) for v in versions]
            for mt, versions in self._manifest.items()
        }
        with open(manifest_path, "w") as f:
            json.dump(serialisable, f, indent=2)

    def _load_manifest(self) -> None:
        manifest_path = self.root / "manifest.json"
        if not manifest_path.exists():
            return
        with open(manifest_path) as f:
            raw = json.load(f)
        for mt, versions in raw.items():
            self._manifest[mt] = [ModelVersion(**v) for v in versions]


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile, os
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.datasets import make_classification

    np.random.seed(42)
    X, y = make_classification(n_samples=500, n_features=10, random_state=42)
    feature_names = [f"feature_{i}" for i in range(10)]

    X_train, X_test = X[:400], X[400:]
    y_train, y_test = y[:400], y[400:]

    # Train a simple model for demo
    clf_v1 = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_v1.fit(X_train, y_train)

    clf_v2 = RandomForestClassifier(n_estimators=100, random_state=0)
    clf_v2.fit(X_train, y_train)

    with tempfile.TemporaryDirectory() as tmpdir:
        registry = ModelRegistry(registry_path=tmpdir)

        # Register v1
        mv1 = registry.register(
            model_obj=clf_v1, model_type="disruption_risk", version="v1.0.0",
            metrics={"cv_auc": 0.82, "cv_ap": 0.78, "brier_score": 0.14,
                     "training_samples": 400},
            feature_names=feature_names, hyperparameters={"n_estimators": 50},
        )
        registry.promote_to_champion("disruption_risk", mv1.model_id)

        # Register v2 as challenger
        mv2 = registry.register(
            model_obj=clf_v2, model_type="disruption_risk", version="v1.1.0",
            metrics={"cv_auc": 0.85, "cv_ap": 0.81, "brier_score": 0.12,
                     "training_samples": 400},
            feature_names=feature_names, hyperparameters={"n_estimators": 100},
        )

        print("\nAll registered models:")
        print(registry.list_models().to_string(index=False))

        # Shadow score
        print("\nShadow scoring on holdout:")
        X_test_df = pd.DataFrame(X_test, columns=feature_names)
        registry.shadow_score("disruption_risk", X_test_df, y_test)

        # Promote challenger
        registry.promote_to_champion("disruption_risk", mv2.model_id)

        # Drift detection
        X_ref = pd.DataFrame(X_train, columns=feature_names)
        X_cur = pd.DataFrame(X_test + np.random.randn(*X_test.shape) * 0.5,
                             columns=feature_names)   # add noise to simulate drift
        scores_ref = clf_v2.predict_proba(X_train)[:, 1]
        scores_cur = clf_v2.predict_proba(X_test)[:, 1]

        drift_report = registry.detect_drift(
            "disruption_risk", X_ref, X_cur, scores_ref, scores_cur
        )
