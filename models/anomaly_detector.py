"""
Anomaly Detector — Isolation Forest + ECOD Ensemble
=====================================================
Unsupervised anomaly detection over supply chain time-series signals.
Designed to catch novel disruption patterns not captured by supervised
XGBoost model (distribution shift, black swan events).

Models:
  1. Isolation Forest (Liu et al., 2008) — tree-based path-length scoring
  2. ECOD — Empirical Cumulative Distribution-based OD (Li & Li, 2022)
     Uses multivariate empirical CDF tails: no parametric assumptions
  3. Ensemble: average of normalised anomaly scores (rank aggregation)

Inputs: supplier feature vectors + rolling time-series aggregations
Output: per-sample anomaly score ∈ [0, 1] + binary flag + explanations

References:
  Liu, F.T., Ting, K.M., & Zhou, Z.H. (2008). Isolation Forest.
    IEEE ICDM.
  Li, Z., Zhao, Y., et al. (2022). ECOD: Unsupervised Outlier Detection
    Using Empirical Cumulative Distribution Functions.
    IEEE TKDE.

Author: Satya Sai Prakash Kantamani
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from scipy import stats
from dataclasses import dataclass, field
import warnings

warnings.filterwarnings("ignore")


# ── Features monitored for anomalies ─────────────────────────────────────────
ANOMALY_FEATURES = [
    "on_time_rate_30d",
    "on_time_rate_90d",
    "avg_delay_days_30d",
    "avg_delay_days_90d",
    "delay_std_90d",
    "disruption_rate_90d",
    "lead_time_variance_ratio",
    "geo_risk_score",
    "financial_risk_score",
    "order_volume_30d",
    "port_congestion_score",
    "commodity_price_volatility_30d",
    "js_divergence_delay_dist",
    "network_pagerank",
]


@dataclass
class AnomalyResult:
    supplier_id: str
    anomaly_score: float            # ensemble score ∈ [0, 1]; higher = more anomalous
    iso_score: float                # Isolation Forest contribution
    ecod_score: float               # ECOD contribution
    is_anomaly: bool                # binary flag (score > threshold)
    threshold: float
    top_anomalous_features: list    # features driving the anomaly
    z_scores: dict = field(default_factory=dict)    # per-feature z-scores
    percentiles: dict = field(default_factory=dict) # per-feature empirical percentiles


class ECOD:
    """
    Empirical Cumulative Distribution-based Outlier Detection.

    For each feature dimension, computes the left-tail and right-tail
    empirical CDF probabilities. An observation is anomalous if it falls
    in an extreme tail of multiple feature dimensions simultaneously.

    Score: -log P(X) where P(X) is the product of univariate tail probabilities
    (independence assumption — justified for ranked ensemble input).
    """

    def __init__(self, contamination: float = 0.1):
        self.contamination = contamination
        self._ecdf_stats: dict = {}    # per-feature (sorted_vals, n)
        self.threshold_: float = 0.0

    def fit(self, X: np.ndarray) -> "ECOD":
        """Store empirical distribution for each feature dimension."""
        n, p = X.shape
        for j in range(p):
            col = X[:, j]
            self._ecdf_stats[j] = {
                "sorted": np.sort(col),
                "n": n,
                "mean": col.mean(),
                "std": col.std() + 1e-9,
            }
        scores = self._score_raw(X)
        self.threshold_ = np.percentile(scores, 100 * (1 - self.contamination))
        return self

    def _ecdf_value(self, j: int, x: float) -> float:
        """F(x) = P(X_j <= x) via sorted array binary search."""
        arr = self._ecdf_stats[j]["sorted"]
        n   = self._ecdf_stats[j]["n"]
        idx = np.searchsorted(arr, x, side="right")
        return idx / n

    def _score_raw(self, X: np.ndarray) -> np.ndarray:
        """
        Compute ECOD scores: O(x) = -log(min(F(x), 1-F(x))) summed over dims.
        Measures extremity in either tail per dimension.
        """
        n, p = X.shape
        scores = np.zeros(n)
        for i in range(n):
            s = 0.0
            for j in range(p):
                f = self._ecdf_value(j, X[i, j])
                tail = min(f, 1.0 - f) + 1e-9
                s += -np.log(tail)
            scores[i] = s
        return scores

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Returns normalised ECOD scores ∈ [0, 1] (higher = more anomalous)."""
        raw = self._score_raw(X)
        # Normalise using min-max of training distribution
        return np.clip((raw - raw.min()) / (raw.max() - raw.min() + 1e-9), 0, 1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns 1 for anomaly, 0 for normal."""
        raw = self._score_raw(X)
        return (raw >= self.threshold_).astype(int)


class SupplyChainAnomalyDetector:
    """
    Ensemble anomaly detector combining Isolation Forest and ECOD.

    Training: fit both models on 90-day historical baseline.
    Inference: ensemble score = 0.5 * iso_norm + 0.5 * ecod_norm.
    Explanation: top-k features by absolute z-score.
    """

    def __init__(
        self,
        contamination:  float = 0.05,   # expected anomaly rate in training data
        iso_n_estimators: int = 200,
        iso_max_features: float = 0.8,
        iso_max_samples: str | int = "auto",
        ensemble_threshold: float = 0.65,
        random_seed: int = 42,
    ):
        self.contamination       = contamination
        self.ensemble_threshold  = ensemble_threshold
        self.random_seed         = random_seed

        self.scaler = RobustScaler()   # robust to outliers during training

        self.iso = IsolationForest(
            n_estimators=iso_n_estimators,
            max_features=iso_max_features,
            max_samples=iso_max_samples,
            contamination=contamination,
            random_state=random_seed,
            n_jobs=-1,
        )
        self.ecod = ECOD(contamination=contamination)

        self._train_stats: dict = {}   # per-feature mean/std for z-score computation
        self.features = ANOMALY_FEATURES
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "SupplyChainAnomalyDetector":
        """
        Fit both models on baseline data.
        df must contain all ANOMALY_FEATURES columns.
        """
        X_raw = df[self.features].fillna(0).astype(float).values
        X     = self.scaler.fit_transform(X_raw)

        # Store training statistics for z-score computation
        for i, feat in enumerate(self.features):
            col = X_raw[:, i]
            self._train_stats[feat] = {"mean": col.mean(), "std": col.std() + 1e-9}

        print(f"[AnomalyDetector] Fitting on {len(df):,} samples | "
              f"contamination={self.contamination:.0%}")
        self.iso.fit(X)
        self.ecod.fit(X)
        self.is_fitted = True
        return self

    def _iso_score_normalised(self, X: np.ndarray) -> np.ndarray:
        """
        Isolation Forest returns anomaly scores in (-∞, 0] range.
        Normalise to [0, 1]: higher = more anomalous.
        """
        raw = self.iso.score_samples(X)    # lower (more negative) = more anomalous
        return np.clip((-raw - 0.0) / 0.5, 0, 1)

    def score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute ensemble anomaly scores for all rows in df.
        Returns df with appended score columns.
        """
        assert self.is_fitted, "Call fit() before score()"
        X_raw = df[self.features].fillna(0).astype(float).values
        X     = self.scaler.transform(X_raw)

        iso_scores  = self._iso_score_normalised(X)
        ecod_scores = self.ecod.score_samples(X)
        ensemble    = 0.5 * iso_scores + 0.5 * ecod_scores

        out = df.copy()
        out["anomaly_score_iso"]      = iso_scores
        out["anomaly_score_ecod"]     = ecod_scores
        out["anomaly_score_ensemble"] = ensemble
        out["is_anomaly"]             = (ensemble >= self.ensemble_threshold).astype(int)
        return out

    def explain(self, row: pd.Series) -> AnomalyResult:
        """
        Generates per-supplier anomaly explanation for AIP narrative prompt.
        Returns AnomalyResult with top anomalous features and z-scores.
        """
        assert self.is_fitted
        X_raw = row[self.features].fillna(0).astype(float).values.reshape(1, -1)
        X     = self.scaler.transform(X_raw)

        iso_s  = float(self._iso_score_normalised(X)[0])
        ecod_s = float(self.ecod.score_samples(X)[0])
        ens    = 0.5 * iso_s + 0.5 * ecod_s

        # Z-scores relative to training distribution
        z_scores    = {}
        percentiles = {}
        for i, feat in enumerate(self.features):
            raw_val = X_raw[0, i]
            stats_  = self._train_stats[feat]
            z       = (raw_val - stats_["mean"]) / stats_["std"]
            z_scores[feat] = round(float(z), 3)
            # Empirical percentile approximation (normal assumption)
            percentiles[feat] = round(float(stats.norm.cdf(z) * 100), 1)

        # Top anomalous features: highest absolute z-score
        top_feats = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]

        return AnomalyResult(
            supplier_id=str(row.get("supplier_id", "UNKNOWN")),
            anomaly_score=round(ens, 4),
            iso_score=round(iso_s, 4),
            ecod_score=round(ecod_s, 4),
            is_anomaly=ens >= self.ensemble_threshold,
            threshold=self.ensemble_threshold,
            top_anomalous_features=[f for f, _ in top_feats],
            z_scores=z_scores,
            percentiles=percentiles,
        )

    def evaluate(
        self,
        df: pd.DataFrame,
        label_col: str = "disruption_label",
    ) -> dict:
        """
        Evaluate anomaly detector against ground truth disruption labels.
        Note: unsupervised anomaly detection is not a perfect proxy for
        disruption labels — this is diagnostic, not a training signal.
        """
        scored = self.score(df)
        y_true = df[label_col].values
        y_score = scored["anomaly_score_ensemble"].values

        metrics = {
            "roc_auc":           round(roc_auc_score(y_true, y_score), 4),
            "anomaly_rate":      round(scored["is_anomaly"].mean(), 4),
            "disruption_rate":   round(y_true.mean(), 4),
            "threshold":         self.ensemble_threshold,
            "n_samples":         len(df),
        }
        print(f"\n[AnomalyDetector] Evaluation Results:")
        for k, v in metrics.items():
            print(f"  {k:<25}: {v}")
        return metrics

    def get_anomaly_summary(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Returns scored DataFrame filtered to anomalies, sorted by score desc.
        """
        scored = self.score(df)
        anomalies = (
            scored
            .query("is_anomaly == 1")
            .sort_values("anomaly_score_ensemble", ascending=False)
            [["supplier_id", "anomaly_score_ensemble", "anomaly_score_iso",
              "anomaly_score_ecod", "is_anomaly"]]
        )
        return anomalies


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    np.random.seed(42)
    N_TRAIN, N_TEST = 2000, 200

    # Simulate baseline supplier feature distribution
    def _make_features(n, anomaly_frac=0.0):
        df = pd.DataFrame({
            feat: np.random.rand(n) for feat in ANOMALY_FEATURES
        })
        df["supplier_id"]       = [f"SUP-{i:04d}" for i in range(n)]
        df["disruption_label"]  = 0

        # Inject anomalies: extreme values in 3+ features simultaneously
        n_anom = int(n * anomaly_frac)
        if n_anom > 0:
            idx = np.random.choice(n, n_anom, replace=False)
            df.loc[idx, "disruption_rate_90d"]    = np.random.uniform(0.6, 1.0, n_anom)
            df.loc[idx, "avg_delay_days_30d"]      = np.random.uniform(15, 30, n_anom)
            df.loc[idx, "on_time_rate_30d"]        = np.random.uniform(0.0, 0.3, n_anom)
            df.loc[idx, "disruption_label"]        = 1
        return df

    train_df = _make_features(N_TRAIN, anomaly_frac=0.0)   # clean baseline
    test_df  = _make_features(N_TEST,  anomaly_frac=0.15)  # 15% anomalies

    print("=" * 55)
    print("  Supply Chain Anomaly Detector — Demo")
    print("=" * 55)

    detector = SupplyChainAnomalyDetector(contamination=0.05, ensemble_threshold=0.60)
    detector.fit(train_df)

    metrics = detector.evaluate(test_df, label_col="disruption_label")

    summary = detector.get_anomaly_summary(test_df)
    print(f"\nDetected {len(summary)} anomalies in {N_TEST} test suppliers")
    print("\nTop 5 Anomalies:")
    print(summary.head(5).to_string(index=False))

    # Explain worst anomaly
    worst = test_df.loc[test_df["supplier_id"].isin(summary.head(1)["supplier_id"])].iloc[0]
    result = detector.explain(worst)
    print(f"\nAnomaly Explanation — {result.supplier_id}:")
    print(f"  Ensemble score:        {result.anomaly_score:.4f}")
    print(f"  Isolation Forest:      {result.iso_score:.4f}")
    print(f"  ECOD:                  {result.ecod_score:.4f}")
    print(f"  Top anomalous features: {result.top_anomalous_features[:3]}")
    print(f"  Z-scores (top 3):")
    for feat in result.top_anomalous_features[:3]:
        print(f"    {feat:<40}: z={result.z_scores[feat]:+.2f} "
              f"({result.percentiles[feat]}th pct)")
