"""
Supplier Disruption Risk Scorer
================================
XGBoost binary classifier with Bayesian hyperparameter optimisation (Optuna),
Platt scaling calibration, and SHAP TreeExplainer for per-prediction attribution.

Predicts P(disruption | supplier, t+30 days) using temporal rolling features,
geopolitical signals, network centrality, and financial distress proxies.

Reference: Chen & Guestrin (2016) XGBoost; Lundberg & Lee (2017) SHAP.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
import warnings
from dataclasses import dataclass, field
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    classification_report, brier_score_loss, make_scorer,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ── Feature schema ─────────────────────────────────────────────────────────
FEATURES = [
    # Temporal delivery performance
    "on_time_rate_7d", "on_time_rate_30d", "on_time_rate_90d",
    "on_time_rate_ewm_alpha02",          # exponentially weighted (α=0.2)
    "avg_delay_days_30d", "avg_delay_days_90d",
    "delay_std_90d",
    "js_divergence_delay_dist",          # Jensen-Shannon divergence vs 1yr baseline

    # Disruption history
    "disruption_rate_90d",
    "days_since_last_disruption",
    "consecutive_on_time_streak",

    # Geopolitical & macro
    "geo_risk_score",                    # composite index (0–1)
    "political_stability_wgi",           # World Bank WGI indicator
    "trade_conflict_intensity",          # bilateral trade dispute score
    "commodity_price_volatility_30d",    # relevant commodity index volatility

    # Financial health
    "financial_risk_score",              # Altman Z-score proxy (inverted, 0–1)
    "payment_delay_rate_90d",            # % invoices paid late

    # Network (from GraphSAGE)
    "network_pagerank",                  # supplier importance in dependency graph
    "second_order_contagion_score",      # P(disruption | connected supplier disrupted)
    "supply_concentration_hhi",          # Herfindahl-Hirschman spend concentration

    # Lead time
    "lead_time_variance_ratio",          # std/mean lead time
    "lead_time_trend_slope",             # OLS slope of lead time over 90d

    # Logistics
    "port_congestion_score",             # origin port congestion index
    "transit_route_risk",                # historical disruption rate on primary route

    # Volume
    "order_volume_30d", "order_volume_90d",
    "spend_share_pct",                   # % of total procurement
]
TARGET = "disruption_label"


@dataclass
class ModelMetrics:
    cv_auc: float = 0.0
    cv_auc_std: float = 0.0
    cv_ap: float = 0.0
    brier_score: float = 0.0
    best_params: dict = field(default_factory=dict)


class DisruptionRiskModel:
    """
    XGBoost disruption risk scorer with:
    - Bayesian HPO via Optuna (TPE sampler, 200 trials)
    - Platt scaling for probability calibration
    - SHAP TreeExplainer for feature attribution
    - Stratified 5-fold cross-validation
    """

    def __init__(self, n_optuna_trials: int = 200, calibrate: bool = True):
        self.n_optuna_trials = n_optuna_trials
        self.calibrate = calibrate
        self.model = None
        self.calibrated_model = None   # LogisticRegression Platt scaler
        self.explainer = None
        self.metrics = ModelMetrics()
        self.threshold = 0.45

    # ── Bayesian HPO ──────────────────────────────────────────────────────
    def _optuna_objective(self, trial, X, y):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 100, 600),
            "max_depth":          trial.suggest_int("max_depth", 3, 8),
            "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":          trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda":         trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "scale_pos_weight":   trial.suggest_float("scale_pos_weight", 2.0, 8.0),
            "random_state": 42,
        }
        clf = xgb.XGBClassifier(**params)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = []
        for ti, vi in cv.split(X, y):
            clf.fit(X.iloc[ti], y.iloc[ti])
            p = clf.predict_proba(X.iloc[vi])[:, 1]
            scores.append(roc_auc_score(y.iloc[vi], p))
        return float(np.mean(scores))

    def train(self, df: pd.DataFrame) -> ModelMetrics:
        X = df[FEATURES].fillna(0).astype(float)
        y = df[TARGET]

        print(f"[DisruptionRiskModel] Training on {len(df):,} samples | "
              f"positive rate: {y.mean():.2%}")

        # ── Bayesian HPO ──────────────────────────────────────────────────
        print(f"[DisruptionRiskModel] Running Optuna HPO ({self.n_optuna_trials} trials)...")
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(
            lambda trial: self._optuna_objective(trial, X, y),
            n_trials=self.n_optuna_trials,
            show_progress_bar=False,
        )
        best_params = study.best_params
        best_params.update({"random_state": 42})
        self.metrics.best_params = best_params
        print(f"[DisruptionRiskModel] Best AUC: {study.best_value:.4f} | "
              f"Params: {best_params}")

        # ── Full fit ──────────────────────────────────────────────────────
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(X, y, verbose=False)

        # ── Platt calibration (manual — sklearn 1.7 compatible) ───────────
        if self.calibrate:
            # Get out-of-fold raw probabilities, fit logistic scaler on top
            cv_c = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            oof = np.zeros(len(X))
            for ti, vi in cv_c.split(X, y):
                tmp = xgb.XGBClassifier(**best_params)
                tmp.fit(X.iloc[ti], y.iloc[ti])
                oof[vi] = tmp.predict_proba(X.iloc[vi])[:, 1]
            self.calibrated_model = LogisticRegression()
            self.calibrated_model.fit(oof.reshape(-1, 1), y)

        # ── SHAP explainer ────────────────────────────────────────────────
        self.explainer = shap.TreeExplainer(self.model)

        # ── Final CV metrics ──────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        auc_scores, ap_scores = [], []
        for ti, vi in cv.split(X, y):
            self.model.fit(X.iloc[ti], y.iloc[ti])
            p = self.model.predict_proba(X.iloc[vi])[:, 1]
            auc_scores.append(roc_auc_score(y.iloc[vi], p))
            ap_scores.append(average_precision_score(y.iloc[vi], p))
        # Refit on full data after CV
        self.model.fit(X, y, verbose=False)

        self.metrics.cv_auc     = float(np.mean(auc_scores))
        self.metrics.cv_auc_std = float(np.std(auc_scores))
        self.metrics.cv_ap      = float(np.mean(ap_scores))

        y_prob = self.predict_proba(df)
        self.metrics.brier_score = brier_score_loss(y, y_prob)

        print(f"\n[Results] CV AUC: {self.metrics.cv_auc:.4f} ± {self.metrics.cv_auc_std:.4f}"
              f" | CV AP: {self.metrics.cv_ap:.4f}"
              f" | Brier: {self.metrics.brier_score:.4f}")
        return self.metrics

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = df[FEATURES].fillna(0).astype(float)
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrate and self.calibrated_model is not None:
            return self.calibrated_model.predict_proba(raw.reshape(-1, 1))[:, 1]
        return raw

    def score_suppliers(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["disruption_probability"] = self.predict_proba(df)
        out["risk_tier"] = pd.cut(
            out["disruption_probability"],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
        out["rank"] = out["disruption_probability"].rank(ascending=False).astype(int)
        return out.sort_values("disruption_probability", ascending=False)

    # ── SHAP explainability ───────────────────────────────────────────────
    def explain(self, df: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        """
        Returns per-row SHAP attribution for top-k features.
        Payload is sent to AIP agent for natural language narrative generation.
        """
        X = df[FEATURES].fillna(0).astype(float)
        shap_values = self.explainer.shap_values(X)

        records = []
        for i in range(len(df)):
            sv = shap_values[i]
            top_idx = np.argsort(np.abs(sv))[-top_k:][::-1]
            for j in top_idx:
                records.append({
                    "row_idx":    i,
                    "supplier_id": df.iloc[i].get("supplier_id", i),
                    "feature":    FEATURES[j],
                    "shap_value": sv[j],
                    "direction":  "↑ risk" if sv[j] > 0 else "↓ risk",
                })
        return pd.DataFrame(records)

    # ── Evaluation ────────────────────────────────────────────────────────
    def evaluate(self, df: pd.DataFrame) -> None:
        X = df[FEATURES].fillna(0).astype(float)
        y = df[TARGET]
        y_prob = self.predict_proba(df)
        y_pred = (y_prob >= self.threshold).astype(int)

        print("\n══════════════════════════════════════════")
        print("  Disruption Risk Model — Evaluation")
        print("══════════════════════════════════════════")
        print(f"  ROC-AUC:              {roc_auc_score(y, y_prob):.4f}")
        print(f"  Average Precision:    {average_precision_score(y, y_prob):.4f}")
        print(f"  Brier Score:          {brier_score_loss(y, y_prob):.4f}")
        print(f"  Decision Threshold:   {self.threshold}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred,
                                    target_names=["No Disruption", "Disruption"]))
        print("\nTop 10 Feature Importances (SHAP mean |value|):")
        sv = self.explainer.shap_values(df[FEATURES].fillna(0).astype(float))
        fi = pd.DataFrame({
            "feature":          FEATURES,
            "mean_abs_shap":    np.abs(sv).mean(axis=0),
        }).sort_values("mean_abs_shap", ascending=False)
        print(fi.head(10).to_string(index=False))


# ── Standalone demo ────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N = 3000
    df = pd.DataFrame({f: np.random.rand(N) for f in FEATURES})

    # Correlated features
    df["on_time_rate_90d"]         = np.random.beta(8, 2, N)
    df["on_time_rate_30d"]         = df["on_time_rate_90d"] * np.random.uniform(0.9, 1.1, N)
    df["on_time_rate_7d"]          = df["on_time_rate_30d"] * np.random.uniform(0.85, 1.15, N)
    df["disruption_rate_90d"]      = np.random.beta(2, 10, N)
    df["geo_risk_score"]           = np.random.uniform(0, 1, N)
    df["financial_risk_score"]     = np.random.beta(3, 7, N)
    df["network_pagerank"]         = np.random.exponential(0.1, N)

    risk_signal = (
        0.30 * df["geo_risk_score"]
        + 0.25 * df["disruption_rate_90d"]
        + 0.25 * (1 - df["on_time_rate_90d"])
        + 0.10 * df["financial_risk_score"]
        + 0.10 * df["network_pagerank"].clip(0, 1)
    )
    df[TARGET] = (risk_signal + np.random.normal(0, 0.08, N) > 0.40).astype(int)
    df["supplier_id"] = [f"SUP-{i:04d}" for i in range(N)]

    print(f"Simulated dataset: {N} suppliers | positive rate: {df[TARGET].mean():.2%}\n")

    model = DisruptionRiskModel(n_optuna_trials=30)  # reduce for demo
    model.train(df)
    model.evaluate(df)

    top5 = model.score_suppliers(df).head(5)
    print("\nTop 5 Highest-Risk Suppliers:")
    print(top5[["supplier_id", "disruption_probability", "risk_tier"]].to_string(index=False))

    print("\nSHAP Attribution — Top Supplier:")
    shap_df = model.explain(top5.head(1))
    print(shap_df[["feature", "shap_value", "direction"]].to_string(index=False))
