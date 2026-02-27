"""
Supplier Disruption Risk Scorer
================================
XGBoost binary classifier predicting the probability that a supplier
will cause a disruption event within the next 30 days.

Features are sourced from the Foundry Gold layer (supply_chain_risk_master).
This module is portable — runs standalone or as a Foundry Code Workbook.

Author: Satya Sai Prakash Kantamani
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")


# ── Feature schema ──────────────────────────────────────────────────────────
FEATURES = [
    "rolling_on_time_rate",       # 90-day on-time delivery rate
    "rolling_avg_delay_days",     # avg delay in days (90d window)
    "disruption_rate",            # % shipments delayed/lost (90d)
    "delay_volatility",           # std dev of delay days
    "geo_risk_score",             # country-level geopolitical risk (0-1)
    "political_stability_index",  # World Bank political stability
    "financial_risk_score",       # supplier financial health (0-1, inverted)
    "lead_time_variance",         # delay_volatility / avg_delay
    "supplier_concentration_risk",# spend share normalised
    "order_volume_90d",           # order count (90d)
]
TARGET = "disruption_label"       # 1 = disruption occurred within 30 days


class DisruptionRiskModel:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            scale_pos_weight=4,        # handles class imbalance (~20% positive)
            eval_metric="auc",
            random_state=42,
            use_label_encoder=False,
        )
        self.feature_names = FEATURES
        self.threshold = 0.45          # tuned for precision/recall balance

    def train(self, df: pd.DataFrame) -> dict:
        X = df[FEATURES].fillna(0)
        y = df[TARGET]

        # ── Stratified 5-fold CV ─────────────────────────────────────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring="roc_auc")

        # ── Full fit ─────────────────────────────────────────────────────────
        self.model.fit(
            X, y,
            eval_set=[(X, y)],
            verbose=False,
        )

        metrics = {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std":  cv_scores.std(),
            "n_train":     len(df),
            "positive_rate": y.mean(),
        }
        print(f"[DisruptionRiskModel] CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        return metrics

    def predict_proba(self, df: pd.DataFrame) -> pd.Series:
        X = df[FEATURES].fillna(0)
        proba = self.model.predict_proba(X)[:, 1]
        return pd.Series(proba, index=df.index, name="disruption_probability")

    def score_suppliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return enriched DataFrame with risk scores and tiers."""
        df = df.copy()
        df["disruption_probability"] = self.predict_proba(df)
        df["risk_tier"] = pd.cut(
            df["disruption_probability"],
            bins=[0, 0.25, 0.50, 0.75, 1.0],
            labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"],
        )
        df["rank"] = df["disruption_probability"].rank(ascending=False).astype(int)
        return df.sort_values("disruption_probability", ascending=False)

    def feature_importance(self) -> pd.DataFrame:
        imp = self.model.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": imp,
        }).sort_values("importance", ascending=False)

    def evaluate(self, df: pd.DataFrame) -> None:
        X = df[FEATURES].fillna(0)
        y = df[TARGET]
        y_prob = self.predict_proba(df)
        y_pred = (y_prob >= self.threshold).astype(int)

        print("\n── Disruption Risk Model Evaluation ──")
        print(f"ROC-AUC:  {roc_auc_score(y, y_prob):.4f}")
        print("\nClassification Report:")
        print(classification_report(y, y_pred, target_names=["No Disruption", "Disruption"]))
        print("\nTop Feature Importances:")
        print(self.feature_importance().head(6).to_string(index=False))


# ── Standalone run ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Simulate Foundry Gold layer output for local testing
    np.random.seed(42)
    n = 2000

    df_sim = pd.DataFrame({
        "supplier_id":                np.arange(n),
        "rolling_on_time_rate":       np.random.beta(8, 2, n),
        "rolling_avg_delay_days":     np.random.exponential(2, n),
        "disruption_rate":            np.random.beta(2, 8, n),
        "delay_volatility":           np.random.exponential(1.5, n),
        "geo_risk_score":             np.random.uniform(0, 1, n),
        "political_stability_index":  np.random.uniform(0, 1, n),
        "financial_risk_score":       np.random.beta(3, 7, n),
        "lead_time_variance":         np.random.exponential(0.5, n),
        "supplier_concentration_risk":np.random.uniform(0, 1, n),
        "order_volume_90d":           np.random.poisson(50, n).astype(float),
    })

    # Synthetic label: high risk = more likely disruption
    risk_signal = (
        0.4 * df_sim["geo_risk_score"] +
        0.3 * df_sim["disruption_rate"] +
        0.3 * (1 - df_sim["rolling_on_time_rate"])
    )
    df_sim[TARGET] = (risk_signal + np.random.normal(0, 0.1, n) > 0.45).astype(int)

    print(f"Dataset: {n} suppliers | {df_sim[TARGET].mean():.1%} positive rate\n")

    model = DisruptionRiskModel()
    model.train(df_sim)
    model.evaluate(df_sim)

    scored = model.score_suppliers(df_sim).head(10)
    print("\nTop 10 Highest Risk Suppliers:")
    print(scored[["supplier_id", "disruption_probability", "risk_tier", "rank"]].to_string(index=False))
