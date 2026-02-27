"""
Monte Carlo Value-at-Risk Simulator
=====================================
Quantifies financial exposure under correlated supply chain disruption
scenarios using Gaussian copula for dependency modelling and network
contagion propagation.

Computes:
  - VaR(α): revenue at risk at confidence level α
  - CVaR(α): Expected Shortfall — mean loss in worst (1-α) scenarios
  - Scenario attribution: which supplier-risk combinations drive tail loss
  - Mitigation ROI: cost of interventions vs expected loss reduction

Reference:
  Embrechts, P., McNeil, A., & Straumann, D. (2002). Correlation and
  Dependence in Risk Management: Properties and Pitfalls.
  (Gaussian copula for joint disruption modelling)
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.linalg import cholesky
from dataclasses import dataclass
import warnings

warnings.filterwarnings("ignore")


@dataclass
class SimulationResult:
    var_95:           float    # Value-at-Risk at 95% confidence
    var_99:           float    # Value-at-Risk at 99% confidence
    cvar_95:          float    # Conditional VaR (Expected Shortfall) at 95%
    cvar_99:          float    # Conditional VaR at 99% confidence
    mean_loss:        float    # Expected loss across all scenarios
    std_loss:         float    # Loss standard deviation
    prob_any_disruption: float # P(at least one supplier disrupts)
    top_contributors: pd.DataFrame  # Scenario attribution
    n_simulations:    int


class GaussianCopula:
    """
    Gaussian copula for modelling correlated disruption probabilities.
    Preserves marginal distributions while capturing cross-supplier dependency.
    """
    def __init__(self, correlation_matrix: np.ndarray):
        self._validate_correlation(correlation_matrix)
        self.corr = correlation_matrix
        self.L    = cholesky(correlation_matrix, lower=True)

    @staticmethod
    def _validate_correlation(C: np.ndarray):
        n = C.shape[0]
        assert C.shape == (n, n), "Correlation matrix must be square"
        assert np.allclose(C, C.T, atol=1e-8), "Must be symmetric"
        assert np.all(np.linalg.eigvals(C) > -1e-8), "Must be positive semi-definite"
        np.fill_diagonal(C, 1.0)

    def sample(self, n_samples: int, marginal_probs: np.ndarray) -> np.ndarray:
        """
        Sample correlated Bernoulli disruption events.

        Steps:
        1. Draw Z ~ N(0, I_n)
        2. Correlate: Y = L @ Z  →  Y ~ N(0, Sigma)
        3. Transform to U[0,1] via standard normal CDF
        4. Apply Bernoulli threshold at marginal disruption probabilities

        Returns: (n_samples, n_suppliers) binary disruption matrix
        """
        n_suppliers = len(marginal_probs)
        Z = np.random.standard_normal((n_suppliers, n_samples))   # (n_sup, n_sim)
        Y = self.L @ Z                                             # correlated normals
        U = stats.norm.cdf(Y).T                                    # (n_sim, n_sup) uniform
        thresholds = marginal_probs[np.newaxis, :]                 # (1, n_sup)
        return (U < thresholds).astype(float)                      # (n_sim, n_sup)


class MonteCarloVaR:
    """
    Monte Carlo supply chain VaR/CVaR simulation with:
    - Gaussian copula correlated disruption sampling
    - Network contagion propagation (second-order disruptions)
    - Inventory depletion modelling under demand uncertainty
    - Financial impact quantification (revenue at risk + mitigation cost)
    """

    def __init__(self, n_simulations: int = 50_000, random_seed: int = 42):
        self.n_sims = n_simulations
        np.random.seed(random_seed)

    def run(
        self,
        suppliers:            pd.DataFrame,
        inventory:            pd.DataFrame,
        contagion_matrix:     np.ndarray,
        correlation_matrix:   np.ndarray,
        demand_uncertainty:   float = 0.15,
        expedite_cost_factor: float = 2.5,
    ) -> SimulationResult:
        """
        Parameters
        ----------
        suppliers : DataFrame with columns:
            supplier_id, disruption_probability, annual_spend_usd,
            avg_lead_time_days, substitution_cost_factor
        inventory : DataFrame with columns:
            sku_id, supplier_id, days_of_supply, avg_daily_revenue,
            demand_p10, demand_p50, demand_p90
        contagion_matrix : (n_sup, n_sup) float — P(j disrupted | i disrupted)
            from GraphSAGE model
        correlation_matrix : (n_sup, n_sup) — cross-supplier disruption correlation
        demand_uncertainty : fractional demand forecast error (1 std dev)
        expedite_cost_factor : cost multiplier for emergency procurement
        """
        n_sup   = len(suppliers)
        n_sku   = len(inventory)
        probs   = suppliers["disruption_probability"].values
        copula  = GaussianCopula(correlation_matrix)

        # ── Step 1: Sample correlated primary disruptions ──────────────────
        primary = copula.sample(self.n_sims, probs)   # (n_sims, n_sup)

        # ── Step 2: Propagate contagion (second-order disruptions) ─────────
        # P(j disrupted in sim s) = max(primary[s,j], max_i(primary[s,i]*C[i,j]))
        contagion = primary @ contagion_matrix          # (n_sims, n_sup)
        disrupted = np.maximum(primary,
                               (contagion > 0.5).astype(float))   # (n_sims, n_sup)

        # ── Step 3: Map supplier disruptions to inventory SKUs ─────────────
        # Build supplier → SKU mask
        sup_ids  = suppliers["supplier_id"].values
        sup_idx  = {s: i for i, s in enumerate(sup_ids)}
        sku_sup  = inventory["supplier_id"].map(sup_idx).fillna(-1).astype(int).values

        # SKU affected if its supplier is disrupted: (n_sims, n_sku)
        sku_disrupted = np.zeros((self.n_sims, n_sku))
        for k, si in enumerate(sku_sup):
            if si >= 0:
                sku_disrupted[:, k] = disrupted[:, si]

        # ── Step 4: Stochastic demand draws ───────────────────────────────
        daily_rev  = inventory["avg_daily_revenue"].values          # (n_sku,)
        dos        = inventory["days_of_supply"].values             # (n_sku,)
        demand_mul = 1 + demand_uncertainty * np.random.randn(self.n_sims, n_sku)
        demand_mul = np.clip(demand_mul, 0.5, 2.0)

        # Lead time draw for each supplier: (n_sims, n_sup)
        lt_mean = suppliers["avg_lead_time_days"].values
        lt_std  = lt_mean * 0.2
        lead_time_draws = np.maximum(1,
            lt_mean + lt_std * np.random.randn(self.n_sims, n_sup))

        # Per-SKU disruption duration = supplier lead time draw
        disruption_days = np.zeros((self.n_sims, n_sku))
        for k, si in enumerate(sku_sup):
            if si >= 0:
                disruption_days[:, k] = lead_time_draws[:, si] * sku_disrupted[:, k]

        # ── Step 5: Stockout revenue at risk ──────────────────────────────
        # Stockout days = max(0, disruption_days - days_of_supply)
        stockout_days = np.maximum(0, disruption_days - dos[np.newaxis, :])
        revenue_at_risk = stockout_days * daily_rev[np.newaxis, :] * demand_mul

        # ── Step 6: Substitution & expedite costs ─────────────────────────
        sub_factor = suppliers["substitution_cost_factor"].values   # (n_sup,)
        sub_cost_per_sku = np.zeros((self.n_sims, n_sku))
        for k, si in enumerate(sku_sup):
            if si >= 0:
                sub_cost_per_sku[:, k] = (
                    sku_disrupted[:, k]
                    * daily_rev[k]
                    * lt_mean[si]
                    * (sub_factor[si] - 1)
                )

        total_loss = revenue_at_risk.sum(axis=1) + sub_cost_per_sku.sum(axis=1)

        # ── Step 7: VaR / CVaR ────────────────────────────────────────────
        var_95  = np.percentile(total_loss, 95)
        var_99  = np.percentile(total_loss, 99)
        cvar_95 = total_loss[total_loss >= var_95].mean()
        cvar_99 = total_loss[total_loss >= var_99].mean()

        # ── Step 8: Scenario attribution ──────────────────────────────────
        tail_mask   = total_loss >= var_95
        tail_disrupted = disrupted[tail_mask]                  # tail scenario disruptions
        contrib = tail_disrupted.mean(axis=0)                  # freq in tail scenarios
        top_contributors = pd.DataFrame({
            "supplier_id":     suppliers["supplier_id"].values,
            "tail_frequency":  contrib,
            "marginal_var":    contrib * var_95 / n_sup,       # simplified attribution
        }).sort_values("tail_frequency", ascending=False)

        return SimulationResult(
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            mean_loss=total_loss.mean(),
            std_loss=total_loss.std(),
            prob_any_disruption=(disrupted.sum(axis=1) > 0).mean(),
            top_contributors=top_contributors.head(10),
            n_simulations=self.n_sims,
        )

    def mitigation_roi(
        self,
        result: SimulationResult,
        intervention_cost: float,
        expected_loss_reduction: float,
    ) -> dict:
        """
        Returns ROI of a mitigation action (e.g., dual sourcing, buffer stock).

        expected_loss_reduction: fraction of CVaR(95) avoided by intervention
        """
        avoided_loss = result.cvar_95 * expected_loss_reduction
        roi = (avoided_loss - intervention_cost) / intervention_cost
        payback_months = intervention_cost / (avoided_loss / 12)
        return {
            "intervention_cost_usd":  intervention_cost,
            "avoided_loss_usd":       avoided_loss,
            "net_benefit_usd":        avoided_loss - intervention_cost,
            "roi_pct":                roi * 100,
            "payback_months":         payback_months,
        }

    def print_report(self, result: SimulationResult) -> None:
        print("\n" + "═" * 58)
        print("  SUPPLY CHAIN MONTE CARLO VaR REPORT")
        print("═" * 58)
        print(f"  Simulations:              {result.n_simulations:,}")
        print(f"  P(any disruption):        {result.prob_any_disruption:.1%}")
        print(f"  Expected Loss (mean):     ${result.mean_loss:>12,.0f}")
        print(f"  Loss Std Dev:             ${result.std_loss:>12,.0f}")
        print(f"  ─────────────────────────────────────────────────")
        print(f"  VaR(95%):                 ${result.var_95:>12,.0f}")
        print(f"  VaR(99%):                 ${result.var_99:>12,.0f}")
        print(f"  CVaR(95%) [Exp. Shortfall]: ${result.cvar_95:>10,.0f}")
        print(f"  CVaR(99%):                ${result.cvar_99:>12,.0f}")
        print(f"\n  Top Loss-Contributing Suppliers (tail freq.):")
        print(result.top_contributors[["supplier_id", "tail_frequency",
                                       "marginal_var"]].to_string(index=False))
        print("═" * 58)


# ── Standalone demo ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(42)
    N_SUP, N_SKU = 15, 30

    suppliers = pd.DataFrame({
        "supplier_id":              [f"SUP-{i:02d}" for i in range(N_SUP)],
        "disruption_probability":   np.random.uniform(0.05, 0.55, N_SUP),
        "annual_spend_usd":         np.random.uniform(500_000, 10_000_000, N_SUP),
        "avg_lead_time_days":       np.random.uniform(5, 45, N_SUP),
        "substitution_cost_factor": np.random.uniform(1.1, 3.0, N_SUP),
    })

    inventory = pd.DataFrame({
        "sku_id":              [f"SKU-{i:03d}" for i in range(N_SKU)],
        "supplier_id":         [f"SUP-{i % N_SUP:02d}" for i in range(N_SKU)],
        "days_of_supply":      np.random.uniform(3, 30, N_SKU),
        "avg_daily_revenue":   np.random.uniform(5_000, 100_000, N_SKU),
        "demand_p10":          np.random.uniform(80, 100, N_SKU),
        "demand_p50":          np.random.uniform(100, 120, N_SKU),
        "demand_p90":          np.random.uniform(120, 150, N_SKU),
    })

    # Construct plausible correlation matrix
    base_corr = np.eye(N_SUP) * 0.8 + np.ones((N_SUP, N_SUP)) * 0.2
    base_corr = (base_corr + base_corr.T) / 2
    np.fill_diagonal(base_corr, 1.0)

    # Contagion matrix from GraphSAGE output (simulated)
    contagion = np.random.uniform(0, 0.3, (N_SUP, N_SUP))
    np.fill_diagonal(contagion, 0)

    sim = MonteCarloVaR(n_simulations=20_000)
    result = sim.run(
        suppliers=suppliers,
        inventory=inventory,
        contagion_matrix=contagion,
        correlation_matrix=base_corr,
    )
    sim.print_report(result)

    roi = sim.mitigation_roi(
        result,
        intervention_cost=500_000,
        expected_loss_reduction=0.35,
    )
    print("\n  Dual-Sourcing Intervention ROI:")
    for k, v in roi.items():
        print(f"  {k:<30}: {v:,.2f}" if isinstance(v, float) else f"  {k}: {v}")
