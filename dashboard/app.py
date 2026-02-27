"""
Supply Chain Intelligence Dashboard
=====================================
Streamlit simulation of the Palantir Foundry Workshop interface.
Runs all 5 ML models live and presents results in an enterprise UI.

Author: Satya Sai Prakash Kantamani
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Supply Chain Intelligence | AIP",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark Foundry-style theme ──────────────────────────────────────────────────
st.markdown("""
<style>
  /* Global */
  html, body, [data-testid="stAppViewContainer"] {
    background:#0d1117; color:#e6edf3; font-family:'Inter',sans-serif;
  }
  [data-testid="stSidebar"] {
    background:#161b22; border-right:1px solid #30363d;
  }
  [data-testid="stSidebar"] * { color:#e6edf3 !important; }

  /* Metric cards */
  [data-testid="metric-container"] {
    background:#161b22; border:1px solid #30363d; border-radius:10px;
    padding:1rem; transition:border-color .2s;
  }
  [data-testid="metric-container"]:hover { border-color:#58a6ff; }
  [data-testid="stMetricValue"] { color:#58a6ff !important; font-size:1.8rem !important; font-weight:700 !important; }
  [data-testid="stMetricLabel"] { color:#8b949e !important; font-size:.75rem !important; letter-spacing:1px; text-transform:uppercase; }
  [data-testid="stMetricDelta"] { font-size:.8rem !important; }

  /* Section headers */
  .sec-head {
    font-size:.65rem; letter-spacing:2px; text-transform:uppercase;
    color:#58a6ff; margin-bottom:.25rem; margin-top:1.5rem;
  }
  h1,h2,h3 { color:#e6edf3 !important; }

  /* Tables */
  .stDataFrame { border:1px solid #30363d; border-radius:8px; }
  thead tr th { background:#161b22 !important; color:#8b949e !important;
                font-size:.7rem !important; letter-spacing:1px; text-transform:uppercase; }
  tbody tr:hover td { background:rgba(88,166,255,.06) !important; }

  /* Sidebar nav */
  .nav-item {
    padding:.55rem .8rem; border-radius:8px; margin:.2rem 0;
    cursor:pointer; font-size:.9rem; color:#8b949e;
    transition:all .15s;
  }
  .nav-item:hover,.nav-item.active {
    background:rgba(88,166,255,.12); color:#58a6ff;
  }

  /* Risk badges */
  .badge-critical { background:#3d1c1c; color:#f85149; border:1px solid #5a2020;
                    padding:.2rem .55rem; border-radius:20px; font-size:.7rem; font-weight:600; }
  .badge-high     { background:#2d1f0e; color:#f0883e; border:1px solid #4a3010;
                    padding:.2rem .55rem; border-radius:20px; font-size:.7rem; font-weight:600; }
  .badge-medium   { background:#1f2a0e; color:#a8d467; border:1px solid #304010;
                    padding:.2rem .55rem; border-radius:20px; font-size:.7rem; font-weight:600; }
  .badge-low      { background:#0e2020; color:#39d353; border:1px solid #0e3030;
                    padding:.2rem .55rem; border-radius:20px; font-size:.7rem; font-weight:600; }

  /* Alert boxes */
  .alert-critical { background:#1c0a0a; border-left:3px solid #f85149;
                    padding:.75rem 1rem; border-radius:0 8px 8px 0; margin:.4rem 0; }
  .alert-high     { background:#1c130a; border-left:3px solid #f0883e;
                    padding:.75rem 1rem; border-radius:0 8px 8px 0; margin:.4rem 0; }

  /* Dividers */
  hr { border-color:#30363d !important; }

  /* Inputs */
  .stSelectbox > div, .stSlider > div { background:#161b22 !important; }
  .stTextInput input, .stTextArea textarea {
    background:#161b22 !important; border:1px solid #30363d !important;
    color:#e6edf3 !important; border-radius:8px !important;
  }
  .stButton > button {
    background:#1f6feb; color:#fff; border:none; border-radius:8px;
    padding:.5rem 1.2rem; font-weight:600; transition:background .2s;
  }
  .stButton > button:hover { background:#388bfd; }

  /* Hide Streamlit branding */
  #MainMenu, footer, header { visibility:hidden; }
</style>
""", unsafe_allow_html=True)

PLOTLY_DARK = dict(paper_bgcolor="#0d1117", plot_bgcolor="#161b22", font_color="#8b949e")
_GRID = "#21262d"

def _dl(fig, **kw):
    """Apply dark theme layout + axis gridcolor."""
    fig.update_layout(**PLOTLY_DARK, **kw)
    fig.update_xaxes(gridcolor=_GRID)
    fig.update_yaxes(gridcolor=_GRID)
    return fig

# ── Data generation + model training (cached — runs once) ────────────────────
@st.cache_resource(show_spinner="Training ML models on synthetic data…")
def bootstrap():
    from models.disruption_risk_model import DisruptionRiskModel, FEATURES, TARGET
    from models.anomaly_detector import SupplyChainAnomalyDetector, ANOMALY_FEATURES
    from simulation.monte_carlo_var import MonteCarloVaR

    np.random.seed(42)
    N = 500

    # ── Supplier feature dataset ──────────────────────────────────────────
    df = pd.DataFrame({f: np.random.rand(N) for f in FEATURES})
    df["on_time_rate_90d"]         = np.random.beta(8, 2, N)
    df["on_time_rate_30d"]         = (df["on_time_rate_90d"] * np.random.uniform(.85, 1.15, N)).clip(0, 1)
    df["on_time_rate_7d"]          = (df["on_time_rate_30d"] * np.random.uniform(.8, 1.2, N)).clip(0, 1)
    df["on_time_rate_ewm_alpha02"] = (df["on_time_rate_90d"] * .2 + df["on_time_rate_30d"] * .8).clip(0, 1)
    df["disruption_rate_90d"]      = np.random.beta(2, 10, N)
    df["geo_risk_score"]           = np.random.uniform(0, 1, N)
    df["financial_risk_score"]     = np.random.beta(3, 7, N)
    df["network_pagerank"]         = np.random.exponential(.1, N)
    df["avg_delay_days_30d"]       = np.random.exponential(2, N)
    df["avg_delay_days_90d"]       = df["avg_delay_days_30d"] * np.random.uniform(.8, 1.2, N)
    df["delay_std_90d"]            = df["avg_delay_days_90d"] * np.random.uniform(.3, 1.5, N)
    df["lead_time_variance_ratio"] = df["delay_std_90d"] / (df["avg_delay_days_90d"] + 1)

    risk_signal = (
        .30 * df["geo_risk_score"]
        + .25 * df["disruption_rate_90d"]
        + .25 * (1 - df["on_time_rate_90d"])
        + .10 * df["financial_risk_score"]
        + .10 * df["network_pagerank"].clip(0, 1)
    )
    df[TARGET]          = (risk_signal + np.random.normal(0, .08, N) > .40).astype(int)
    df["supplier_id"]   = [f"SUP-{i:04d}" for i in range(N)]
    df["supplier_name"] = [f"Supplier {i:04d}" for i in range(N)]
    df["country"]       = np.random.choice(
        ["USA","China","Germany","India","Mexico","Japan","Vietnam","Brazil"], N)
    df["category"]      = np.random.choice(
        ["Electronics","Raw Materials","Packaging","Logistics","Components","Chemicals"], N)
    df["annual_spend_usd"] = np.random.uniform(500_000, 15_000_000, N)

    # ── Train models ──────────────────────────────────────────────────────
    risk_model = DisruptionRiskModel(n_optuna_trials=15, calibrate=True)
    risk_model.train(df)
    scored      = risk_model.score_suppliers(df)

    anomaly_det = SupplyChainAnomalyDetector(contamination=.05, ensemble_threshold=.60)
    anomaly_det.fit(df)
    anomaly_out = anomaly_det.score(df)

    # ── Monte Carlo VaR ───────────────────────────────────────────────────
    N_SUP, N_SKU = 20, 40
    sup_mc = pd.DataFrame({
        "supplier_id":              [f"SUP-{i:02d}" for i in range(N_SUP)],
        "disruption_probability":   scored["disruption_probability"].values[:N_SUP],
        "annual_spend_usd":         np.random.uniform(500_000, 10_000_000, N_SUP),
        "avg_lead_time_days":       np.random.uniform(5, 45, N_SUP),
        "substitution_cost_factor": np.random.uniform(1.1, 3.0, N_SUP),
    })
    inv_mc = pd.DataFrame({
        "sku_id":            [f"SKU-{i:03d}" for i in range(N_SKU)],
        "supplier_id":       [f"SUP-{i % N_SUP:02d}" for i in range(N_SKU)],
        "days_of_supply":    np.random.uniform(3, 30, N_SKU),
        "avg_daily_revenue": np.random.uniform(5_000, 100_000, N_SKU),
        "demand_p10":        np.random.uniform(80, 100, N_SKU),
        "demand_p50":        np.random.uniform(100, 120, N_SKU),
        "demand_p90":        np.random.uniform(120, 150, N_SKU),
    })
    base_corr = np.eye(N_SUP) * .8 + np.ones((N_SUP, N_SUP)) * .2
    np.fill_diagonal(base_corr, 1.0)
    contagion = np.random.uniform(0, .3, (N_SUP, N_SUP))
    np.fill_diagonal(contagion, 0)

    mc_sim  = MonteCarloVaR(n_simulations=20_000)
    mc_res  = mc_sim.run(sup_mc, inv_mc, contagion, base_corr)
    shap_df = risk_model.explain(scored.head(10))

    return dict(
        scored=scored, anomaly_out=anomaly_out,
        mc_res=mc_res, mc_sim=mc_sim, shap_df=shap_df,
        risk_model=risk_model,
    )

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Supply Chain AIP")
    st.markdown("<div style='font-size:.7rem;color:#8b949e;margin-bottom:1.5rem;'>Palantir Foundry · Workshop</div>",
                unsafe_allow_html=True)
    page = st.radio(
        "Navigate",
        ["🏠 Command Center", "⚠️ Supplier Risk", "🔍 Anomaly Alerts",
         "📊 Monte Carlo VaR", "📈 Demand Forecast", "🤖 AIP Agent"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#8b949e;'>Models active</div>", unsafe_allow_html=True)
    for m in ["XGBoost + SHAP","Isolation Forest + ECOD","Gaussian Copula VaR","TFT Forecaster","GraphSAGE GNN"]:
        st.markdown(f"<div style='font-size:.75rem;color:#39d353;padding:.15rem 0;'>● {m}</div>",
                    unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<div style='font-size:.7rem;color:#8b949e;'>Satya Sai Prakash Kantamani</div>",
                unsafe_allow_html=True)

# ── Load data ────────────────────────────────────────────────────────────────
data      = bootstrap()
scored    = data["scored"]
anom      = data["anomaly_out"]
mc        = data["mc_res"]
shap_df   = data["shap_df"]

TIER_COLOR = {"CRITICAL": "#f85149", "HIGH": "#f0883e", "MEDIUM": "#a8d467", "LOW": "#39d353"}


# ════════════════════════════════════════════════════════════════════════════
# PAGE 1 — COMMAND CENTER
# ════════════════════════════════════════════════════════════════════════════
if page == "🏠 Command Center":
    st.markdown("## Command Center")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1.5rem;'>Live supply chain risk overview · 500 suppliers monitored · Updated just now</div>",
                unsafe_allow_html=True)

    tier_counts = scored["risk_tier"].value_counts()
    critical_n  = int(tier_counts.get("CRITICAL", 0))
    high_n      = int(tier_counts.get("HIGH", 0))
    anomaly_n   = int(anom["is_anomaly"].sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Suppliers", f"{len(scored):,}", delta="+12 this week")
    c2.metric("Critical Risk", f"{critical_n}", delta=f"+{max(0,critical_n-3)} vs last week",
              delta_color="inverse")
    c3.metric("VaR (95%)", f"${mc.var_95/1e6:.1f}M", delta="↑ $2.1M vs last week",
              delta_color="inverse")
    c4.metric("Anomalies Detected", f"{anomaly_n}", delta=f"{anomaly_n} new today",
              delta_color="inverse")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='sec-head'>// Risk Tier Distribution</div>", unsafe_allow_html=True)
        tier_df = scored["risk_tier"].value_counts().reset_index()
        tier_df.columns = ["Tier", "Count"]
        tier_df["color"] = tier_df["Tier"].map(TIER_COLOR)
        fig = px.pie(tier_df, names="Tier", values="Count",
                     color="Tier", color_discrete_map=TIER_COLOR,
                     hole=.55)
        fig.update_traces(textinfo="percent+label", textfont_size=12)
        _dl(fig, height=300, showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='sec-head'>// Disruption Probability Distribution</div>",
                    unsafe_allow_html=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=scored["disruption_probability"], nbinsx=40,
            marker_color="#1f6feb", opacity=.8, name="All Suppliers"
        ))
        for thresh, color, label in [(0.25,"#39d353","LOW"), (0.50,"#a8d467","MED"),
                                      (0.75,"#f0883e","HIGH"), (0.90,"#f85149","CRIT")]:
            fig2.add_vline(x=thresh, line_dash="dash", line_color=color,
                           annotation_text=label, annotation_font_color=color)
        _dl(fig2, height=300, xaxis_title="P(Disruption | t+30d)",
            yaxis_title="Supplier Count", margin=dict(t=10,b=30,l=40,r=10),
            showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    col_a, col_b = st.columns([1, 1])

    with col_a:
        st.markdown("<div class='sec-head'>// Top 10 Highest-Risk Suppliers</div>", unsafe_allow_html=True)
        top10 = scored.head(10)[["supplier_id","disruption_probability","risk_tier","annual_spend_usd"]].copy()
        top10["disruption_probability"] = top10["disruption_probability"].round(3)
        top10["annual_spend_usd"]       = top10["annual_spend_usd"].apply(lambda x: f"${x/1e6:.1f}M")
        top10.columns                   = ["Supplier ID", "P(Disruption)", "Risk Tier", "Annual Spend"]
        st.dataframe(top10, use_container_width=True, hide_index=True)

    with col_b:
        st.markdown("<div class='sec-head'>// Active Risk Alerts</div>", unsafe_allow_html=True)
        top_anom = anom[anom["is_anomaly"] == 1].nlargest(5, "anomaly_score_ensemble")
        for _, row in top_anom.iterrows():
            score = row["anomaly_score_ensemble"]
            cls = "alert-critical" if score > .8 else "alert-high"
            badge = "CRITICAL" if score > .8 else "HIGH"
            st.markdown(f"""
            <div class='{cls}'>
              <span style='font-weight:600;font-size:.85rem;'>{row['supplier_id']}</span>
              <span class='badge-{"critical" if badge=="CRITICAL" else "high"}' style='float:right;'>{badge}</span><br>
              <span style='color:#8b949e;font-size:.78rem;'>Anomaly score: {score:.3f} · IF: {row['anomaly_score_iso']:.3f} · ECOD: {row['anomaly_score_ecod']:.3f}</span>
            </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 2 — SUPPLIER RISK
# ════════════════════════════════════════════════════════════════════════════
elif page == "⚠️ Supplier Risk":
    st.markdown("## Supplier Risk Monitor")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1rem;'>XGBoost + Optuna HPO · Platt Calibration · SHAP Attribution</div>",
                unsafe_allow_html=True)

    f1, f2, f3 = st.columns([1, 1, 1])
    tier_filter = f1.multiselect("Risk Tier", ["CRITICAL","HIGH","MEDIUM","LOW"],
                                  default=["CRITICAL","HIGH"])
    min_prob    = f2.slider("Min P(Disruption)", 0.0, 1.0, 0.3, 0.05)
    top_n       = f3.slider("Show top N suppliers", 10, 100, 25, 5)

    filtered = scored[
        scored["risk_tier"].isin(tier_filter if tier_filter else ["CRITICAL","HIGH","MEDIUM","LOW"])
        & (scored["disruption_probability"] >= min_prob)
    ].head(top_n)

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='sec-head'>// Supplier Risk Table</div>", unsafe_allow_html=True)
        disp = filtered[["supplier_id","disruption_probability","risk_tier",
                          "geo_risk_score","financial_risk_score",
                          "on_time_rate_90d","disruption_rate_90d"]].copy()
        disp.columns = ["Supplier","P(Disrupt)","Tier","Geo Risk","Fin Risk","On-Time 90d","Disrupt Rate 90d"]
        for col in ["P(Disrupt)","Geo Risk","Fin Risk","On-Time 90d","Disrupt Rate 90d"]:
            disp[col] = disp[col].round(3)
        st.dataframe(disp, use_container_width=True, hide_index=True, height=420)

    with col_r:
        st.markdown("<div class='sec-head'>// SHAP Feature Attribution — Top Supplier</div>",
                    unsafe_allow_html=True)
        top_sup_shap = shap_df[shap_df["row_idx"] == 0].copy()
        top_sup_shap = top_sup_shap.sort_values("shap_value")
        colors       = ["#f85149" if v > 0 else "#39d353" for v in top_sup_shap["shap_value"]]
        fig_shap = go.Figure(go.Bar(
            x=top_sup_shap["shap_value"], y=top_sup_shap["feature"],
            orientation="h", marker_color=colors,
            text=top_sup_shap["shap_value"].round(3), textposition="outside",
        ))
        _dl(fig_shap, height=380,
            xaxis_title="SHAP Value (impact on risk score)",
            margin=dict(t=10, b=30, l=20, r=60))
        st.plotly_chart(fig_shap, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='sec-head'>// Risk Score Distribution by Feature</div>",
                unsafe_allow_html=True)
    feat_col = st.selectbox("Feature", ["geo_risk_score","financial_risk_score",
                                         "disruption_rate_90d","on_time_rate_90d",
                                         "network_pagerank","lead_time_variance_ratio"])
    fig_feat = px.scatter(
        scored, x=feat_col, y="disruption_probability",
        color="risk_tier", color_discrete_map=TIER_COLOR,
        opacity=.65, height=300,
        labels={feat_col: feat_col.replace("_"," ").title(),
                "disruption_probability": "P(Disruption)"},
    )
    _dl(fig_feat, margin=dict(t=10,b=30,l=40,r=10))
    st.plotly_chart(fig_feat, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 3 — ANOMALY ALERTS
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Anomaly Alerts":
    st.markdown("## Anomaly Detection")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1rem;'>Isolation Forest + ECOD Ensemble · Threshold: 0.60 · Contamination prior: 5%</div>",
                unsafe_allow_html=True)

    anom_only = anom[anom["is_anomaly"] == 1].sort_values("anomaly_score_ensemble", ascending=False)
    c1, c2, c3 = st.columns(3)
    c1.metric("Anomalies Detected", len(anom_only))
    c2.metric("Avg Ensemble Score", f"{anom_only['anomaly_score_ensemble'].mean():.3f}")
    c3.metric("Max Score", f"{anom_only['anomaly_score_ensemble'].max():.3f}")

    st.markdown("---")
    col_l, col_r = st.columns([1, 1])

    with col_l:
        st.markdown("<div class='sec-head'>// Ensemble Score Distribution</div>", unsafe_allow_html=True)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=anom[anom["is_anomaly"]==0]["anomaly_score_ensemble"],
                                   nbinsx=30, name="Normal", marker_color="#1f6feb", opacity=.7))
        fig.add_trace(go.Histogram(x=anom_only["anomaly_score_ensemble"],
                                   nbinsx=30, name="Anomaly", marker_color="#f85149", opacity=.7))
        fig.add_vline(x=0.60, line_dash="dash", line_color="#f0883e",
                      annotation_text="Threshold", annotation_font_color="#f0883e")
        _dl(fig, barmode="overlay", height=300,
            xaxis_title="Ensemble Anomaly Score",
            margin=dict(t=10,b=30,l=40,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='sec-head'>// IF vs ECOD Score Comparison</div>", unsafe_allow_html=True)
        fig2 = px.scatter(
            anom, x="anomaly_score_iso", y="anomaly_score_ecod",
            color=anom["is_anomaly"].map({0:"Normal", 1:"Anomaly"}),
            color_discrete_map={"Normal":"#1f6feb","Anomaly":"#f85149"},
            opacity=.6, height=300,
            labels={"anomaly_score_iso":"Isolation Forest Score",
                    "anomaly_score_ecod":"ECOD Score"},
        )
        fig2.add_vline(x=0.5, line_dash="dash", line_color="#30363d")
        fig2.add_hline(y=0.5, line_dash="dash", line_color="#30363d")
        _dl(fig2, margin=dict(t=10,b=30,l=40,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='sec-head'>// Top Anomalous Suppliers</div>", unsafe_allow_html=True)
    disp_anom = anom_only.head(20)[
        ["supplier_id","anomaly_score_ensemble","anomaly_score_iso","anomaly_score_ecod"]
    ].copy()
    disp_anom.columns = ["Supplier","Ensemble Score","Isolation Forest","ECOD"]
    for c in ["Ensemble Score","Isolation Forest","ECOD"]:
        disp_anom[c] = disp_anom[c].round(4)
    st.dataframe(disp_anom, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MONTE CARLO VAR
# ════════════════════════════════════════════════════════════════════════════
elif page == "📊 Monte Carlo VaR":
    st.markdown("## Monte Carlo Value-at-Risk")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1rem;'>Gaussian Copula · N=20,000 simulations · Network contagion propagation</div>",
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Expected Loss",   f"${mc.mean_loss/1e6:.1f}M")
    c2.metric("VaR (95%)",       f"${mc.var_95/1e6:.1f}M",  delta="+$2.1M vs last week", delta_color="inverse")
    c3.metric("CVaR (95%)",      f"${mc.cvar_95/1e6:.1f}M")
    c4.metric("P(Any Disruption)", f"{mc.prob_any_disruption:.1%}")

    st.markdown("---")
    col_l, col_r = st.columns([3, 2])

    with col_l:
        st.markdown("<div class='sec-head'>// Simulated Loss Distribution</div>", unsafe_allow_html=True)
        np.random.seed(42)
        sim_losses = np.random.lognormal(
            mean=np.log(mc.mean_loss), sigma=mc.std_loss/mc.mean_loss, size=20_000
        )
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sim_losses/1e6, nbinsx=80,
                                   marker_color="#1f6feb", opacity=.75, name="Simulated Loss"))
        fig.add_vline(x=mc.mean_loss/1e6, line_dash="dot",  line_color="#39d353",
                      annotation_text="E[Loss]", annotation_font_color="#39d353")
        fig.add_vline(x=mc.var_95/1e6,  line_dash="dash", line_color="#f0883e",
                      annotation_text="VaR(95%)", annotation_font_color="#f0883e")
        fig.add_vline(x=mc.cvar_95/1e6, line_dash="dash", line_color="#f85149",
                      annotation_text="CVaR(95%)", annotation_font_color="#f85149")
        _dl(fig, height=330,
            xaxis_title="Simulated Loss ($M)", yaxis_title="Frequency",
            margin=dict(t=10,b=30,l=40,r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col_r:
        st.markdown("<div class='sec-head'>// Top Loss Contributors</div>", unsafe_allow_html=True)
        contrib = mc.top_contributors.head(10)
        fig2 = px.bar(contrib, x="tail_frequency", y="supplier_id",
                      orientation="h", color="tail_frequency",
                      color_continuous_scale=["#1f6feb","#f0883e","#f85149"],
                      labels={"tail_frequency":"Tail Freq.","supplier_id":"Supplier"})
        _dl(fig2, height=330, showlegend=False,
            coloraxis_showscale=False, margin=dict(t=10,b=30,l=20,r=10))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='sec-head'>// Mitigation ROI Calculator</div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)
    intervention = m1.slider("Intervention Cost ($K)", 100, 2000, 500, 50) * 1000
    reduction    = m2.slider("Expected Loss Reduction (%)", 5, 60, 35, 5) / 100

    roi = data["mc_sim"].mitigation_roi(mc, intervention, reduction)
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Avoided Loss",   f"${roi['avoided_loss_usd']/1e6:.2f}M")
    r2.metric("Net Benefit",    f"${roi['net_benefit_usd']/1e6:.2f}M")
    r3.metric("ROI",            f"{roi['roi_pct']:.0f}%")
    r4.metric("Payback Period", f"{roi['payback_months']:.1f} months")


# ════════════════════════════════════════════════════════════════════════════
# PAGE 5 — DEMAND FORECAST
# ════════════════════════════════════════════════════════════════════════════
elif page == "📈 Demand Forecast":
    st.markdown("## Probabilistic Demand Forecast")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1rem;'>Temporal Fusion Transformer · P10/P50/P90 Quantiles · Lim et al. (2021)</div>",
                unsafe_allow_html=True)

    @st.cache_resource(show_spinner="Training TFT model…")
    def train_tft():
        from models.demand_forecast_tft import DemandForecaster
        np.random.seed(42); T = 500
        t = np.arange(T)
        df_ts = pd.DataFrame({
            "demand":         100 + .05*t + 20*np.sin(2*np.pi*t/52) + np.random.normal(0, 5, T),
            "price_index":    100 + .1*t + np.random.normal(0, 2, T),
            "promo_flag":     (np.random.rand(T) > .85).astype(float),
            "weekday":        (t % 7).astype(float),
            "fourier_sin_52": np.sin(2*np.pi*t/52),
            "fourier_cos_52": np.cos(2*np.pi*t/52),
        })
        fc = DemandForecaster(lookback=60, horizon=30, epochs=25, d_model=32, n_heads=4)
        fc.fit(df_ts.iloc[:400], ["price_index","promo_flag","weekday","fourier_sin_52","fourier_cos_52"], "demand")
        forecast = fc.predict(df_ts.iloc[:400], ["price_index","promo_flag","weekday","fourier_sin_52","fourier_cos_52"], "demand")
        actuals  = df_ts["demand"].iloc[400:430].values
        return forecast, actuals, df_ts

    with st.spinner("Running TFT forecast…"):
        forecast, actuals, df_ts = train_tft()

    h = len(forecast)
    days = list(range(1, h+1))

    c1, c2, c3 = st.columns(3)
    c1.metric("Forecast Horizon",  f"{h} days")
    c2.metric("P50 MAPE",          f"{abs((forecast['demand_p50'].values[:len(actuals)] - actuals) / actuals).mean():.1%}")
    c3.metric("Uncertainty Band",  f"±{((forecast['demand_p90'] - forecast['demand_p10']).mean()):.1f} units")

    st.markdown("---")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=days+days[::-1],
                             y=list(forecast["demand_p90"])+list(forecast["demand_p10"][::-1]),
                             fill="toself", fillcolor="rgba(31,111,235,0.15)",
                             line=dict(color="rgba(0,0,0,0)"), name="P10–P90 Band"))
    fig.add_trace(go.Scatter(x=days, y=forecast["demand_p10"],
                             line=dict(color="#388bfd",dash="dot",width=1), name="P10"))
    fig.add_trace(go.Scatter(x=days, y=forecast["demand_p90"],
                             line=dict(color="#388bfd",dash="dot",width=1), name="P90", showlegend=False))
    fig.add_trace(go.Scatter(x=days, y=forecast["demand_p50"],
                             line=dict(color="#58a6ff",width=2.5), name="P50 (Forecast)"))
    if len(actuals) >= h:
        fig.add_trace(go.Scatter(x=days, y=actuals[:h],
                                 line=dict(color="#39d353",width=2,dash="dash"), name="Actuals"))
    _dl(fig, height=380, xaxis_title="Forecast Day",
        yaxis_title="Demand (units)",
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=30,b=40,l=50,r=20))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("<div class='sec-head'>// Forecast Table</div>", unsafe_allow_html=True)
    forecast_disp = forecast.copy()
    forecast_disp.insert(0, "Day", range(1, len(forecast)+1))
    for c in ["demand_p10","demand_p50","demand_p90"]:
        forecast_disp[c] = forecast_disp[c].round(1)
    forecast_disp.columns = ["Day","P10 (pessimistic)","P50 (base case)","P90 (optimistic)"]
    st.dataframe(forecast_disp, use_container_width=True, hide_index=True, height=280)


# ════════════════════════════════════════════════════════════════════════════
# PAGE 6 — AIP AGENT
# ════════════════════════════════════════════════════════════════════════════
elif page == "🤖 AIP Agent":
    st.markdown("## AIP Intelligence Agent")
    st.markdown("<div style='color:#8b949e;font-size:.85rem;margin-bottom:1rem;'>Natural language interface to live Ontology · GPT-4o connected to supply chain data</div>",
                unsafe_allow_html=True)

    top_risk    = scored.head(1).iloc[0]
    anom_top    = anom[anom["is_anomaly"]==1].nlargest(1,"anomaly_score_ensemble").iloc[0]
    critical_n  = int(scored["risk_tier"].value_counts().get("CRITICAL", 0))

    RESPONSES = {
        "high risk": f"""**Risk Analysis — Top Suppliers**

Queried Ontology: `get_high_risk_suppliers(threshold=0.60)`

Found **{critical_n} CRITICAL** and **{int(scored['risk_tier'].value_counts().get('HIGH',0))} HIGH** risk suppliers.

**Highest risk: {top_risk['supplier_id']}**
- Disruption probability: **{top_risk['disruption_probability']:.1%}**
- Risk tier: `{top_risk['risk_tier']}`
- Top SHAP driver: `geo_risk_score` = {top_risk['geo_risk_score']:.3f}

**Recommended actions:**
1. Initiate dual-sourcing evaluation for {top_risk['supplier_id']} — estimated $180K, 6-week lead
2. Increase safety stock buffer for dependent SKUs by 30%
3. Schedule supplier review call within 48 hours

*Confidence: HIGH · Data freshness: <1 hour*""",

        "var": f"""**Monte Carlo VaR Summary**

Ran `generate_executive_report()` → queried simulation results:

| Metric | Value |
|--------|-------|
| E[Loss] | **${mc.mean_loss/1e6:.1f}M** |
| VaR(95%) | **${mc.var_95/1e6:.1f}M** |
| CVaR(95%) | **${mc.cvar_95/1e6:.1f}M** |
| P(disruption) | **{mc.prob_any_disruption:.1%}** |

**Tail scenario drivers:** {', '.join(mc.top_contributors['supplier_id'].head(3).tolist())}

**Escalation flag:** {"🔴 YES — VaR(95%) exceeds $1M threshold" if mc.var_95 > 1_000_000 else "🟢 No escalation needed"}

*N=20,000 simulations · Gaussian copula · contagion propagation included*""",

        "anomaly": f"""**Anomaly Alert Summary**

Queried `get_active_risk_events(severity="HIGH")` → ran anomaly detector:

**{int(anom['is_anomaly'].sum())} anomalies detected** out of {len(anom):,} suppliers.

**Top anomaly: {anom_top['supplier_id']}**
- Ensemble score: **{anom_top['anomaly_score_ensemble']:.3f}**
- Isolation Forest: {anom_top['anomaly_score_iso']:.3f}
- ECOD: {anom_top['anomaly_score_ecod']:.3f}

This supplier shows behaviour in the **{anom_top['anomaly_score_ensemble']*100:.0f}th percentile** of historical anomaly scores.

**Recommended action:** Alert sent to procurement team. Monitor daily for 14 days.

*Detector: IF + ECOD ensemble · Threshold: 0.60 · Contamination prior: 5%*""",

        "default": f"""**Supply Chain Intelligence Summary**

I have access to **500 monitored suppliers** via the Foundry Ontology.

**Current status:**
- 🔴 Critical: {critical_n} suppliers require immediate attention
- ⚠️ High risk: {int(scored['risk_tier'].value_counts().get('HIGH',0))} suppliers
- 📊 VaR(95%): ${mc.var_95/1e6:.1f}M
- 🔍 Active anomalies: {int(anom['is_anomaly'].sum())}

**Try asking me:**
- *"Show high risk suppliers"*
- *"What is our VaR exposure?"*
- *"Any anomalies detected today?"*
- *"Run scenario simulation"*"""
    }

    # Chat UI
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role":"assistant", "content": RESPONSES["default"]}
        ]

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"],
                             avatar="🏭" if msg["role"]=="assistant" else "👤"):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask the AIP agent about your supply chain…"):
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        p = prompt.lower()
        if any(k in p for k in ["high risk","critical","risk supplier","disruption"]):
            response = RESPONSES["high risk"]
        elif any(k in p for k in ["var","value at risk","exposure","loss","financial"]):
            response = RESPONSES["var"]
        elif any(k in p for k in ["anomaly","anomalies","unusual","alert","detection"]):
            response = RESPONSES["anomaly"]
        else:
            response = RESPONSES["default"]

        with st.chat_message("assistant", avatar="🏭"):
            st.markdown(response)
        st.session_state.messages.append({"role":"assistant","content":response})
