"""
AIP Prompt Templates — Structured LLM Prompt Engineering
==========================================================
Structured prompt templates for the AIP supply chain intelligence agent.

Design principles:
  - Chain-of-thought reasoning for complex multi-signal analysis
  - Structured output enforcement (JSON schema) for downstream action routing
  - SHAP attribution injection for grounded, explainable narratives
  - Calibrated uncertainty communication (no false precision)
  - Escalation criteria baked into system prompt

Author: Satya Sai Prakash Kantamani
"""

from string import Template


# ── System Prompt ──────────────────────────────────────────────────────────
SYSTEM_PROMPT = """\
You are an expert supply chain risk intelligence agent operating on live \
enterprise data via Palantir Foundry's Ontology.

Your responsibilities:
1. Analyse supply chain disruption signals with precision and quantitative rigor
2. Ground every claim in data retrieved from Ontology queries — never speculate
3. Communicate uncertainty explicitly using ML confidence intervals
4. Prioritise actionable recommendations over descriptive summaries
5. Escalate automatically when VaR(95%) exceeds $1M or risk tier = CRITICAL

Output format rules:
- Lead with the single most important finding
- Support with 2–3 data points (supplier IDs, probabilities, dollar impacts)
- List recommended actions in priority order with estimated cost/benefit
- Flag any limitations in data freshness or model confidence
- Keep responses under 400 words unless an executive report is explicitly requested

You have access to these actions:
  get_high_risk_suppliers(threshold) | get_delayed_shipments(min_delay_days)
  get_stockout_risks(days_of_supply) | get_active_risk_events(severity)
  run_scenario_simulation(supplier_ids) | create_purchase_order(...)
  send_supplier_alert(...) | generate_executive_report(scope)
"""


# ── Root Cause Analysis ────────────────────────────────────────────────────
ROOT_CAUSE_TEMPLATE = Template("""\
## Supplier Risk Analysis Request

**Supplier:** $supplier_name (ID: $supplier_id)
**Current Disruption Probability:** $disruption_probability (Risk Tier: $risk_tier)
**Model Confidence:** $confidence_interval

**SHAP Feature Attribution (top drivers):**
$shap_attribution

**Recent Ontology Context:**
- On-time rate trend (90d): $on_time_rate_trend
- Recent RiskEvents linked: $linked_risk_events
- Active delayed shipments: $delayed_shipments_count
- Inventory SKUs at risk: $sku_at_risk_count

**Network Contagion:**
- Second-order suppliers affected if disrupted: $contagion_suppliers
- Network PageRank (supply importance): $pagerank_score

**Task:**
1. Identify the primary root cause of the elevated risk score, grounded in the \
   SHAP attribution above
2. Quantify the potential business impact (revenue at risk, lead time extension)
3. Recommend 2–3 prioritised mitigation actions with estimated cost and timeline
4. State confidence in your assessment and any data gaps

Use chain-of-thought reasoning. Do not fabricate data points not present above.
""")


# ── Executive Weekly Briefing ──────────────────────────────────────────────
EXECUTIVE_BRIEFING_TEMPLATE = Template("""\
## Weekly Supply Chain Intelligence Briefing

**Reporting Period:** $start_date to $end_date
**Generated:** $timestamp

**Portfolio Risk Overview:**
- Total suppliers monitored: $total_suppliers
- Critical risk tier: $critical_count ($critical_pct% of spend)
- High risk tier: $high_count
- New risk events this week: $new_risk_events

**Monte Carlo VaR Summary (30-day horizon, N=50,000 simulations):**
- Expected loss (mean scenario): $mean_loss_usd
- VaR(95%): $var_95_usd
- CVaR(95%) — Expected Shortfall: $cvar_95_usd
- P(material disruption > $500K): $prob_material_disruption

**Top 5 Highest-Risk Suppliers:**
$top_5_suppliers_table

**Active Risk Events:**
$active_risk_events_summary

**Inventory Alerts:**
- SKUs with < 7 days supply: $critical_stockout_count
- SKUs with < 14 days supply: $high_stockout_count
- Estimated revenue at risk from stockouts: $stockout_revenue_risk

**Task:**
Produce a concise executive briefing (max 300 words) that:
1. Opens with the single most urgent risk requiring C-suite attention
2. Summarises the VaR position vs prior week trend
3. Highlights any new developments (new risk events, supplier deterioration)
4. Closes with 3 recommended actions for the coming week

Tone: direct, quantitative, no filler. Audience: CPO and CFO.
""")


# ── Natural Language Query Routing ────────────────────────────────────────
NL_QUERY_ROUTER_TEMPLATE = Template("""\
## Query Classification and Data Retrieval Plan

**User Query:** "$user_query"

**Available Actions:**
  get_high_risk_suppliers | get_delayed_shipments | get_stockout_risks
  get_active_risk_events  | run_scenario_simulation | generate_executive_report

**Task:**
1. Classify this query into one of: [risk_analysis, delay_status, inventory_risk,
   market_events, scenario_planning, executive_summary, action_request]
2. Identify which Ontology actions to call and in what order
3. List any clarifying parameters needed (supplier_id, threshold, date range)
4. Output a JSON execution plan:

{
  "query_type": "<classification>",
  "actions": [
    {"action": "<action_name>", "params": {<params>}, "reason": "<why>"}
  ],
  "requires_human_approval": <true/false>,
  "escalation_flag": <true/false>,
  "escalation_reason": "<if applicable>"
}

Return only the JSON. Do not add explanation.
""")


# ── Anomaly Narrative Generator ───────────────────────────────────────────
ANOMALY_NARRATIVE_TEMPLATE = Template("""\
## Anomaly Detection Event — Narrative Generation

**Anomaly ID:** $anomaly_id
**Detected:** $detection_timestamp
**Anomaly Score:** $anomaly_score (threshold: $threshold)
**Detector:** $detector_type (Isolation Forest / ECOD ensemble)

**Anomalous Signal:**
- Feature: $anomalous_feature
- Observed value: $observed_value
- Expected range (μ ± 2σ): [$expected_low, $expected_high]
- Z-score: $z_score
- Historical percentile: $percentile

**Linked Ontology Objects:**
$linked_objects

**Recent similar events (historical):**
$similar_historical_events

**Task:**
Generate a concise anomaly alert (max 150 words) for the procurement operations team:
1. What was detected and why it is unusual (cite the z-score / percentile)
2. Which supplier/SKU/route is affected
3. Potential causes (geopolitical, operational, financial)
4. Recommended immediate action (if any)
5. Monitoring recommendation (frequency, threshold adjustment)

Write in plain English. Avoid jargon. Include a severity label: [LOW/MEDIUM/HIGH/CRITICAL].
""")


# ── Scenario What-If Analysis ─────────────────────────────────────────────
SCENARIO_WHATIF_TEMPLATE = Template("""\
## Scenario Analysis: What-If Disruption Assessment

**Scenario:** $scenario_description
**Affected Suppliers:** $affected_supplier_list
**Scenario Probability (from ML model):** $scenario_probability
**Horizon:** $horizon_days days

**Monte Carlo Simulation Results (N=50,000):**
- Revenue at risk (mean): $mean_revenue_risk
- VaR(95%): $var_95
- CVaR(95%): $cvar_95
- Inventory SKUs reaching stockout within $horizon_days days: $stockout_skus

**Network Contagion (GraphSAGE):**
- Second-order suppliers impacted: $second_order_count
- Third-order suppliers impacted: $third_order_count
- Total spend coverage: $total_spend_pct%

**Mitigation Options Modelled:**
$mitigation_options_table

**Task:**
Provide a structured what-if analysis:
1. Describe the scenario impact in business terms (not just numbers)
2. Identify the 2–3 most exposed business units / product lines
3. Rank mitigation options by ROI (highest first) with clear trade-offs
4. Recommend a go/no-go decision on immediate mitigation with justification
5. Identify information gaps that would improve decision quality

Be direct. The audience is making a capital allocation decision.
""")


def build_root_cause_prompt(supplier_data: dict, shap_df) -> str:
    """Formats SHAP attribution into readable string and fills template."""
    shap_lines = "\n".join(
        f"  {row['feature']}: {row['shap_value']:+.4f} ({row['direction']})"
        for _, row in shap_df.iterrows()
    )
    return ROOT_CAUSE_TEMPLATE.substitute(
        **supplier_data,
        shap_attribution=shap_lines,
    )
