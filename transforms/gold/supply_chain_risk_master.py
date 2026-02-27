"""
Supply Chain Risk Master — Foundry Gold Transform
==================================================
Builds the final feature table for the ML risk engine.
Joins supplier, shipment, order, and geopolitical risk data
into a single enriched dataset ready for model scoring.

Author: Satya Sai Prakash Kantamani
"""

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from transforms.api import transform, Input, Output


@transform(
    output=Output("/supply-chain/gold/supply_chain_risk_master"),
    suppliers=Input("/supply-chain/silver/suppliers_clean"),
    shipments=Input("/supply-chain/silver/shipments_merged"),
    geo_risk=Input("/supply-chain/silver/geo_risk_scored"),
    orders=Input("/supply-chain/silver/orders_enriched"),
)
def compute(output, suppliers, shipments, geo_risk, orders):
    df_sup = suppliers.dataframe()
    df_ship = shipments.dataframe()
    df_geo = geo_risk.dataframe()
    df_ord = orders.dataframe()

    # ── Rolling 90-day on-time delivery rate per supplier ──────────────────────
    w90 = Window.partitionBy("supplier_id").orderBy("order_date") \
                .rangeBetween(-90 * 86400, 0)

    df_ord = df_ord.withColumn(
        "rolling_on_time_rate",
        F.avg(F.col("on_time_flag").cast("double")).over(w90)
    ).withColumn(
        "rolling_avg_delay_days",
        F.avg("delay_days").over(w90)
    ).withColumn(
        "order_volume_90d",
        F.count("order_id").over(w90)
    )

    # ── Latest shipment metrics per supplier ───────────────────────────────────
    ship_agg = df_ship.groupBy("supplier_id").agg(
        F.mean("delay_days").alias("avg_shipment_delay"),
        F.stddev("delay_days").alias("delay_volatility"),
        F.sum(F.col("status").isin(["DELAYED", "LOST"]).cast("int"))
         .alias("disrupted_shipments_90d"),
        F.count("shipment_id").alias("total_shipments_90d"),
    ).withColumn(
        "disruption_rate",
        F.col("disrupted_shipments_90d") / F.col("total_shipments_90d")
    )

    # ── Join all signals ───────────────────────────────────────────────────────
    df = df_sup \
        .join(ship_agg, on="supplier_id", how="left") \
        .join(df_geo.select("country_code", "geo_risk_score", "political_stability_index"),
              on="country_code", how="left") \
        .join(df_ord.select("supplier_id", "rolling_on_time_rate",
                             "rolling_avg_delay_days", "order_volume_90d"),
              on="supplier_id", how="left")

    # ── Composite risk features ────────────────────────────────────────────────
    df = df.withColumn(
        "lead_time_variance",
        F.col("delay_volatility") / (F.col("avg_shipment_delay") + F.lit(1e-6))
    ).withColumn(
        "supplier_concentration_risk",
        F.col("order_volume_90d") / F.lit(1000)           # normalised spend share
    ).withColumn(
        "composite_risk_input",
        0.30 * F.col("geo_risk_score") +
        0.25 * F.col("disruption_rate") +
        0.20 * (F.lit(1) - F.col("rolling_on_time_rate")) +
        0.15 * F.col("lead_time_variance") +
        0.10 * F.col("financial_risk_score")
    ).withColumn(
        "risk_tier",
        F.when(F.col("composite_risk_input") >= 0.7, "CRITICAL")
         .when(F.col("composite_risk_input") >= 0.4, "HIGH")
         .when(F.col("composite_risk_input") >= 0.2, "MEDIUM")
         .otherwise("LOW")
    ).withColumn(
        "last_updated", F.current_timestamp()
    )

    output.write_dataframe(df)
