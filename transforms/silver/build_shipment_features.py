"""
Silver Layer — Shipment-Level Feature Engineering
==================================================
Transforms Bronze shipment events into a structured feature table for:
  (a) Real-time delay prediction (per-shipment inference)
  (b) Historical disruption analysis (training labels)
  (c) Route-level risk scoring (transit_route_risk feature)

Feature groups:
  - Shipment-level: cargo value, transport mode, incoterm risk, leg count
  - Temporal: departure seasonality, day-of-week effects, Q4 crunch flag
  - Route-level: historical exception rate, median delay on origin→dest pair
  - Supplier-level join: current risk tier, recent on-time rate

Author: Satya Sai Prakash Kantamani
"""

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform_df, Input, Output


# ── Route risk: minimum denominator to avoid sparse-route noise ──────────────
MIN_ROUTE_SHIPMENTS = 10


@transform_df(
    Output("/supply-chain/silver/shipment_features"),
    shipments=Input("/supply-chain/bronze/shipment_events"),
    supplier_features=Input("/supply-chain/silver/supplier_features"),
)
def build_shipment_features(
    shipments: DataFrame,
    supplier_features: DataFrame,
) -> DataFrame:
    """
    Produces one row per shipment with enriched features and a binary
    disruption label (delay > 3 days OR canonical_status == EXCEPTION).
    """

    # ── 1. Clean base ──────────────────────────────────────────────────────
    base = (
        shipments
        .filter(F.col("dq_score") >= 0.6)
        .withColumn("event_date",    F.to_date(F.col("event_ts")))
        .withColumn("cargo_value_usd", F.coalesce(F.col("cargo_value_usd"), F.lit(0.0)))
        .withColumn("delay_days",    F.coalesce(F.col("delay_days"), F.lit(0.0)))
    )

    # ── 2. Binary disruption label ────────────────────────────────────────
    # A shipment is labelled DISRUPTED if:
    #   - delay > 3 calendar days, OR
    #   - status is EXCEPTION / CUSTOMS_HOLD
    base = base.withColumn(
        "disruption_label",
        F.when(
            (F.col("delay_days") > 3)
            | F.col("canonical_status").isin("EXCEPTION", "CUSTOMS_HOLD"),
            F.lit(1)
        ).otherwise(F.lit(0))
        .cast(T.IntegerType())
    )

    # ── 3. Temporal features ──────────────────────────────────────────────
    base = (
        base
        .withColumn("departure_month",      F.month(F.col("expected_delivery")))
        .withColumn("departure_dow",        F.dayofweek(F.col("expected_delivery")))   # 1=Sun
        .withColumn("departure_quarter",    F.quarter(F.col("expected_delivery")))
        .withColumn("is_q4_crunch",         (F.col("departure_quarter") == 4).cast(T.IntegerType()))
        .withColumn("is_weekend_departure", F.col("departure_dow").isin(1, 7).cast(T.IntegerType()))
        # Seasonality: sine/cosine encoding of month
        .withColumn("month_sin",
            F.sin(F.lit(2 * 3.14159265) * F.col("departure_month").cast(T.DoubleType()) / F.lit(12.0))
        )
        .withColumn("month_cos",
            F.cos(F.lit(2 * 3.14159265) * F.col("departure_month").cast(T.DoubleType()) / F.lit(12.0))
        )
    )

    # ── 4. Route-level statistics ─────────────────────────────────────────
    # Grain: (origin_port, destination_port) historical performance
    route_stats_window = Window.partitionBy("origin_port", "destination_port")

    route_stats = (
        base
        .groupBy("origin_port", "destination_port")
        .agg(
            F.count("*").alias("route_total_shipments"),
            F.avg("disruption_label").alias("route_historical_exception_rate"),
            F.percentile_approx("delay_days", 0.5).alias("route_median_delay_days"),
            F.stddev("delay_days").alias("route_delay_std"),
            F.avg("cargo_value_usd").alias("route_avg_cargo_value"),
        )
        # Only trust routes with sufficient history
        .withColumn(
            "transit_route_risk",
            F.when(
                F.col("route_total_shipments") >= MIN_ROUTE_SHIPMENTS,
                F.col("route_historical_exception_rate")
            ).otherwise(F.lit(0.25))   # prior for sparse routes
        )
    )

    # ── 5. Join route stats back to shipment level ─────────────────────────
    base = base.join(route_stats, on=["origin_port", "destination_port"], how="left")

    # ── 6. Transport mode encoding ────────────────────────────────────────
    base = (
        base
        .withColumn("is_ocean_freight",   (F.col("transport_mode") == "OCEAN").cast(T.IntegerType()))
        .withColumn("is_air_freight",     (F.col("transport_mode") == "AIR_PARCEL").cast(T.IntegerType()))
        .withColumn("is_truck_freight",   (F.col("transport_mode") == "TRUCK").cast(T.IntegerType()))
        .withColumn("log_cargo_value",    F.log1p(F.col("cargo_value_usd")))
    )

    # ── 7. Join supplier risk context ─────────────────────────────────────
    sup_slim = supplier_features.select(
        "supplier_id",
        F.col("disruption_rate_90d").alias("sup_disruption_rate_90d"),
        F.col("on_time_rate_90d").alias("sup_on_time_rate_90d"),
        F.col("geo_risk_score").alias("sup_geo_risk"),
        F.col("financial_risk_score").alias("sup_financial_risk"),
        F.col("network_pagerank").alias("sup_pagerank"),
    )
    base = base.join(sup_slim, on="supplier_id", how="left")

    # ── 8. Fill remaining nulls ────────────────────────────────────────────
    base = base.fillna({
        "transit_route_risk": 0.25,
        "route_historical_exception_rate": 0.25,
        "route_median_delay_days": 2.0,
        "route_delay_std": 3.0,
        "route_total_shipments": 0,
        "sup_disruption_rate_90d": 0.1,
        "sup_on_time_rate_90d": 0.9,
        "sup_geo_risk": 0.3,
        "sup_financial_risk": 0.3,
        "sup_pagerank": 0.01,
        "month_sin": 0.0, "month_cos": 1.0,
        "is_q4_crunch": 0, "is_weekend_departure": 0,
        "log_cargo_value": 0.0,
    })

    # ── 9. Audit ──────────────────────────────────────────────────────────
    base = (
        base
        .withColumn("_enriched_at",  F.current_timestamp())
        .withColumn("_silver_layer", F.lit(True))
    )

    total = base.count()
    pos_rate = base.filter(F.col("disruption_label") == 1).count() / max(total, 1)
    print(f"[Silver/shipment_features] {total:,} rows | "
          f"Disruption label rate: {pos_rate:.2%}")

    return base
