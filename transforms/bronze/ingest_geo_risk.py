"""
Bronze Layer — Geopolitical & Macro Risk Signal Ingestion
=========================================================
Ingests external risk intelligence feeds into Foundry Bronze layer:
  - GDELT 2.0 event stream (conflict intensity, protest indices)
  - World Bank WGI (Worldwide Governance Indicators) — annual
  - UN Comtrade bilateral trade data — monthly aggregates
  - Commodity price indices (LME, CME) — daily OHLCV
  - Port congestion indices — daily (Sea-Intelligence, Marine Traffic API)

All feeds are aligned to a canonical [country_code, date] grain for
downstream join with supplier master data.

Author: Satya Sai Prakash Kantamani
"""

from pyspark.sql import DataFrame, Window
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform_df, Input, Output
from datetime import datetime


# ── Severity tiers for composite geo risk index ──────────────────────────────
# Weights derived from logistic regression on historical disruption labels
RISK_COMPONENT_WEIGHTS = {
    "political_instability":    0.30,
    "trade_conflict_intensity": 0.25,
    "conflict_proximity_score": 0.20,
    "infrastructure_quality":   0.15,   # inverted: lower infra = higher risk
    "natural_disaster_freq":    0.10,
}


@transform_df(
    Output("/supply-chain/bronze/geo_risk_signals"),
    gdelt=Input("/supply-chain/raw/gdelt_country_daily"),
    wgi=Input("/supply-chain/raw/world_bank_wgi_annual"),
    comtrade=Input("/supply-chain/raw/un_comtrade_bilateral_monthly"),
    commodities=Input("/supply-chain/raw/commodity_price_indices_daily"),
    port_congestion=Input("/supply-chain/raw/port_congestion_daily"),
)
def ingest_geo_risk(
    gdelt: DataFrame,
    wgi: DataFrame,
    comtrade: DataFrame,
    commodities: DataFrame,
    port_congestion: DataFrame,
) -> DataFrame:
    """
    Joins 5 external risk feeds at (country_code, date) grain.
    Computes a composite geo_risk_score ∈ [0, 1] as weighted linear combination
    of sub-indices. Higher score = higher risk.
    """
    ingestion_ts = F.lit(datetime.utcnow().isoformat()).cast(T.TimestampType())

    # ── 1. GDELT: conflict intensity aggregated per country per day ────────
    gdelt_agg = (
        gdelt
        .toDF(*[c.lower() for c in gdelt.columns])
        .withColumn("event_date", F.to_date(F.col("event_date_str"), "yyyyMMdd"))
        .groupBy("country_code", "event_date")
        .agg(
            F.avg("goldstein_scale").alias("goldstein_avg"),           # -10 to +10 stability
            F.sum("num_articles").alias("media_coverage_volume"),
            F.avg("avg_tone").alias("avg_media_tone"),
            F.count("*").alias("event_count_daily"),
        )
        # Normalise Goldstein to [0,1] risk (lower Goldstein = higher risk)
        .withColumn(
            "conflict_proximity_score",
            F.greatest(F.lit(0.0),
                F.least(F.lit(1.0),
                    (F.lit(-1.0) * F.col("goldstein_avg") + F.lit(10.0)) / F.lit(20.0)
                )
            )
        )
    )

    # ── 2. WGI: annual governance indicators ──────────────────────────────
    # Forward-fill annual values to daily grain
    wgi_clean = (
        wgi
        .toDF(*[c.lower() for c in wgi.columns])
        .withColumn("year",   F.col("year").cast(T.IntegerType()))
        # political_stability_wgi ∈ [-2.5, 2.5]; normalise to [0,1] (inverted: high WGI = low risk)
        .withColumn(
            "political_instability",
            F.greatest(F.lit(0.0),
                F.least(F.lit(1.0),
                    (F.lit(2.5) - F.col("political_stability_wgi").cast(T.DoubleType()))
                    / F.lit(5.0)
                )
            )
        )
        .select("country_code", "year", "political_instability",
                "rule_of_law_wgi", "regulatory_quality_wgi")
    )

    # ── 3. Comtrade: bilateral trade conflict intensity ────────────────────
    comtrade_agg = (
        comtrade
        .toDF(*[c.lower() for c in comtrade.columns])
        .withColumn("trade_month", F.to_date(F.col("period"), "yyyy-MM"))
        .groupBy("reporter_country_code", "trade_month")
        .agg(
            F.avg("trade_deficit_ratio").alias("trade_deficit_ratio"),
            F.sum("tariff_dispute_flag").alias("active_tariff_disputes"),
            F.avg("import_concentration_hhi").alias("import_hhi"),
        )
        .withColumn(
            "trade_conflict_intensity",
            F.least(F.lit(1.0),
                (F.col("active_tariff_disputes") * F.lit(0.1))
                + F.col("trade_deficit_ratio").cast(T.DoubleType()) * F.lit(0.5)
            )
        )
        .withColumnRenamed("reporter_country_code", "country_code")
    )

    # ── 4. Commodity volatility: 30-day rolling std / mean ────────────────
    comm_window = Window.partitionBy("commodity_code").orderBy("price_date") \
                        .rowsBetween(-29, 0)
    commodity_vol = (
        commodities
        .toDF(*[c.lower() for c in commodities.columns])
        .withColumn("price_date", F.to_date(F.col("date_str")))
        .withColumn("close_price", F.col("close_price").cast(T.DoubleType()))
        .withColumn("rolling_std",  F.stddev("close_price").over(comm_window))
        .withColumn("rolling_mean", F.avg("close_price").over(comm_window))
        .withColumn(
            "commodity_price_volatility_30d",
            F.when(F.col("rolling_mean") > 0,
                   F.col("rolling_std") / F.col("rolling_mean")).otherwise(F.lit(0.0))
        )
        .groupBy("price_date")
        .agg(F.avg("commodity_price_volatility_30d").alias("commodity_price_volatility_30d"))
    )

    # ── 5. Port congestion: normalise 0–1 ─────────────────────────────────
    port_agg = (
        port_congestion
        .toDF(*[c.lower() for c in port_congestion.columns])
        .withColumn("congestion_date", F.to_date(F.col("date_str")))
        .withColumn("congestion_index_raw", F.col("wait_time_hours").cast(T.DoubleType()))
        .groupBy("country_code", "congestion_date")
        .agg(F.avg("congestion_index_raw").alias("avg_wait_hours"))
        # Normalise: cap at 168h (1 week)
        .withColumn(
            "port_congestion_score",
            F.least(F.lit(1.0), F.col("avg_wait_hours") / F.lit(168.0))
        )
        .withColumnRenamed("congestion_date", "event_date")
    )

    # ── 6. Combine all signals ────────────────────────────────────────────
    df = (
        gdelt_agg
        .join(port_agg, on=["country_code", "event_date"], how="left")
        # Join WGI by year (approximate to daily)
        .withColumn("year_col", F.year(F.col("event_date")))
        .join(wgi_clean, on=[
            gdelt_agg["country_code"] == wgi_clean["country_code"],
            F.col("year_col") == wgi_clean["year"],
        ], how="left")
        .drop(wgi_clean["country_code"]).drop("year", "year_col")
        # Join comtrade by month
        .withColumn("month_col", F.date_trunc("month", F.col("event_date")).cast(T.DateType()))
        .join(comtrade_agg.withColumnRenamed("trade_month", "month_col"),
              on=["country_code", "month_col"], how="left")
        .drop("month_col")
        # Join commodity volatility (global, no country join)
        .join(commodity_vol, on=gdelt_agg["event_date"] == commodity_vol["price_date"], how="left")
        .drop("price_date")
        # Fill nulls for optional components
        .fillna({
            "political_instability":         0.5,
            "trade_conflict_intensity":       0.0,
            "port_congestion_score":          0.0,
            "commodity_price_volatility_30d": 0.0,
            "conflict_proximity_score":       0.0,
        })
        # ── 7. Composite geo risk index ───────────────────────────────────
        .withColumn(
            "geo_risk_score",
            F.least(F.lit(1.0), F.greatest(F.lit(0.0),
                F.col("political_instability")       * F.lit(RISK_COMPONENT_WEIGHTS["political_instability"])
                + F.col("trade_conflict_intensity")  * F.lit(RISK_COMPONENT_WEIGHTS["trade_conflict_intensity"])
                + F.col("conflict_proximity_score")  * F.lit(RISK_COMPONENT_WEIGHTS["conflict_proximity_score"])
                + F.col("port_congestion_score")     * F.lit(RISK_COMPONENT_WEIGHTS["infrastructure_quality"])
                + F.col("commodity_price_volatility_30d") * F.lit(RISK_COMPONENT_WEIGHTS["natural_disaster_freq"])
            ))
        )
        # ── 8. Risk tier label ────────────────────────────────────────────
        .withColumn(
            "geo_risk_tier",
            F.when(F.col("geo_risk_score") >= 0.75, F.lit("CRITICAL"))
             .when(F.col("geo_risk_score") >= 0.50, F.lit("HIGH"))
             .when(F.col("geo_risk_score") >= 0.25, F.lit("MEDIUM"))
             .otherwise(F.lit("LOW"))
        )
        # ── 9. Audit ──────────────────────────────────────────────────────
        .withColumn("_ingested_at",   ingestion_ts)
        .withColumn("_source_system", F.lit("GDELT+WGI+COMTRADE+LME+PORT_CONGESTION"))
        .withColumn("_bronze_layer",  F.lit(True))
    )

    total = df.count()
    critical = df.filter(F.col("geo_risk_tier") == "CRITICAL").count()
    print(f"[Bronze/geo_risk] Ingested {total:,} country-day records | "
          f"Critical tier: {critical:,} ({critical/max(total,1):.1%})")

    return df
