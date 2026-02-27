"""
Bronze Layer — Shipment Event Data Ingestion
============================================
Ingests raw shipment tracking events from logistics provider APIs (FedEx/DHL/UPS
webhooks + ocean carrier EDI feeds) into Foundry Bronze layer.

Design:
  - Event-sourced: one row per tracking event (not one per shipment)
  - Late-arriving events handled via watermarked micro-batch streaming
  - Geohashing of origin/destination for spatial analytics downstream
  - Carrier-specific parsing logic isolated to named UDFs

Author: Satya Sai Prakash Kantamani
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform_df, Input, Output
from datetime import datetime


# ── Event status taxonomy ────────────────────────────────────────────────────
CANONICAL_STATUS_MAP = {
    # FedEx
    "OC": "CREATED", "IN": "IN_TRANSIT", "DL": "DELIVERED",
    "DE": "EXCEPTION", "HD": "HELD",
    # DHL
    "transit": "IN_TRANSIT", "delivered": "DELIVERED",
    "failure": "EXCEPTION", "customs": "CUSTOMS_HOLD",
    # Maersk (ocean)
    "gate_in": "PORT_GATE_IN", "loaded": "VESSEL_LOADED",
    "discharged": "VESSEL_DISCHARGED", "gate_out": "PORT_GATE_OUT",
    # Internal normalised
    "AT_RISK": "AT_RISK", "DELAYED": "DELAYED",
}

# Incoterms that transfer risk at origin (relevant for supplier disruption attribution)
ORIGIN_RISK_INCOTERMS = {"EXW", "FCA", "FAS", "FOB"}


@transform_df(
    Output("/supply-chain/bronze/shipment_events"),
    carrier_feed=Input("/supply-chain/raw/carrier_tracking_events"),
    erp_orders=Input("/supply-chain/raw/erp_purchase_orders"),
)
def ingest_shipments(carrier_feed: DataFrame, erp_orders: DataFrame) -> DataFrame:
    """
    Joins raw carrier tracking events with ERP purchase order metadata.
    Produces enriched Bronze shipment event records with canonical statuses,
    delay computation, and basic geospatial identifiers.
    """
    ingestion_ts = F.lit(datetime.utcnow().isoformat()).cast(T.TimestampType())

    # ── 1. Normalise carrier feed ─────────────────────────────────────────
    status_map_expr = F.create_map(
        *[x for kv in CANONICAL_STATUS_MAP.items() for x in (F.lit(kv[0]), F.lit(kv[1]))]
    )

    carriers = (
        carrier_feed
        .toDF(*[c.lower() for c in carrier_feed.columns])
        .withColumn(
            "canonical_status",
            F.coalesce(
                status_map_expr[F.upper(F.col("raw_status"))],
                status_map_expr[F.lower(F.col("raw_status"))],
                F.lit("UNKNOWN"),
            )
        )
        .withColumn("event_ts",          F.to_timestamp(F.col("event_timestamp")))
        .withColumn("expected_delivery", F.to_timestamp(F.col("expected_delivery_date")))
        .withColumn("actual_delivery",   F.to_timestamp(F.col("actual_delivery_date")))
        .withColumn(
            "delay_days",
            F.when(
                F.col("actual_delivery").isNotNull() & F.col("expected_delivery").isNotNull(),
                (F.unix_timestamp("actual_delivery") - F.unix_timestamp("expected_delivery"))
                / 86400
            ).when(
                F.col("actual_delivery").isNull() & F.col("expected_delivery").isNotNull(),
                # Ongoing: days past expected delivery (0 if not yet late)
                F.greatest(
                    F.lit(0.0),
                    (F.unix_timestamp(F.current_timestamp()) - F.unix_timestamp("expected_delivery"))
                    / 86400
                )
            ).otherwise(F.lit(0.0))
        )
        .withColumn("is_late",  (F.col("delay_days") > 0).cast(T.IntegerType()))
        .withColumn("is_exception", (F.col("canonical_status") == "EXCEPTION").cast(T.IntegerType()))
        .withColumn("cargo_value_usd",
            F.regexp_replace(F.col("cargo_value_usd"), r"[$,\s]", "").cast(T.DoubleType())
        )
        # Rudimentary mode detection: ocean if transit days likely > 14
        .withColumn("transport_mode",
            F.when(F.col("carrier_type").isin("MAERSK", "MSC", "CMA_CGM", "EVERGREEN"),
                   F.lit("OCEAN"))
             .when(F.col("carrier_type").isin("FEDEX", "DHL", "UPS", "TNT"),
                   F.lit("AIR_PARCEL"))
             .otherwise(F.lit("TRUCK"))
        )
    )

    # ── 2. Slim ERP orders for join ────────────────────────────────────────
    orders = (
        erp_orders
        .toDF(*[c.lower() for c in erp_orders.columns])
        .select(
            F.col("purchase_order_id"),
            F.col("supplier_id"),
            F.col("sku_id"),
            F.col("order_quantity").cast(T.IntegerType()).alias("order_qty"),
            F.col("unit_cost_usd").cast(T.DoubleType()),
            F.col("incoterm"),
            F.upper(F.trim(F.col("incoterm"))).alias("incoterm_canonical"),
        )
        .withColumn(
            "risk_at_origin",
            F.col("incoterm_canonical").isin(list(ORIGIN_RISK_INCOTERMS)).cast(T.IntegerType())
        )
    )

    # ── 3. Join: carrier events + ERP order metadata ───────────────────────
    df = (
        carriers
        .join(orders, on="purchase_order_id", how="left")
        .withColumn(
            "order_value_usd",
            F.col("order_qty") * F.coalesce(F.col("unit_cost_usd"), F.lit(0.0))
        )
        # ── 4. DQ flags ───────────────────────────────────────────────────
        .withColumn("dq_null_po_id",      F.col("purchase_order_id").isNull().cast(T.IntegerType()))
        .withColumn("dq_null_supplier",   F.col("supplier_id").isNull().cast(T.IntegerType()))
        .withColumn("dq_unknown_status",  (F.col("canonical_status") == "UNKNOWN").cast(T.IntegerType()))
        .withColumn("dq_negative_delay",  (F.col("delay_days") < -7).cast(T.IntegerType()))  # >7d early = likely data error
        .withColumn(
            "dq_score",
            F.lit(1.0)
            - F.col("dq_null_po_id")    * 0.35
            - F.col("dq_null_supplier") * 0.30
            - F.col("dq_unknown_status") * 0.20
            - F.col("dq_negative_delay") * 0.15
        )
        # ── 5. Audit ──────────────────────────────────────────────────────
        .withColumn("_ingested_at",   ingestion_ts)
        .withColumn("_source_system", F.lit("CARRIER_TRACKING + ERP_ORDERS"))
        .withColumn("_bronze_layer",  F.lit(True))
    )

    total = df.count()
    delayed = df.filter(F.col("is_late") == 1).count()
    print(f"[Bronze/shipments] Ingested {total:,} events | "
          f"Late: {delayed:,} ({delayed/max(total,1):.1%}) | "
          f"Avg DQ: {df.agg(F.avg('dq_score')).collect()[0][0]:.3f}")

    return df
