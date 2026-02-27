"""
Bronze Layer — Supplier Master Data Ingestion
=============================================
Ingests raw supplier master data from upstream ERP / SRM systems into the
Foundry Bronze layer with schema enforcement and data quality checks.

Design:
  - Schema-on-read with explicit casting and null handling
  - Row-level DQ metrics stored alongside data (never silently dropped)
  - Idempotent: safe to re-run on same input partition
  - Audit columns appended for full lineage tracing

Foundry Transform: runs daily at 06:00 UTC on incremental input datasets.

Author: Satya Sai Prakash Kantamani
"""

from pyspark.sql import DataFrame
import pyspark.sql.functions as F
import pyspark.sql.types as T
from transforms.api import transform_df, Input, Output
from datetime import datetime


# ── Expected raw schema ──────────────────────────────────────────────────────
RAW_SCHEMA = T.StructType([
    T.StructField("SUPPLIER_ID",          T.StringType(),  True),
    T.StructField("SUPPLIER_NAME",        T.StringType(),  True),
    T.StructField("COUNTRY_CODE",         T.StringType(),  True),
    T.StructField("REGION",               T.StringType(),  True),
    T.StructField("PRIMARY_CATEGORY",     T.StringType(),  True),
    T.StructField("ANNUAL_SPEND_USD",     T.StringType(),  True),   # raw: may have commas / currency
    T.StructField("CONTRACT_TYPE",        T.StringType(),  True),
    T.StructField("PAYMENT_TERMS_DAYS",   T.StringType(),  True),
    T.StructField("AVG_LEAD_TIME_DAYS",   T.StringType(),  True),
    T.StructField("SOLE_SOURCE_FLAG",     T.StringType(),  True),
    T.StructField("TIER",                 T.StringType(),  True),    # 1 / 2 / 3
    T.StructField("ERP_LAST_UPDATED",     T.StringType(),  True),
    T.StructField("FINANCIAL_SCORE_EXT",  T.StringType(),  True),    # Dun & Bradstreet
    T.StructField("CONTACT_EMAIL",        T.StringType(),  True),
])

# DQ thresholds
DQ_NULL_THRESHOLD      = 0.05   # flag if >5% of rows have null in critical columns
CRITICAL_COLUMNS       = ["SUPPLIER_ID", "SUPPLIER_NAME", "COUNTRY_CODE", "ANNUAL_SPEND_USD"]


@transform_df(
    Output("/supply-chain/bronze/suppliers"),
    raw=Input("/supply-chain/raw/erp_supplier_master"),
)
def ingest_suppliers(raw: DataFrame) -> DataFrame:
    """
    Reads raw ERP supplier master export and produces a cleaned,
    typed Bronze layer dataset with DQ flags.
    """
    ingestion_ts = F.lit(datetime.utcnow().isoformat()).cast(T.TimestampType())

    df = (
        raw
        # ── 1. Standardise column names to snake_case ──────────────────────
        .toDF(*[c.lower() for c in raw.columns])

        # ── 2. Type casting with safe fallbacks ───────────────────────────
        .withColumn(
            "annual_spend_usd",
            F.regexp_replace(F.col("annual_spend_usd"), r"[$,\s]", "").cast(T.DoubleType())
        )
        .withColumn("avg_lead_time_days",   F.col("avg_lead_time_days").cast(T.IntegerType()))
        .withColumn("payment_terms_days",   F.col("payment_terms_days").cast(T.IntegerType()))
        .withColumn("financial_score_ext",  F.col("financial_score_ext").cast(T.DoubleType()))
        .withColumn("sole_source_flag",
            F.when(F.upper(F.col("sole_source_flag")).isin("Y", "YES", "1", "TRUE"), F.lit(1))
             .otherwise(F.lit(0))
             .cast(T.IntegerType())
        )
        .withColumn("tier",                 F.col("tier").cast(T.IntegerType()))
        .withColumn(
            "erp_last_updated",
            F.to_timestamp(F.col("erp_last_updated"), "yyyy-MM-dd HH:mm:ss")
        )

        # ── 3. Normalise text fields ───────────────────────────────────────
        .withColumn("supplier_name",        F.initcap(F.trim(F.col("supplier_name"))))
        .withColumn("country_code",         F.upper(F.trim(F.col("country_code"))))
        .withColumn("primary_category",     F.upper(F.trim(F.col("primary_category"))))
        .withColumn("contract_type",        F.upper(F.trim(F.col("contract_type"))))
        .withColumn("region",               F.upper(F.trim(F.col("region"))))

        # ── 4. Data quality flags ──────────────────────────────────────────
        .withColumn("dq_null_supplier_id",    F.col("supplier_id").isNull().cast(T.IntegerType()))
        .withColumn("dq_null_name",           F.col("supplier_name").isNull().cast(T.IntegerType()))
        .withColumn("dq_null_country",        F.col("country_code").isNull().cast(T.IntegerType()))
        .withColumn("dq_negative_spend",
            (F.col("annual_spend_usd") < 0).cast(T.IntegerType())
        )
        .withColumn("dq_invalid_lead_time",
            (F.col("avg_lead_time_days") <= 0).cast(T.IntegerType())
        )
        .withColumn(
            "dq_score",   # 1.0 = perfect, lower = has quality issues
            F.lit(1.0)
            - (F.col("dq_null_supplier_id") * 0.40)
            - (F.col("dq_null_name")        * 0.20)
            - (F.col("dq_null_country")     * 0.20)
            - (F.col("dq_negative_spend")   * 0.10)
            - (F.col("dq_invalid_lead_time") * 0.10)
        )

        # ── 5. Audit columns ───────────────────────────────────────────────
        .withColumn("_ingested_at",   ingestion_ts)
        .withColumn("_source_system", F.lit("ERP_SUPPLIER_MASTER"))
        .withColumn("_bronze_layer",  F.lit(True))
    )

    # ── 6. Aggregate DQ report (logged, not blocking) ─────────────────────
    total_rows = df.count()
    for col_name in CRITICAL_COLUMNS:
        null_rate = df.filter(F.col(col_name.lower()).isNull()).count() / max(total_rows, 1)
        if null_rate > DQ_NULL_THRESHOLD:
            print(f"[DQ WARNING] {col_name}: {null_rate:.1%} null rate exceeds "
                  f"{DQ_NULL_THRESHOLD:.0%} threshold")

    print(f"[Bronze/suppliers] Ingested {total_rows:,} rows | "
          f"Avg DQ score: {df.agg(F.avg('dq_score')).collect()[0][0]:.3f}")

    return df
