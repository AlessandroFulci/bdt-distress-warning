"""
silver.build_silver
===================
Silver layer – EDGAR Financial Distress Early Warning System.

Reads raw company-facts JSON from the MinIO **bronze** bucket using
PySpark (local mode — no cluster needed), extracts and standardises
key financial line items, and writes clean Parquet (partitioned by
fiscal_year) to the **silver** bucket.

Why PySpark over pandas
-----------------------
* Handles the full dataset (700+ companies, 40k+ rows) in parallel
* Spark's window functions make dedup and ranking clean and scalable
* Parquet partitioned writes are native to Spark
* Same code would run on a real cluster with zero changes

Pipeline
--------
1. Start Spark in local mode, configured to talk to MinIO via S3A
2. Stream bronze JSON via minio SDK (Spark reads the parsed records)
3. Extract financial concepts via tag-alias map
4. Explode → one row per (CIK, period, field, value)
5. Pivot  → one row per (CIK, period)
6. Deduplicate via Spark window: FY > Q4 > Q3 > Q2 > Q1, then latest filed
7. Compute derived fields: working_capital, ebit
8. Write Parquet to silver bucket, partitioned by fiscal_year

Output schema (one row per company per reporting period)
---------------------------------------------------------
cik, company_name, period_end, fiscal_year, fiscal_period,
form_type, filed_date, assets_total, assets_current,
liabilities_total, liabilities_current, equity, retained_earnings,
net_income, revenue, operating_income, interest_expense,
cash, long_term_debt, ebit, working_capital

Usage
-----
python -m silver.build_silver --prefix edgar_company_facts/
python -m silver.build_silver --prefix edgar_company_facts/ --dry-run
python -m silver.build_silver --help

Environment variables
---------------------
MINIO_ENDPOINT       (default: localhost:9000)
MINIO_ACCESS_KEY     (default: minioadmin)
MINIO_SECRET_KEY     (default: minioadmin)
MINIO_SECURE         (default: false)
MINIO_BRONZE_BUCKET  (default: bronze)
MINIO_SILVER_BUCKET  (default: silver)
BRONZE_PREFIX        (default: "")
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
load_dotenv()

from minio import Minio
from minio.error import S3Error

from pyspark.sql import SparkSession, Row
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    StructType, StructField,
    LongType, StringType, DateType, DoubleType, IntegerType
)

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

RELEVANT_FORMS: set[str] = {"10-K", "10-Q", "20-F", "40-F", "6-K"}

OUTPUT_COLUMNS: list[str] = [
    "cik", "company_name", "period_end", "fiscal_year", "fiscal_period",
    "form_type", "filed_date",
    "assets_total", "assets_current",
    "liabilities_total", "liabilities_current",
    "equity", "retained_earnings",
    "net_income", "revenue", "operating_income", "interest_expense",
    "cash", "long_term_debt",
    "ebit", "working_capital",
]

# Maps canonical field → ordered list of GAAP/IFRS XBRL tags (priority order)
TAG_ALIASES: dict[str, list[str]] = {
    "assets_total": [
        "Assets",
    ],
    "assets_current": [
        "AssetsCurrent",
    ],
    "liabilities_total": [
        "Liabilities",
    ],
    "liabilities_current": [
        "LiabilitiesCurrent",
    ],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        "PartnersCapital",
        "MembersEquity",
    ],
    "retained_earnings": [
        "RetainedEarningsAccumulatedDeficit",
        "RetainedEarnings",
    ],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
        "ProfitLoss",
        "IncomeLossFromContinuingOperations",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
        "RevenuesNetOfInterestExpense",
    ],
    "operating_income": [
        "OperatingIncomeLoss",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
        "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
    ],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
        "InterestExpenseRelatedParty",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        "Cash",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
        "LongTermDebtAndCapitalLeaseObligations",
    ],
}

# Priority for fiscal_period dedup: lower = preferred
PERIOD_RANK: dict[str, int] = {"FY": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}


# =============================================================================
# Parsing  (pure Python — runs inside Spark mapPartitions)
# =============================================================================

def _parse_date(val: Any) -> Optional[str]:
    """Parse ISO date string → 'YYYY-MM-DD' string, or None."""
    if not val:
        return None
    try:
        date.fromisoformat(str(val))   # validate
        return str(val)[:10]
    except (ValueError, TypeError):
        return None


def extract_rows_from_blob(cik: int, company_name: str, raw: dict) -> list[dict]:
    """
    Parse one company-facts JSON blob → list of flat dicts.

    Each dict represents one (company, period, field, value) tuple.
    This is the unit Spark will distribute and aggregate.

    Returns an empty list if no extractable data is found.
    """
    facts_root: dict = raw.get("facts", {})
    namespaces: dict = {
        **facts_root.get("ifrs-full", {}),
        **facts_root.get("us-gaap", {}),
    }

    if not namespaces:
        return []

    rows: list[dict] = []

    for canonical_field, tags in TAG_ALIASES.items():
        found = False
        for tag in tags:
            if found:
                break
            concept = namespaces.get(tag)
            if not concept:
                continue
            for entry in concept.get("units", {}).get("USD", []):
                form = entry.get("form", "")
                if not any(form.startswith(f) for f in RELEVANT_FORMS):
                    continue
                period_end = _parse_date(entry.get("end"))
                if period_end is None:
                    continue
                val = entry.get("val")
                if val is None:
                    continue
                fy = entry.get("fy")
                rows.append({
                    "cik":           int(cik),
                    "company_name":  str(company_name),
                    "period_end":    period_end,
                    "fiscal_year":   int(period_end[:4]),
                    "fiscal_period": str(entry.get("fp", "")),
                    "form_type":     str(form),
                    "filed_date":    _parse_date(entry.get("filed")) or "",
                    "field":         canonical_field,
                    "value":         float(val),
                })
            if any(r["field"] == canonical_field for r in rows):
                found = True

    return rows


# =============================================================================
# MinIO  (download bronze JSON — same as before)
# =============================================================================

def _get_minio_client(
    endpoint: str, access_key: str, secret_key: str, secure: bool
) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _stream_bronze(
    client: Minio, bucket: str, prefix: str = ""
) -> Iterator[tuple[str, dict]]:
    """Yield (object_name, parsed_json) for every .json in the bronze bucket."""
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    for obj in objects:
        name = obj.object_name
        if not name.endswith(".json") or name.endswith(".meta.json"):
            continue
        try:
            resp = client.get_object(bucket, name)
            try:
                raw = json.loads(resp.read())
            finally:
                resp.close()
                resp.release_conn()
            yield name, raw
        except (S3Error, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            log.warning("SKIP %s — %s", name, exc)


# =============================================================================
# Spark session
# =============================================================================

def _build_spark(
    endpoint: str,
    access_key: str,
    secret_key: str,
    secure: bool,
    app_name: str = "silver-layer",
) -> SparkSession:
    """
    Create a Spark session in local mode configured for MinIO S3A access.

    local[*] means: use all available CPU cores on this machine.
    No cluster required — Spark runs entirely inside your Python process.
    """
    protocol = "https" if secure else "http"

    spark = (
        SparkSession.builder
        .master("local[*]")
        .appName(app_name)
        # S3A connector settings for MinIO
        .config("spark.hadoop.fs.s3a.endpoint",            f"{protocol}://{endpoint}")
        .config("spark.hadoop.fs.s3a.access.key",          access_key)
        .config("spark.hadoop.fs.s3a.secret.key",          secret_key)
        .config("spark.hadoop.fs.s3a.path.style.access",   "true")
        .config("spark.hadoop.fs.s3a.impl",
                "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider",
                "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        .config("spark.hadoop.fs.s3a.buffer.dir",          "C:/tmp/s3a")
        .config("spark.hadoop.fs.s3a.fast.upload",         "true")
        .config("spark.hadoop.fs.s3a.fast.upload.buffer",  "array")
        # Parquet settings
        .config("spark.sql.parquet.compression.codec",     "snappy")
        .config("spark.sql.sources.partitionOverwriteMode","dynamic")
        # Reduce console noise
        .config("spark.ui.showConsoleProgress",            "false")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")
    return spark


# =============================================================================
# Spark schema for the raw extracted rows
# =============================================================================

RAW_SCHEMA = StructType([
    StructField("cik",           LongType(),   nullable=False),
    StructField("company_name",  StringType(), nullable=False),
    StructField("period_end",    StringType(), nullable=False),
    StructField("fiscal_year",   IntegerType(),nullable=False),
    StructField("fiscal_period", StringType(), nullable=True),
    StructField("form_type",     StringType(), nullable=True),
    StructField("filed_date",    StringType(), nullable=True),
    StructField("field",         StringType(), nullable=False),
    StructField("value",         DoubleType(), nullable=False),
])


# =============================================================================
# Pipeline
# =============================================================================

def run(
    endpoint:      str  = os.getenv("MINIO_ENDPOINT",      "localhost:9000"),
    access_key:    str  = os.getenv("MINIO_ACCESS_KEY",    "minioadmin"),
    secret_key:    str  = os.getenv("MINIO_SECRET_KEY",    "minioadmin"),
    secure:        bool = os.getenv("MINIO_SECURE",        "false") == "true",
    bronze_bucket: str  = os.getenv("MINIO_BRONZE_BUCKET", "bronze"),
    silver_bucket: str  = os.getenv("MINIO_SILVER_BUCKET", "silver"),
    prefix:        str  = os.getenv("BRONZE_PREFIX",       ""),
    dry_run:       bool = False,
) -> None:
    """
    Execute the full PySpark silver pipeline.

    Steps
    -----
    1. Download + parse all bronze JSON via minio SDK (Python)
    2. Create a Spark DataFrame from the extracted rows
    3. Pivot  → one row per (cik, period_end)
    4. Dedup  → window function: prefer FY, then most recently filed
    5. Derive → working_capital, ebit
    6. Write  → Parquet to MinIO silver bucket via S3A
    """
    log.info("=" * 60)
    log.info("Silver layer (PySpark) — start")
    log.info("  MinIO  : %s  (secure=%s)", endpoint, secure)
    log.info("  Bronze : %s/%s", bronze_bucket, prefix or "*")
    log.info("  Silver : %s", silver_bucket)
    log.info("  Dry run: %s", dry_run)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Parse bronze JSON → flat Python list (driver only)
    # ------------------------------------------------------------------
    minio_client = _get_minio_client(endpoint, access_key, secret_key, secure)

    all_rows: list[dict] = []
    n_objects = n_errors = 0

    for obj_name, raw in _stream_bronze(minio_client, bronze_bucket, prefix):
        n_objects += 1
        cik          = raw.get("cik", 0)
        company_name = raw.get("entityName", "Unknown")
        log.info("[%4d] %s  (CIK %s, %s)", n_objects, obj_name, cik, company_name)
        try:
            rows = extract_rows_from_blob(cik, company_name, raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("  SKIP — extraction error: %s", exc)
            n_errors += 1
            continue
        log.info("  → %d field-value entries", len(rows))
        all_rows.extend(rows)

    log.info("-" * 60)
    log.info("Objects processed  : %d  (errors: %d)", n_objects, n_errors)
    log.info("Raw field entries  : %d", len(all_rows))

    if not all_rows:
        log.error("No data extracted. Nothing to write.")
        return

    # ------------------------------------------------------------------
    # 2. Create Spark session + DataFrame
    # ------------------------------------------------------------------
    spark = _build_spark(endpoint, access_key, secret_key, secure)
    log.info("Spark version: %s", spark.version)

    df = spark.createDataFrame(
        [Row(**r) for r in all_rows],
        schema=RAW_SCHEMA,
    )

    # Cast period_end and filed_date to proper date columns
    df = (
        df
        .withColumn("period_end",  F.to_date("period_end",  "yyyy-MM-dd"))
        .withColumn("filed_date",  F.to_date("filed_date",  "yyyy-MM-dd"))
    )

    # ------------------------------------------------------------------
    # 3. Pivot → one row per (cik, period_end)
    #    For each (cik, period_end, field) keep the most recently filed value
    # ------------------------------------------------------------------
    # Window: per (cik, period_end, field), latest filed_date wins
    w_field = Window.partitionBy("cik", "period_end", "field") \
                    .orderBy(F.col("filed_date").desc_nulls_last())

    df = (
        df
        .withColumn("rn", F.row_number().over(w_field))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    # Pivot fields into columns
    pivot_df = (
        df
        .groupBy("cik", "company_name", "period_end",
                 "fiscal_year", "fiscal_period", "form_type", "filed_date")
        .pivot("field", list(TAG_ALIASES.keys()))
        .agg(F.last("value", ignorenulls=True))
    )

    # Ensure all canonical columns exist
    for col in TAG_ALIASES:
        if col not in pivot_df.columns:
            pivot_df = pivot_df.withColumn(col, F.lit(None).cast(DoubleType()))

    # ------------------------------------------------------------------
    # 4. Deduplicate: same (cik, period_end) → prefer FY, then latest filed
    # ------------------------------------------------------------------
    period_rank_map = F.create_map(
        *[item for pair in
          [(F.lit(k), F.lit(v)) for k, v in PERIOD_RANK.items()]
          for item in pair]
    )

    pivot_df = pivot_df.withColumn(
        "_period_rank",
        F.coalesce(period_rank_map[F.col("fiscal_period")], F.lit(9))
    )

    w_dedup = (
        Window.partitionBy("cik", "period_end")
              .orderBy(
                  F.col("_period_rank").asc(),
                  F.col("filed_date").desc_nulls_last(),
                  F.col("form_type").asc()
              )
    )

    pivot_df = (
        pivot_df
        .withColumn("_rn", F.row_number().over(w_dedup))
        .filter(F.col("_rn") == 1)
        .drop("_rn", "_period_rank")
    )

    # ------------------------------------------------------------------
    # 5. Derived fields
    # ------------------------------------------------------------------

    # working_capital = assets_current - liabilities_current  (Altman X1 numerator)
    pivot_df = pivot_df.withColumn(
        "working_capital",
        F.col("assets_current") - F.col("liabilities_current")
    )

    # ebit: prefer operating_income; fall back to net_income + interest_expense
    pivot_df = pivot_df.withColumn(
        "ebit",
        F.when(
            F.col("operating_income").isNotNull(),
            F.col("operating_income")
        ).when(
            F.col("net_income").isNotNull() | F.col("interest_expense").isNotNull(),
            F.coalesce(F.col("net_income"), F.lit(0.0))
            + F.coalesce(F.col("interest_expense"), F.lit(0.0))
        ).otherwise(F.lit(None).cast(DoubleType()))
    )

    # ------------------------------------------------------------------
    # 6. Final column selection + sort
    # ------------------------------------------------------------------
    pivot_df = pivot_df.select(OUTPUT_COLUMNS)

    total_rows = pivot_df.count()
    companies  = pivot_df.select("cik").distinct().count()
    log.info("Total rows after dedup : %d", total_rows)
    log.info("Companies              : %d", companies)

    # ------------------------------------------------------------------
    # 7. Write Parquet to silver bucket (partitioned by fiscal_year)
    # ------------------------------------------------------------------
    if dry_run:
        log.info("[DRY RUN] Skipping MinIO write.")
        pivot_df.show(5, truncate=False)
    else:
        # Convert to pandas and write via minio SDK
        # Avoids S3A Windows filesystem issues entirely
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from minio.error import S3Error

        log.info("Converting to pandas and writing via MinIO SDK...")

        pandas_df = pivot_df.toPandas()
        # Final hard dedup — sort by fiscal_year asc so earlier year wins,
        # then drop duplicates keeping first (earliest fiscal_year per period)
        pandas_df = pandas_df.sort_values(
            ["cik", "period_end", "fiscal_year"], ascending=[True, True, True]
        )
        pandas_df = pandas_df.drop_duplicates(subset=["cik", "period_end"], keep="first")
        log.info("Rows after final dedup: %d", len(pandas_df))

        # Ensure silver bucket exists
        if not minio_client.bucket_exists(silver_bucket):
            minio_client.make_bucket(silver_bucket)
            log.info("Created bucket: %s", silver_bucket)

        # Write one partition per fiscal_year
        for fy in sorted(pandas_df["fiscal_year"].dropna().unique()):
            partition = pandas_df[pandas_df["fiscal_year"] == fy].copy()
            object_name = f"financial_facts/fiscal_year={int(fy)}/data.parquet"
            buf = io.BytesIO()
            pq.write_table(
                pa.Table.from_pandas(partition, preserve_index=False),
                buf, compression="snappy"
            )
            data = buf.getvalue()
            minio_client.put_object(
                silver_bucket, object_name,
                data=io.BytesIO(data), length=len(data),
                content_type="application/octet-stream",
            )
            log.info("Uploaded %s  (%d rows, %d KB)",
                     object_name, len(partition), len(data) // 1024)

        log.info("Write complete.")

    spark.stop()
    log.info("Silver layer (PySpark) — complete")
    log.info("=" * 60)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silver layer (PySpark): parse EDGAR bronze JSON → clean Parquet.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--endpoint",      default=os.getenv("MINIO_ENDPOINT",      "localhost:9000"))
    p.add_argument("--access-key",    default=os.getenv("MINIO_ACCESS_KEY",    "minioadmin"))
    p.add_argument("--secret-key",    default=os.getenv("MINIO_SECRET_KEY",    "minioadmin"))
    p.add_argument("--secure",        action="store_true",
                   default=os.getenv("MINIO_SECURE", "false").lower() == "true")
    p.add_argument("--bronze-bucket", default=os.getenv("MINIO_BRONZE_BUCKET", "bronze"))
    p.add_argument("--silver-bucket", default=os.getenv("MINIO_SILVER_BUCKET", "silver"))
    p.add_argument("--prefix",        default=os.getenv("BRONZE_PREFIX",       ""),
                   help="Object key prefix inside the bronze bucket")
    p.add_argument("--dry-run",       action="store_true",
                   help="Parse and log without writing to MinIO")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    run(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=args.secure,
        bronze_bucket=args.bronze_bucket,
        silver_bucket=args.silver_bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
