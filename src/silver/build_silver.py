"""
silver.build_silver
===================

Pipeline
--------
1. List every .json in the bronze bucket
2. Download + parse each company-facts blob (tag-alias map handles
   the many GAAP variants companies use for the same metric)
3. Pivot → one row per (CIK, period_end)
4. Deduplicate: same period filed twice → keep most-recently-filed values
5. Compute derived fields: working_capital, ebit
6. Write Hive-partitioned Parquet to silver bucket

Output schema
-------------
cik, company_name, period_end, fiscal_year, fiscal_period,
form_type, filed_date, assets_total, assets_current,
liabilities_total, liabilities_current, equity, retained_earnings,
net_income, revenue, operating_income, interest_expense,
cash, long_term_debt, ebit, working_capital

Usage
-----
python -m silver.build_silver                  # env-var / default config
python -m silver.build_silver --dry-run        # parse only, no writes
python -m silver.build_silver --help

Environment variables
---------------------
MINIO_ENDPOINT    (default: localhost:9000)
MINIO_ACCESS_KEY  (default: minioadmin)
MINIO_SECRET_KEY  (default: minioadmin)
MINIO_SECURE      (default: false)
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
from typing import Any, Iterator, Optional

from dotenv import load_dotenv
load_dotenv()

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from minio import Minio
from minio.error import S3Error



from dotenv import load_dotenv
load_dotenv()

log = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

# Forms we extract data from — everything else (8-K, DEF 14A, …) is skipped
RELEVANT_FORMS: set[str] = {"10-K", "10-Q", "20-F", "40-F", "6-K"}

# Final column order — must match what gold/build_gold.py expects
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

# Maps each canonical field → ordered list of GAAP/IFRS XBRL tags.
# The extractor tries each alias in priority order and stops at the first hit,
# because different companies (and the same company across years) use different
# tags for the same concept.
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


# =============================================================================
# Parsing
# =============================================================================

def _parse_date(val: Any) -> Optional[date]:
    """Parse an ISO-8601 string → date, or return None on failure."""
    if not val:
        return None
    try:
        return date.fromisoformat(str(val))
    except (ValueError, TypeError):
        return None


def extract_facts(cik: int, company_name: str, raw: dict) -> pd.DataFrame:
    """
    Parse one company-facts JSON blob → tidy DataFrame.

    SEC company-facts structure::

        {
          "cik": 320193,
          "entityName": "Apple Inc.",
          "facts": {
            "us-gaap": {
              "Assets": {
                "units": {
                  "USD": [
                    {"end": "2022-09-24", "val": 352755000000,
                     "fy": 2022, "fp": "FY", "form": "10-K",
                     "filed": "2022-10-28"},
                    ...
                  ]
                }
              }
            }
          }
        }

    Returns an empty DataFrame if no extractable data is found.
    """
    facts_root: dict = raw.get("facts", {})
    # Merge namespaces; us-gaap takes precedence over ifrs-full
    namespaces: dict = {
        **facts_root.get("ifrs-full", {}),
        **facts_root.get("us-gaap", {}),
    }

    if not namespaces:
        log.debug("CIK %s: no facts namespace — skipping", cik)
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # 1. Collect raw (period, field, value) tuples
    # ------------------------------------------------------------------
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
                    "cik":          cik,
                    "company_name": company_name,
                    "period_end":   period_end,
                    "fiscal_year":  int(fy) if fy else period_end.year,
                    "fiscal_period": entry.get("fp", ""),
                    "form_type":    form,
                    "filed_date":   _parse_date(entry.get("filed")),
                    "field":        canonical_field,
                    "value":        float(val),
                })
            if any(r["field"] == canonical_field for r in rows):
                found = True  # found via this tag; skip lower-priority aliases

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).sort_values("filed_date", na_position="first")

    # ------------------------------------------------------------------
    # 2. Pivot → one row per (cik, period_end, fiscal_period)
    #    aggfunc="last" picks the most-recently-filed value per field,
    #    but values only present in an earlier filing are still preserved
    #    (amended filings don't re-submit every field).
    # ------------------------------------------------------------------
    pivot = df.pivot_table(
        index=["cik", "company_name", "period_end", "fiscal_year", "fiscal_period"],
        columns="field",
        values="value",
        aggfunc="last",
    ).reset_index()
    pivot.columns.name = None

    # Attach form_type and filed_date from the most-recently-filed row per period
    meta = (
        df[["cik", "period_end", "form_type", "filed_date"]]
        .drop_duplicates(subset=["cik", "period_end"], keep="last")
    )
    pivot = pivot.merge(meta, on=["cik", "period_end"], how="left")

    # Ensure all canonical columns exist
    for col in TAG_ALIASES:
        if col not in pivot.columns:
            pivot[col] = float("nan")

    # ------------------------------------------------------------------
    # 3. Derived fields (consumed directly by gold layer Z-score ratios)
    # ------------------------------------------------------------------

    # working_capital = current_assets - current_liabilities  (Altman X1 numerator)
    pivot["working_capital"] = (
        pivot["assets_current"] - pivot["liabilities_current"]
    )

    # ebit: prefer operating_income; fall back to net_income + interest_expense
    # so Altman X3 always has a value when operating_income is missing.
    pivot["ebit"] = pivot["operating_income"].copy()
    missing_oi = pivot["ebit"].isna()
    pivot.loc[missing_oi, "ebit"] = (
        pivot.loc[missing_oi, "net_income"].fillna(0)
        + pivot.loc[missing_oi, "interest_expense"].fillna(0)
    )
    # Restore NaN when both income fields were absent (avoid a spurious 0)
    both_nan = missing_oi & pivot["net_income"].isna() & pivot["interest_expense"].isna()
    pivot.loc[both_nan, "ebit"] = float("nan")

    # ------------------------------------------------------------------
    # 4. Enforce output column order
    # ------------------------------------------------------------------
    for col in OUTPUT_COLUMNS:
        if col not in pivot.columns:
            pivot[col] = float("nan")

    # Final dedup: keep one row per (cik, period_end)
    # Priority: FY > Q4 > Q3 > Q2 > Q1, then most recently filed
    period_rank = {"FY": 0, "Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4}
    pivot["_period_rank"] = pivot["fiscal_period"].map(period_rank).fillna(9)
    pivot = pivot.sort_values(
        ["_period_rank", "filed_date"], ascending=[True, False], na_position="last"
    )
    pivot = pivot.drop_duplicates(subset=["cik", "period_end"], keep="first")
    pivot = pivot.drop(columns=["_period_rank"])

    return pivot[OUTPUT_COLUMNS].reset_index(drop=True)


# =============================================================================
# MinIO I/O
# =============================================================================

def _get_client(endpoint: str, access_key: str, secret_key: str, secure: bool) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _ensure_bucket(client: Minio, bucket: str) -> None:
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log.info("Created bucket: %s", bucket)


def _stream_bronze(
    client: Minio, bucket: str, prefix: str = ""
) -> Iterator[tuple[str, dict]]:
    """Yield (object_name, parsed_json) for every .json in the bronze bucket."""
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    for obj in objects:
        if not obj.object_name.endswith(".json") or obj.object_name.endswith(".meta.json"):
            continue
        try:
            resp = client.get_object(bucket, obj.object_name)
            try:
                raw = json.loads(resp.read())
            finally:
                resp.close()
                resp.release_conn()
            yield obj.object_name, raw
        except (S3Error, json.JSONDecodeError, Exception) as exc:  # noqa: BLE001
            log.warning("SKIP %s — %s", obj.object_name, exc)


def _upload_parquet(client: Minio, bucket: str, object_name: str, df: pd.DataFrame) -> None:
    """Serialise df → Snappy Parquet in memory, then upload to MinIO."""
    buf = io.BytesIO()
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
    data = buf.getvalue()
    client.put_object(
        bucket, object_name,
        data=io.BytesIO(data), length=len(data),
        content_type="application/octet-stream",
    )
    log.info("Uploaded  %-60s  (%d rows, %d KB)", object_name, len(df), len(data) // 1024)


# =============================================================================
# Pipeline
# =============================================================================

def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce dtypes for clean Parquet output (no CAST() needed in gold DuckDB queries)."""
    df = df.copy()
    df["cik"]          = df["cik"].astype("int64")
    df["period_end"]   = pd.to_datetime(df["period_end"])
    df["filed_date"]   = pd.to_datetime(df["filed_date"])
    df["fiscal_year"]  = df["fiscal_year"].astype("Int64")   # nullable int
    df["fiscal_period"] = df["fiscal_period"].astype("string")
    df["form_type"]    = df["form_type"].astype("string")
    df["company_name"] = df["company_name"].astype("string")
    float_cols = [c for c in OUTPUT_COLUMNS if c not in
                  ("cik","company_name","period_end","fiscal_year",
                   "fiscal_period","form_type","filed_date")]
    for col in float_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


def run(
    endpoint:      str  = os.getenv("MINIO_ENDPOINT",       "localhost:9000"),
    access_key:    str  = os.getenv("MINIO_ACCESS_KEY",     "minioadmin"),
    secret_key:    str  = os.getenv("MINIO_SECRET_KEY",     "minioadmin"),
    secure:        bool = os.getenv("MINIO_SECURE",         "false") == "true",
    bronze_bucket: str  = os.getenv("MINIO_BRONZE_BUCKET",  "bronze"),
    silver_bucket: str  = os.getenv("MINIO_SILVER_BUCKET",  "silver"),
    prefix:        str  = os.getenv("BRONZE_PREFIX",        ""),
    dry_run:       bool = False,
) -> pd.DataFrame:
    """
    Execute the full silver pipeline and return the assembled DataFrame.

    The DataFrame is also written to MinIO unless *dry_run* is True.
    Returning the DataFrame lets notebooks chain directly into the gold layer
    without a round-trip through MinIO.
    """
    log.info("=" * 60)
    log.info("Silver layer — start")
    log.info("  MinIO  : %s  (secure=%s)", endpoint, secure)
    log.info("  Bronze : %s/%s", bronze_bucket, prefix or "*")
    log.info("  Silver : %s", silver_bucket)
    log.info("  Dry run: %s", dry_run)
    log.info("=" * 60)

    client = _get_client(endpoint, access_key, secret_key, secure)

    frames: list[pd.DataFrame] = []
    n_objects = n_errors = 0

    for obj_name, raw in _stream_bronze(client, bronze_bucket, prefix):
        n_objects += 1
        cik          = raw.get("cik", 0)
        company_name = raw.get("entityName", "Unknown")
        log.info("[%4d] %s  (CIK %s, %s)", n_objects, obj_name, cik, company_name)
        try:
            df = extract_facts(cik, company_name, raw)
        except Exception as exc:  # noqa: BLE001
            log.warning("  SKIP — extraction error: %s", exc)
            n_errors += 1
            continue
        if df.empty:
            log.debug("  No rows extracted")
            continue
        log.info("  → %d rows", len(df))
        frames.append(df)

    if not frames:
        log.error("No data extracted. Nothing to write.")
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = _coerce_types(pd.concat(frames, ignore_index=True))

    log.info("-" * 60)
    log.info("Objects processed : %d  (errors: %d)", n_objects, n_errors)
    log.info("Total rows        : %d", len(combined))
    log.info("Companies         : %d", combined["cik"].nunique())
    log.info("Fiscal years      : %s – %s",
             int(combined["fiscal_year"].min()), int(combined["fiscal_year"].max()))

    if dry_run:
        log.info("[DRY RUN] Skipping MinIO write.")
    else:
        _ensure_bucket(client, silver_bucket)
        for fy in sorted(combined["fiscal_year"].dropna().unique()):
            partition = combined[combined["fiscal_year"] == fy].copy()
            object_name = f"financial_facts/fiscal_year={int(fy)}/data.parquet"
            try:
                _upload_parquet(client, silver_bucket, object_name, partition)
            except S3Error as exc:
                log.error("Failed to upload %s: %s", object_name, exc)

    log.info("Silver layer — complete")
    log.info("=" * 60)
    return combined


# =============================================================================
# CLI  (mirrors gold/build_gold.py usage)
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Silver: parse EDGAR bronze JSON → clean Parquet in MinIO silver.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--endpoint",      default=os.getenv("MINIO_ENDPOINT",    "localhost:9000"))
    p.add_argument("--access-key",    default=os.getenv("MINIO_ACCESS_KEY",  "minioadmin"))
    p.add_argument("--secret-key",    default=os.getenv("MINIO_SECRET_KEY",  "minioadmin"))
    p.add_argument("--secure",        action="store_true",
                   default=os.getenv("MINIO_SECURE", "false").lower() == "true")
    p.add_argument("--bronze-bucket", default=os.getenv("MINIO_BRONZE_BUCKET", "bronze"))
    p.add_argument("--silver-bucket", default=os.getenv("MINIO_SILVER_BUCKET", "silver"))
    p.add_argument("--prefix",        default=os.getenv("BRONZE_PREFIX", ""),
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
    result = run(
        endpoint=args.endpoint,
        access_key=args.access_key,
        secret_key=args.secret_key,
        secure=args.secure,
        bronze_bucket=args.bronze_bucket,
        silver_bucket=args.silver_bucket,
        prefix=args.prefix,
        dry_run=args.dry_run,
    )
    if result.empty:
        sys.exit(1)
