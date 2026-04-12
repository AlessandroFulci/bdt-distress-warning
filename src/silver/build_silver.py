"""
Silver layer: parse raw EDGAR company facts JSON from MinIO bronze,
extract key financial line items, standardize schema, and write Parquet.

Pipeline:
  1. Read every company facts JSON from the bronze bucket
  2. Extract key financial concepts using a tag-alias map
     (different companies use different GAAP tags for the same metric)
  3. Pivot to one row per (CIK, period_end, form_type)
  4. Deduplicate: for the same reporting period, keep the most recently filed entry
  5. Compute derived fields: working_capital
  6. Write to MinIO silver bucket as Parquet, partitioned by fiscal_year

Output schema (one row per company per reporting period):
  cik, company_name, period_end, fiscal_year, fiscal_period,
  form_type, filed_date, assets_total, assets_current,
  liabilities_total, liabilities_current, equity, retained_earnings,
  net_income, revenue, operating_income, interest_expense,
  cash, long_term_debt, ebit, working_capital
"""

import glob
import json
import logging
import os
from pathlib import Path

os.environ["JAVA_TOOL_OPTIONS"] = (
    "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
)

import pandas as pd
from dotenv import load_dotenv
from minio import Minio
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Values are GAAP tags tried in order — the first one found in the filing wins.
# ---------------------------------------------------------------------------
CONCEPT_MAP = {
    "assets_total": ["Assets"],
    "assets_current": ["AssetsCurrent"],
    "liabilities_total": ["Liabilities"],
    "liabilities_current": ["LiabilitiesCurrent"],
    "equity": [
        "StockholdersEquity",
        "StockholdersEquityAttributableToParent",
    ],
    "retained_earnings": ["RetainedEarningsAccumulatedDeficit"],
    "net_income": [
        "NetIncomeLoss",
        "NetIncomeLossAvailableToCommonStockholdersBasic",
    ],
    "revenue": [
        "Revenues",
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
        "SalesRevenueNet",
        "SalesRevenueGoodsNet",
    ],
    "operating_income": ["OperatingIncomeLoss"],
    "interest_expense": [
        "InterestExpense",
        "InterestExpenseDebt",
        "InterestAndDebtExpense",
    ],
    "cash": [
        "CashAndCashEquivalentsAtCarryingValue",
        "CashCashEquivalentsAndShortTermInvestments",
        "Cash",
    ],
    "long_term_debt": [
        "LongTermDebt",
        "LongTermDebtNoncurrent",
        "LongTermNotesPayable",
    ],
    # EBIT proxy: operating income is EBIT before D&A adjustments
    "ebit": ["OperatingIncomeLoss"],
}

# Only ingest rows from standard annual/quarterly reports (include amendments)
VALID_FORMS = {"10-K", "10-K/A", "10-Q", "10-Q/A"}


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "silver_build.log"),
            logging.StreamHandler(),
        ],
    )
    logging.getLogger("py4j").setLevel(logging.WARNING)
    logging.getLogger("pyspark").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------
def get_minio_client() -> Minio:
    return Minio(
        os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        secure=os.getenv("MINIO_SECURE") == "true",
    )


def ensure_bucket(client: Minio, bucket: str):
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        logging.getLogger("silver").info(f"Created bucket: {bucket}")


# ---------------------------------------------------------------------------
# XBRL parsing
# ---------------------------------------------------------------------------
def extract_concept(facts_gaap: dict, tags: list) -> dict:
    """
    Try each tag alias in order. Return a dict keyed by
    (period_end, form_normalized, accn) -> {val, fy, fp, filed, form}.
    """
    for tag in tags:
        if tag not in facts_gaap:
            continue
        usd_entries = facts_gaap[tag].get("units", {}).get("USD", [])
        if not usd_entries:
            continue

        result = {}
        for entry in usd_entries:
            form = entry.get("form", "")
            if form not in VALID_FORMS:
                continue
            period_end = entry.get("end", "")
            accn = entry.get("accn", "")
            val = entry.get("val")
            if val is None or not period_end:
                continue
            # Normalize amendments: 10-K/A -> 10-K, 10-Q/A -> 10-Q
            form_norm = form.replace("/A", "")
            key = (period_end, form_norm, accn)
            result[key] = {
                "val": val,
                "fy": entry.get("fy"),
                "fp": entry.get("fp", ""),
                "filed": entry.get("filed", ""),
                "form": form,
            }
        if result:
            return result
    return {}


def parse_company_facts(cik: str, company_name: str, facts_json: dict) -> list:
    """
    Parse one company facts JSON into a list of row dicts
    (one per reporting period anchored on assets_total).
    """
    gaap = facts_json.get("facts", {}).get("us-gaap", {})
    if not gaap:
        return []

    # Extract all concepts
    concept_data = {name: extract_concept(gaap, tags) for name, tags in CONCEPT_MAP.items()}

    # Use assets_total as the anchor to discover all valid reporting periods
    anchor = concept_data.get("assets_total", {})
    if not anchor:
        return []

    # For each (period_end, form_norm) keep only the most recently filed accn
    best_per_period = {}
    for (period_end, form_norm, accn), entry in anchor.items():
        key = (period_end, form_norm)
        if key not in best_per_period or entry["filed"] > best_per_period[key]["filed"]:
            best_per_period[key] = {
                "accn": accn,
                "filed": entry["filed"],
                "fy": entry["fy"],
                "fp": entry["fp"],
            }

    rows = []
    for (period_end, form_norm), best in best_per_period.items():
        best_accn = best["accn"]
        row = {
            "cik": cik,
            "company_name": company_name,
            "period_end": period_end,
            "fiscal_year": best["fy"],
            "fiscal_period": best["fp"],
            "form_type": form_norm,
            "filed_date": best["filed"],
        }

        # Pull each concept value — prefer exact accn match, then any for same period/form
        for concept_name in CONCEPT_MAP:
            vals = concept_data.get(concept_name, {})
            val = None
            if (period_end, form_norm, best_accn) in vals:
                val = vals[(period_end, form_norm, best_accn)]["val"]
            else:
                for (pe, fn, _acc), entry in vals.items():
                    if pe == period_end and fn == form_norm:
                        val = entry["val"]
                        break
            row[concept_name] = val

        # Derived field
        ac = row.get("assets_current")
        lc = row.get("liabilities_current")
        row["working_capital"] = (ac - lc) if (ac is not None and lc is not None) else None

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Bronze → rows
# ---------------------------------------------------------------------------
def process_all_companies(client: Minio, bronze_bucket: str) -> list:
    logger = logging.getLogger("silver.parse")

    objects = list(client.list_objects(bronze_bucket, recursive=True))
    facts_objects = [
        o for o in objects
        if "edgar_company_facts" in o.object_name
        and o.object_name.endswith(".json")
        and not o.object_name.endswith(".meta.json")
    ]
    logger.info(f"Found {len(facts_objects)} company facts files in bronze")

    all_rows = []
    failed = 0
    for i, obj in enumerate(facts_objects):
        try:
            resp = client.get_object(bronze_bucket, obj.object_name)
            data = json.loads(resp.read().decode("utf-8"))
            resp.close()

            cik = str(data.get("cik", "")).zfill(10)
            company_name = data.get("entityName", data.get("name", ""))
            rows = parse_company_facts(cik, company_name, data)
            all_rows.extend(rows)

            if (i + 1) % 50 == 0:
                logger.info(f"  [{i+1}/{len(facts_objects)}] {len(all_rows)} rows so far")
        except Exception as e:
            logger.warning(f"Failed to process {obj.object_name}: {e}")
            failed += 1

    logger.info(f"Parsing complete: {len(all_rows)} rows from {len(facts_objects) - failed} companies ({failed} failed)")
    return all_rows


# ---------------------------------------------------------------------------
# Write Parquet → MinIO
# ---------------------------------------------------------------------------
def write_parquet_to_minio(spark, df, client: Minio, silver_bucket: str) -> str:
    """
    Write the silver DataFrame as Parquet using pyarrow (avoids Spark JDK
    compatibility issues with newer Java versions), then upload to MinIO.
    Partitioned by fiscal_year.
    """
    import shutil
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger = logging.getLogger("silver.write")
    ensure_bucket(client, silver_bucket)

    local_path = "/tmp/silver_financials"

    # Clean up any previous run
    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    # Convert Spark DF → pandas, then write with pyarrow partitioned by fiscal_year
    logger.info("Converting Spark DataFrame to pandas for Parquet write...")
    pdf = df.toPandas()

    # Ensure fiscal_year is nullable int for partitioning
    pdf["fiscal_year"] = pd.to_numeric(pdf["fiscal_year"], errors="coerce")

    for fy, group in pdf.groupby("fiscal_year", dropna=False):
        fy_label = f"fiscal_year={int(fy)}" if pd.notna(fy) else "fiscal_year=unknown"
        partition_dir = os.path.join(local_path, fy_label)
        os.makedirs(partition_dir, exist_ok=True)
        out_file = os.path.join(partition_dir, "part-0.parquet")
        group.drop(columns=["fiscal_year"]).to_parquet(out_file, index=False, engine="pyarrow")

    # Upload all partition files to MinIO
    parquet_files = glob.glob(f"{local_path}/**/*.parquet", recursive=True)
    logger.info(f"Uploading {len(parquet_files)} parquet files to MinIO '{silver_bucket}' bucket")

    for pf in parquet_files:
        rel = pf.replace(local_path + "/", "")
        client.fput_object(silver_bucket, f"financials/{rel}", pf)

    logger.info(f"Done — silver data at MinIO: {silver_bucket}/financials/")
    return local_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_logging()
    logger = logging.getLogger("silver.main")

    bronze_bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")

    logger.info("=" * 70)
    logger.info("Silver build started")
    logger.info("=" * 70)

    client = get_minio_client()

    # --- Step 1: Parse bronze ---
    logger.info("Step 1/3: Parsing bronze company facts...")
    rows = process_all_companies(client, bronze_bucket)

    if not rows:
        logger.error("No rows extracted — is the bronze bucket populated?")
        return

    # --- Step 2: Build Spark DataFrame ---
    logger.info("Step 2/3: Building Spark DataFrame...")
    warehouse_path = str(PROJECT_ROOT / "spark-warehouse")
    jvm_opens = "--add-opens=java.base/javax.security.auth=ALL-UNNAMED"
    spark = (
        SparkSession.builder
        .appName("BDT-Silver-Financials")
        .config("spark.driver.memory", "2g")
        .config("spark.sql.warehouse.dir", warehouse_path)
        .config("spark.driver.extraJavaOptions", jvm_opens)
        .config("spark.executor.extraJavaOptions", jvm_opens)
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    pdf = pd.DataFrame(rows)

    # Cast numeric columns
    numeric_cols = [
        c for c in pdf.columns
        if c not in ("cik", "company_name", "period_end", "fiscal_period", "form_type", "filed_date")
    ]
    for col in numeric_cols:
        if col == "fiscal_year":
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce").astype("Int64")
        else:
            pdf[col] = pd.to_numeric(pdf[col], errors="coerce")

    pdf["period_end"] = pd.to_datetime(pdf["period_end"], errors="coerce")
    pdf["filed_date"] = pd.to_datetime(pdf["filed_date"], errors="coerce")

    df = spark.createDataFrame(pdf)

    # Deduplicate: for same (cik, period_end, form_type), keep most recently filed
    window = Window.partitionBy("cik", "period_end", "form_type").orderBy(F.col("filed_date").desc())
    df = (
        df.withColumn("_rank", F.row_number().over(window))
        .filter(F.col("_rank") == 1)
        .drop("_rank")
        .orderBy("cik", "period_end")
    )

    row_count = df.count()
    logger.info(f"DataFrame ready: {row_count:,} rows × {len(df.columns)} columns")
    df.printSchema()

    # Show a quick sample
    logger.info("Sample rows:")
    df.select("cik", "company_name", "period_end", "form_type", "assets_total", "revenue", "net_income").show(5, truncate=False)

    # --- Step 3: Write Parquet to MinIO ---
    logger.info("Step 3/3: Writing Parquet to MinIO silver bucket...")
    local_path = write_parquet_to_minio(spark, df, client, silver_bucket)

    logger.info("=" * 70)
    logger.info("Silver build complete!")
    logger.info(f"  Rows:              {row_count:,}")
    logger.info(f"  Local Parquet:     {local_path}")
    logger.info(f"  MinIO silver:      {silver_bucket}/financials/")
    logger.info("=" * 70)

    spark.stop()


if __name__ == "__main__":
    main()
