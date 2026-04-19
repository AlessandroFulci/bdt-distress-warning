"""
Gold layer: compute financial distress indicators from the silver Parquet table.

Uses DuckDB for fast in-process SQL analytics — no JVM, no cluster needed.

Outputs per (company, period):
  Altman Z-Score
    X1 = working_capital / assets_total
    X2 = retained_earnings / assets_total
    X3 = ebit / assets_total
    X4 = equity / liabilities_total          (book value proxy for market cap)
    X5 = revenue / assets_total
    Z  = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

  Distress zone
    Z < 1.81            → distress
    1.81 <= Z <= 2.99   → grey
    Z > 2.99            → safe

  Quarter-over-quarter trend features (via LAG window functions)
    revenue_growth, net_income_growth, asset_growth, equity_growth

  Trend-break flags (3+ consecutive declining periods)
    revenue_declining_3q, net_income_declining_3q

  Ground-truth label (from company_universe.csv)
    distress_label, distress_event_date, distress_event_type
"""

import glob
import logging
import os
from pathlib import Path

import duckdb
import pandas as pd
from dotenv import load_dotenv
from minio import Minio

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
UNIVERSE_PATH = PROJECT_ROOT / "config" / "company_universe.csv"
SILVER_LOCAL = "/tmp/silver_financials"
GOLD_LOCAL = "/tmp/gold_distress"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "gold_build.log"),
            logging.StreamHandler(),
        ],
    )


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
        logging.getLogger("gold").info(f"Created bucket: {bucket}")


def download_silver_from_minio(client: Minio, silver_bucket: str, local_path: str):
    """Download silver Parquet files from MinIO if not already cached locally."""
    logger = logging.getLogger("gold.download")
    existing = glob.glob(f"{local_path}/**/*.parquet", recursive=True)
    if existing:
        logger.info(f"Using cached silver Parquet at {local_path} ({len(existing)} files)")
        return

    logger.info("Downloading silver Parquet from MinIO...")
    os.makedirs(local_path, exist_ok=True)
    objects = list(client.list_objects(silver_bucket, prefix="financial_facts/", recursive=True))  # ← era "financials/"
    for obj in objects:
        dest = os.path.join(local_path, obj.object_name.replace("financial_facts/", ""))  # ← era "financials/"
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        client.fget_object(silver_bucket, obj.object_name, dest)
    logger.info(f"Downloaded {len(objects)} files")


def upload_gold_to_minio(client: Minio, gold_bucket: str, local_path: str):
    """Upload gold Parquet files to MinIO."""
    logger = logging.getLogger("gold.upload")
    ensure_bucket(client, gold_bucket)
    parquet_files = glob.glob(f"{local_path}/**/*.parquet", recursive=True)
    logger.info(f"Uploading {len(parquet_files)} gold files to MinIO '{gold_bucket}' bucket")
    for pf in parquet_files:
        rel = pf.replace(local_path, "").replace("\\", "/").lstrip("/")
        client.fput_object(gold_bucket, f"distress_scores/{rel}", pf)
    logger.info(f"Done — gold data at MinIO: {gold_bucket}/distress_scores/")


# ---------------------------------------------------------------------------
# DuckDB analytics
# ---------------------------------------------------------------------------
GOLD_SQL = """
-- ============================================================
-- Step 1: load silver + universe label
-- ============================================================
CREATE OR REPLACE VIEW silver AS
    SELECT * FROM read_parquet('{silver_path}/**/*.parquet', hive_partitioning=true);

CREATE OR REPLACE VIEW universe AS
    SELECT
        LPAD(CAST(cik AS VARCHAR), 10, '0') AS cik,
        distress_label,
        distress_event_date,
        distress_event_type
    FROM read_csv_auto('{universe_path}');

-- ============================================================
-- Step 2: base table — only annual + quarterly, non-null assets
-- ============================================================
CREATE OR REPLACE TABLE base AS
SELECT
    s.cik,
    s.company_name,
    CAST(s.period_end AS DATE)   AS period_end,
    s.fiscal_year,
    s.fiscal_period,
    s.form_type,
    CAST(s.filed_date AS DATE)   AS filed_date,
    s.assets_total,
    s.assets_current,
    s.liabilities_total,
    s.liabilities_current,
    s.equity,
    s.retained_earnings,
    s.net_income,
    s.revenue,
    s.operating_income,
    s.interest_expense,
    s.cash,
    s.long_term_debt,
    s.ebit,
    s.working_capital,
    COALESCE(u.distress_label, 0)        AS distress_label,
    u.distress_event_date,
    u.distress_event_type
FROM silver s
LEFT JOIN universe u USING (cik)
WHERE s.assets_total IS NOT NULL
  AND s.assets_total > 0;

-- ============================================================
-- Step 3: Altman Z-Score + zone classification
-- ============================================================
CREATE OR REPLACE TABLE altman AS
SELECT
    *,
    -- Component ratios (NULL-safe division)
    CASE WHEN assets_total > 0 THEN working_capital   / assets_total ELSE NULL END AS x1_wc_to_assets,
    CASE WHEN assets_total > 0 THEN retained_earnings / assets_total ELSE NULL END AS x2_re_to_assets,
    CASE WHEN assets_total > 0 THEN ebit              / assets_total ELSE NULL END AS x3_ebit_to_assets,
    CASE WHEN liabilities_total > 0 THEN equity       / liabilities_total ELSE NULL END AS x4_equity_to_liab,
    CASE WHEN assets_total > 0 THEN revenue           / assets_total ELSE NULL END AS x5_rev_to_assets,

    -- Z-Score
    CASE
        WHEN assets_total > 0 AND liabilities_total > 0
             AND working_capital IS NOT NULL
             AND retained_earnings IS NOT NULL
             AND ebit IS NOT NULL
             AND equity IS NOT NULL
             AND revenue IS NOT NULL
        THEN
            1.2 * (working_capital   / assets_total)
          + 1.4 * (retained_earnings / assets_total)
          + 3.3 * (ebit              / assets_total)
          + 0.6 * (equity            / liabilities_total)
          + 1.0 * (revenue           / assets_total)
        ELSE NULL
    END AS altman_z_score
FROM base;

-- ============================================================
-- Step 4: zone labels + QoQ trend features
-- ============================================================
CREATE OR REPLACE TABLE gold AS
SELECT
    *,

    -- Distress zone
    CASE
        WHEN altman_z_score IS NULL   THEN NULL
        WHEN altman_z_score < 1.81    THEN 'distress'
        WHEN altman_z_score <= 2.99   THEN 'grey'
        ELSE                               'safe'
    END AS distress_zone,

    -- Quarter-over-quarter growth rates (using LAG over prior period for same company + form)
    LAG(revenue,    1) OVER w AS prev_revenue,
    LAG(net_income, 1) OVER w AS prev_net_income,
    LAG(assets_total, 1) OVER w AS prev_assets,
    LAG(equity,     1) OVER w AS prev_equity,

    CASE WHEN LAG(revenue,    1) OVER w > 0
         THEN (revenue    - LAG(revenue,    1) OVER w) / LAG(revenue,    1) OVER w
         ELSE NULL END AS revenue_growth_qoq,

    CASE WHEN LAG(net_income, 1) OVER w IS NOT NULL AND LAG(net_income, 1) OVER w != 0
         THEN (net_income - LAG(net_income, 1) OVER w) / ABS(LAG(net_income, 1) OVER w)
         ELSE NULL END AS net_income_growth_qoq,

    CASE WHEN LAG(assets_total, 1) OVER w > 0
         THEN (assets_total - LAG(assets_total, 1) OVER w) / LAG(assets_total, 1) OVER w
         ELSE NULL END AS asset_growth_qoq,

    -- Trend-break flags: 3+ consecutive declines
    CASE
        WHEN revenue < LAG(revenue, 1) OVER w
         AND LAG(revenue, 1) OVER w < LAG(revenue, 2) OVER w
         AND LAG(revenue, 2) OVER w < LAG(revenue, 3) OVER w
        THEN 1 ELSE 0
    END AS revenue_declining_3q,

    CASE
        WHEN net_income < LAG(net_income, 1) OVER w
         AND LAG(net_income, 1) OVER w < LAG(net_income, 2) OVER w
         AND LAG(net_income, 2) OVER w < LAG(net_income, 3) OVER w
        THEN 1 ELSE 0
    END AS net_income_declining_3q

FROM altman
WINDOW w AS (PARTITION BY cik, form_type ORDER BY period_end);
"""


def run_gold_pipeline(silver_local: str, universe_path: str) -> pd.DataFrame:
    logger = logging.getLogger("gold.duckdb")
    logger.info("Running DuckDB gold pipeline...")

    con = duckdb.connect()

    sql = GOLD_SQL.format(
        silver_path=silver_local,
        universe_path=str(universe_path),
    )

    # Execute all statements
    for statement in sql.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            con.execute(stmt)

    df = con.execute("SELECT * FROM gold ORDER BY cik, period_end").df()
    logger.info(f"Gold table: {len(df):,} rows × {len(df.columns)} columns")

    # Quick summary
    if "distress_zone" in df.columns:
        zone_counts = df["distress_zone"].value_counts()
        logger.info(f"Distress zone distribution:\n{zone_counts.to_string()}")

    if "altman_z_score" in df.columns:
        logger.info(f"Z-Score stats:\n{df['altman_z_score'].describe().to_string()}")

    con.close()
    return df


# ---------------------------------------------------------------------------
# Write Parquet
# ---------------------------------------------------------------------------
def write_gold_parquet(df: pd.DataFrame, local_path: str):
    import shutil
    logger = logging.getLogger("gold.write")

    if os.path.exists(local_path):
        shutil.rmtree(local_path)
    os.makedirs(local_path, exist_ok=True)

    # Partition by distress_label for easy downstream filtering
    for label, group in df.groupby("distress_label", dropna=False):
        label_str = f"distress_label={int(label)}" if pd.notna(label) else "distress_label=unknown"
        part_dir = os.path.join(local_path, label_str)
        os.makedirs(part_dir, exist_ok=True)
        out_file = os.path.join(part_dir, "part-0.parquet")
        group.drop(columns=["distress_label"]).to_parquet(out_file, index=False, engine="pyarrow")

    files = glob.glob(f"{local_path}/**/*.parquet", recursive=True)
    logger.info(f"Wrote {len(files)} gold Parquet files to {local_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    setup_logging()
    logger = logging.getLogger("gold.main")

    silver_bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
    gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")

    logger.info("=" * 70)
    logger.info("Gold build started")
    logger.info("=" * 70)

    client = get_minio_client()

    # Step 1: ensure silver Parquet is available locally
    logger.info("Step 1/4: Ensuring silver data is available locally...")
    download_silver_from_minio(client, silver_bucket, SILVER_LOCAL)

    # Step 2: run DuckDB pipeline
    logger.info("Step 2/4: Running DuckDB analytics...")
    df = run_gold_pipeline(SILVER_LOCAL, UNIVERSE_PATH)

    # Step 3: write gold Parquet locally
    logger.info("Step 3/4: Writing gold Parquet locally...")
    write_gold_parquet(df, GOLD_LOCAL)

    # Step 4: upload to MinIO
    logger.info("Step 4/4: Uploading gold to MinIO...")
    upload_gold_to_minio(client, gold_bucket, GOLD_LOCAL)

    logger.info("=" * 70)
    logger.info("Gold build complete!")
    logger.info(f"  Rows:          {len(df):,}")
    logger.info(f"  Columns:       {len(df.columns)}")
    logger.info(f"  Local Parquet: {GOLD_LOCAL}")
    logger.info(f"  MinIO gold:    {gold_bucket}/distress_scores/")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
