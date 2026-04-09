"""
Bronze layer ingestion: fetch SEC EDGAR company facts + submissions
for every company in the universe and store them in MinIO.

Resumable: skips objects that already exist in the bucket.
Logged: writes a structured log to logs/ingestion_<date>.log
"""
import os
import sys
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# Make 'src' importable when running this file directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.sec_client import SECClient
from src.ingestion.minio_client import BronzeStore

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UNIVERSE_PATH = PROJECT_ROOT / "config" / "company_universe.csv"
LOGS_DIR = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

INGESTION_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
LOG_FILE = LOGS_DIR / f"ingestion_{INGESTION_DATE}.log"

# Bronze layout (key prefixes inside the bucket)
FACTS_PREFIX = f"edgar_company_facts/ingestion_date={INGESTION_DATE}"
SUBS_PREFIX = f"edgar_submissions/ingestion_date={INGESTION_DATE}"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )
    # Quiet down noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("minio").setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Main ingestion logic
# ---------------------------------------------------------------------------
def ingest_company(cik: str, sec: SECClient, store: BronzeStore) -> dict:
    """
    Fetch company facts + submissions for one CIK and store both in MinIO.
    Returns a result dict for tracking.
    """
    result = {
        "cik": cik,
        "facts_status": "skipped",
        "subs_status": "skipped",
        "facts_size": 0,
        "subs_size": 0,
    }

    # ---- Company facts ----
    facts_object = f"{FACTS_PREFIX}/CIK{cik}.json"
    if store.object_exists(facts_object):
        result["facts_status"] = "already_exists"
    else:
        facts_url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
        facts_bytes = sec.fetch_company_facts(cik)
        if facts_bytes is not None:
            meta = store.put_with_metadata(
                object_name=facts_object,
                data=facts_bytes,
                source_url=facts_url,
                http_status=200,
                extra_metadata={"cik": cik, "dataset": "company_facts"},
            )
            result["facts_status"] = "ok"
            result["facts_size"] = meta["size_bytes"]
        else:
            result["facts_status"] = "failed"

    # ---- Submissions ----
    subs_object = f"{SUBS_PREFIX}/CIK{cik}.json"
    if store.object_exists(subs_object):
        result["subs_status"] = "already_exists"
    else:
        subs_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        subs_bytes = sec.fetch_submissions(cik)
        if subs_bytes is not None:
            meta = store.put_with_metadata(
                object_name=subs_object,
                data=subs_bytes,
                source_url=subs_url,
                http_status=200,
                extra_metadata={"cik": cik, "dataset": "submissions"},
            )
            result["subs_status"] = "ok"
            result["subs_size"] = meta["size_bytes"]
        else:
            result["subs_status"] = "failed"

    return result


def main():
    setup_logging()
    logger = logging.getLogger("fetch_bronze")

    logger.info("=" * 70)
    logger.info(f"Bronze ingestion run: {INGESTION_DATE}")
    logger.info("=" * 70)

    # Load company universe
    if not UNIVERSE_PATH.exists():
        logger.error(f"Universe file not found: {UNIVERSE_PATH}")
        sys.exit(1)
    universe = pd.read_csv(UNIVERSE_PATH, dtype={"cik": str})
    logger.info(f"Loaded {len(universe)} companies from universe")

    # Build clients
    sec = SECClient(user_agent=os.getenv("SEC_USER_AGENT"), max_per_second=5.0)
    store = BronzeStore(
        endpoint=os.getenv("MINIO_ENDPOINT"),
        access_key=os.getenv("MINIO_ACCESS_KEY"),
        secret_key=os.getenv("MINIO_SECRET_KEY"),
        bucket=os.getenv("MINIO_BRONZE_BUCKET"),
        secure=os.getenv("MINIO_SECURE") == "true",
    )

    # Ingest each company with a progress bar
    results = []
    for cik in tqdm(universe["cik"].tolist(), desc="Ingesting", unit="company"):
        try:
            res = ingest_company(cik, sec, store)
            results.append(res)
        except Exception as e:
            logger.exception(f"Unexpected error for CIK={cik}: {e}")
            results.append({
                "cik": cik,
                "facts_status": "error",
                "subs_status": "error",
                "facts_size": 0,
                "subs_size": 0,
            })

    # Summary
    df = pd.DataFrame(results)
    logger.info("=" * 70)
    logger.info("Run summary:")
    logger.info(f"  Total companies processed: {len(df)}")
    logger.info(f"  Company facts: {df['facts_status'].value_counts().to_dict()}")
    logger.info(f"  Submissions:   {df['subs_status'].value_counts().to_dict()}")
    total_mb = (df["facts_size"].sum() + df["subs_size"].sum()) / (1024 * 1024)
    logger.info(f"  Total downloaded: {total_mb:.1f} MB")
    logger.info("=" * 70)

    # Write a per-run summary CSV alongside the log
    summary_path = LOGS_DIR / f"ingestion_{INGESTION_DATE}_summary.csv"
    df.to_csv(summary_path, index=False)
    logger.info(f"Per-company results saved to: {summary_path}")


if __name__ == "__main__":
    main()
