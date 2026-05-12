"""
Bronze text ingestion: fetch SEC EDGAR 10-K / 10-Q filing text sections
(MD&A, Risk Factors, Business Description) for every company in the universe
and store them as plain-text blobs in MinIO.

This complements fetch_bronze.py which stores structured XBRL facts.
The LLM silver pipeline (build_silver_text.py) reads from this bucket.

Layout in MinIO (bronze bucket)
--------------------------------
  edgar_filings_text/
    ingestion_date=<YYYY-MM-DD>/
      CIK<cik>_<accession>_<section>.txt

Sections extracted
------------------
  mda         → Item 7  "Management's Discussion and Analysis"
  risk        → Item 1A "Risk Factors"
  business    → Item 1  "Business"

Usage
-----
  python -m src.ingestion.fetch_bronze_text
  python -m src.ingestion.fetch_bronze_text --max-companies 10 --dry-run

Environment variables (same .env as fetch_bronze.py)
------------------------------------------------------
  SEC_USER_AGENT       e.g. "MyApp admin@example.com"
  MINIO_ENDPOINT       e.g. "localhost:9000"
  MINIO_ACCESS_KEY
  MINIO_SECRET_KEY
  MINIO_BRONZE_BUCKET  (default: bronze)
  MINIO_SECURE         (default: false)
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
UNIVERSE_PATH    = PROJECT_ROOT / "config" / "company_universe.csv"
LOGS_DIR         = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
INGESTION_DATE   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
TEXT_PREFIX      = f"edgar_filings_text/ingestion_date={INGESTION_DATE}"
SEC_BASE         = "https://data.sec.gov"
EDGAR_FULL_BASE  = "https://efts.sec.gov/LATEST/search-index"
HEADERS: dict    = {}   # populated from SEC_USER_AGENT env var

# Maximum bytes we keep per section (avoid storing 50MB+ 10-Ks in full)
MAX_SECTION_BYTES = 200_000   # ~200 KB ≈ ~50K tokens — enough for LLM context


# ---------------------------------------------------------------------------
# Section extraction patterns  (regex over raw HTML / text)
# ---------------------------------------------------------------------------
# We look for the SEC standard item headings in the filing text.
# Real 10-Ks use many formats; these patterns catch the most common ones.

SECTION_PATTERNS: dict[str, list[str]] = {
    "mda": [
        r"item\s*7[.\s]*management.{0,60}discussion",
        r"item\s*7[.\s]*md&a",
    ],
    "risk": [
        r"item\s*1a[.\s]*risk\s*factor",
    ],
    "business": [
        r"item\s*1[.\s]*business",
    ],
}

# Item that ends each section (approximate)
SECTION_END_PATTERNS: dict[str, list[str]] = {
    "mda":      [r"item\s*7a", r"item\s*8[.\s]*financial"],
    "risk":     [r"item\s*1b", r"item\s*2[.\s]"],
    "business": [r"item\s*1a", r"item\s*2[.\s]"],
}


def _clean_html(raw: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def _extract_section(text: str, section: str) -> Optional[str]:
    """
    Find and extract one section from cleaned filing text.
    Returns None if the section is not found.
    """
    lower = text.lower()
    start = -1
    for pat in SECTION_PATTERNS[section]:
        m = re.search(pat, lower)
        if m:
            start = m.start()
            break
    if start == -1:
        return None

    # Find end
    end = len(text)
    for pat in SECTION_END_PATTERNS.get(section, []):
        m = re.search(pat, lower[start + 50:])   # +50 to skip the header itself
        if m:
            end = start + 50 + m.start()
            break

    snippet = text[start:end].strip()
    # Truncate to MAX_SECTION_BYTES (utf-8)
    encoded = snippet.encode("utf-8")[:MAX_SECTION_BYTES]
    return encoded.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# SEC EDGAR helpers
# ---------------------------------------------------------------------------

def _get_recent_annual_filings(cik: str, max_filings: int = 3) -> list[dict]:
    """
    Return the most recent 10-K (or 20-F / 40-F) filing accession numbers
    for a given CIK, using the submissions endpoint already in bronze.
    """
    url = f"{SEC_BASE}/submissions/CIK{cik.zfill(10)}.json"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
    except Exception as exc:
        logging.getLogger("fetch_text").warning("Submissions fetch failed for CIK=%s: %s", cik, exc)
        return []

    filings = data.get("filings", {}).get("recent", {})
    forms       = filings.get("form", [])
    accessions  = filings.get("accessionNumber", [])
    dates       = filings.get("filingDate", [])
    primary_docs= filings.get("primaryDocument", [])

    annual_forms = {"10-K", "20-F", "40-F"}
    results = []
    for form, acc, date_, doc in zip(forms, accessions, dates, primary_docs):
        if form in annual_forms:
            results.append({
                "form": form,
                "accession": acc.replace("-", ""),
                "accession_dashed": acc,
                "date": date_,
                "primary_doc": doc,
            })
        if len(results) >= max_filings:
            break
    return results


def _fetch_filing_text(cik: str, accession: str, primary_doc: str) -> Optional[str]:
    """
    Download the primary HTML/HTM document for a filing and return cleaned text.
    Falls back to the filing index if the primary document is not HTML.
    """
    cik_padded = cik.zfill(10)
    acc_path   = accession  # no dashes

    # Primary document URL
    doc_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc_path}/{primary_doc}"
    try:
        r = requests.get(doc_url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "")
        if "html" in content_type or primary_doc.lower().endswith((".htm", ".html")):
            return _clean_html(r.text)
        # Plain text
        return r.text[:MAX_SECTION_BYTES * 5]
    except Exception as exc:
        logging.getLogger("fetch_text").warning(
            "Filing doc fetch failed CIK=%s acc=%s doc=%s: %s",
            cik, accession, primary_doc, exc,
        )
        return None


# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _get_minio(endpoint, access_key, secret_key, secure) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _object_exists(client: Minio, bucket: str, name: str) -> bool:
    try:
        client.stat_object(bucket, name)
        return True
    except S3Error:
        return False


def _put_text(client: Minio, bucket: str, object_name: str, text: str) -> int:
    data = text.encode("utf-8")
    client.put_object(
        bucket, object_name,
        data=io.BytesIO(data),
        length=len(data),
        content_type="text/plain; charset=utf-8",
    )
    return len(data)


# ---------------------------------------------------------------------------
# Main per-company logic
# ---------------------------------------------------------------------------

def ingest_company_text(
    cik: str,
    client: Minio,
    bucket: str,
    dry_run: bool = False,
    max_filings: int = 3,
) -> list[dict]:
    """
    For one CIK: find recent 10-Ks → download → extract sections → store in MinIO.
    Returns a list of result dicts (one per section stored).
    """
    log = logging.getLogger("fetch_text")
    filings = _get_recent_annual_filings(cik, max_filings=max_filings)
    if not filings:
        return [{"cik": cik, "status": "no_filings"}]

    results = []
    for filing in filings:
        acc      = filing["accession"]
        acc_dash = filing["accession_dashed"]
        form     = filing["form"]
        date_    = filing["date"]

        log.info("  %s  %s  (%s)", form, acc_dash, date_)

        # Rate-limit: SEC allows ~10 req/sec
        time.sleep(0.15)

        raw_text = _fetch_filing_text(cik, acc, filing["primary_doc"])
        if not raw_text:
            results.append({"cik": cik, "accession": acc, "status": "fetch_failed"})
            continue

        for section in ["mda", "risk", "business"]:
            obj_name = f"{TEXT_PREFIX}/CIK{cik.zfill(10)}_{acc}_{section}.txt"

            if not dry_run and _object_exists(client, bucket, obj_name):
                results.append({
                    "cik": cik, "accession": acc, "section": section,
                    "status": "already_exists", "bytes": 0,
                })
                continue

            extracted = _extract_section(raw_text, section)
            if extracted is None:
                results.append({
                    "cik": cik, "accession": acc, "section": section,
                    "status": "section_not_found", "bytes": 0,
                })
                log.debug("    section %s not found", section)
                continue

            if dry_run:
                log.info("    [DRY RUN] Would store %s — %d chars", section, len(extracted))
                results.append({
                    "cik": cik, "accession": acc, "section": section,
                    "status": "dry_run", "bytes": len(extracted),
                })
                continue

            size = _put_text(client, bucket, obj_name, extracted)
            log.info("    stored %s → %s  (%d KB)", section, obj_name, size // 1024)
            results.append({
                "cik": cik, "accession": acc, "section": section,
                "status": "ok", "bytes": size,
            })

        time.sleep(0.25)   # polite pause between filings

    return results


# ---------------------------------------------------------------------------
# CLI / entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Bronze text ingestion: 10-K filing sections → MinIO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--endpoint",      default=os.getenv("MINIO_ENDPOINT",      "localhost:9000"))
    p.add_argument("--access-key",    default=os.getenv("MINIO_ACCESS_KEY",    "minioadmin"))
    p.add_argument("--secret-key",    default=os.getenv("MINIO_SECRET_KEY",    "minioadmin"))
    p.add_argument("--secure",        action="store_true",
                   default=os.getenv("MINIO_SECURE", "false").lower() == "true")
    p.add_argument("--bronze-bucket", default=os.getenv("MINIO_BRONZE_BUCKET", "bronze"))
    p.add_argument("--max-companies", type=int, default=None,
                   help="Limit number of companies (useful for testing)")
    p.add_argument("--max-filings",   type=int, default=3,
                   help="Max annual filings to pull per company")
    p.add_argument("--dry-run",       action="store_true")
    p.add_argument("--log-level",     default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.FileHandler(LOGS_DIR / f"ingestion_text_{INGESTION_DATE}.log"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    log = logging.getLogger("fetch_text")

    # Populate SEC headers
    global HEADERS
    HEADERS = {"User-Agent": os.getenv("SEC_USER_AGENT", "BDT-Project admin@example.com")}

    log.info("=" * 70)
    log.info("Bronze text ingestion — %s", INGESTION_DATE)
    log.info("=" * 70)

    universe = pd.read_csv(UNIVERSE_PATH, dtype={"cik": str})
    if args.max_companies:
        universe = universe.head(args.max_companies)
    log.info("Companies to process: %d", len(universe))

    client = _get_minio(
        args.endpoint, args.access_key, args.secret_key, args.secure
    )

    # Ensure bronze bucket exists before writing
    if not args.dry_run:
        if not client.bucket_exists(args.bronze_bucket):
            client.make_bucket(args.bronze_bucket)
            log.info("Created MinIO bucket: %s", args.bronze_bucket)
        else:
            log.info("MinIO bucket already exists: %s", args.bronze_bucket)

    all_results = []
    for cik in tqdm(universe["cik"].tolist(), desc="Fetching text", unit="company"):
        try:
            res = ingest_company_text(
                cik, client, args.bronze_bucket,
                dry_run=args.dry_run,
                max_filings=args.max_filings,
            )
            all_results.extend(res)
        except Exception as exc:
            log.exception("Unexpected error CIK=%s: %s", cik, exc)

    # Summary
    df = pd.DataFrame(all_results)
    log.info("=" * 70)
    if "status" in df.columns:
        log.info("Status breakdown:\n%s", df["status"].value_counts().to_string())
    total_mb = df.get("bytes", pd.Series([0])).sum() / (1024 * 1024)
    log.info("Total stored: %.1f MB", total_mb)
    log.info("=" * 70)

    summary_path = LOGS_DIR / f"ingestion_text_{INGESTION_DATE}_summary.csv"
    df.to_csv(summary_path, index=False)
    log.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
