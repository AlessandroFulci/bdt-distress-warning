"""
run_pipeline.py  —  Smart pipeline runner for BDT Distress Warning System
==========================================================================
Checks what data already exists on disk / in MinIO and only runs the steps
that are actually needed. You never have to re-fetch bronze data unless you
want to update it.

Usage
-----
  python run_pipeline.py             # run everything that is out of date
  python run_pipeline.py --dashboard # run pipeline (if needed) then launch dashboard
  python run_pipeline.py --force     # force re-run all steps
  python run_pipeline.py --status    # show what data exists, then exit
  python run_pipeline.py --step gold # run only one specific step

Steps (in order)
-----------------
  bronze      fetch_bronze.py      — SEC EDGAR JSON → MinIO bronze
  bronze_text fetch_bronze_text.py — 10-K text sections → MinIO bronze  (optional)
  silver      build_silver.py      — bronze JSON → silver Parquet
  gold        build_gold.py        — silver Parquet → gold Z-scores
  silver_text build_silver_text.py — bronze text → LLM NLP features      (optional)
  composite   build_composite.py   — Z-Score + LLM + Trend → composite score
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_DIR        = PROJECT_ROOT / "data" / "cache"
GOLD_LOCAL       = CACHE_DIR / "gold_distress"
SILVER_LOCAL     = CACHE_DIR / "silver_financials"
COMPOSITE_LOCAL  = CACHE_DIR / "composite_scores"
LOGS_DIR     = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("run_pipeline")

# How old (in days) gold/silver cache can be before we consider it stale
CACHE_MAX_AGE_DAYS = 7


# ---------------------------------------------------------------------------
# Cache / staleness checks
# ---------------------------------------------------------------------------

def _parquet_files(directory: Path) -> list[Path]:
    return list(directory.rglob("*.parquet")) if directory.exists() else []


def _newest_mtime(files: list[Path]) -> float | None:
    if not files:
        return None
    return max(f.stat().st_mtime for f in files)


def _age_days(mtime: float) -> float:
    return (datetime.now().timestamp() - mtime) / 86400


def composite_is_fresh() -> bool:
    f = COMPOSITE_LOCAL / "composite.parquet"
    if not f.exists():
        return False
    age = _age_days(f.stat().st_mtime)
    log.info("Composite cache: %.1f days old", age)
    return age < CACHE_MAX_AGE_DAYS


def gold_is_fresh() -> bool:
    files = _parquet_files(GOLD_LOCAL)
    if not files:
        return False
    age = _age_days(_newest_mtime(files))
    log.info("Gold cache: %d files, %.1f days old", len(files), age)
    return age < CACHE_MAX_AGE_DAYS


def silver_is_fresh() -> bool:
    files = _parquet_files(SILVER_LOCAL)
    if not files:
        return False
    age = _age_days(_newest_mtime(files))
    log.info("Silver cache: %d files, %.1f days old", len(files), age)
    return age < CACHE_MAX_AGE_DAYS


def bronze_exists_in_minio() -> bool:
    """Quick check: can we connect to MinIO and does the bronze bucket have objects?"""
    try:
        from minio import Minio
        from minio.error import S3Error
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false") == "true",
        )
        bucket = os.getenv("MINIO_BRONZE_BUCKET", "bronze")
        if not client.bucket_exists(bucket):
            return False
        # Just check there is at least one object
        objects = list(client.list_objects(bucket, recursive=False))
        return len(objects) > 0
    except Exception as exc:
        log.warning("MinIO check failed: %s", exc)
        return False


def silver_exists_in_minio() -> bool:
    try:
        from minio import Minio
        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false") == "true",
        )
        bucket = os.getenv("MINIO_SILVER_BUCKET", "silver")
        if not client.bucket_exists(bucket):
            return False
        objects = list(client.list_objects(bucket, prefix="financial_facts/", recursive=False))
        return len(objects) > 0
    except Exception as exc:
        log.warning("MinIO silver check failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Step runner
# ---------------------------------------------------------------------------

def run_step(module: str, extra_args: list[str] | None = None) -> bool:
    """
    Run a pipeline step as a Python module subprocess.
    Returns True on success, False on failure.
    """
    cmd = [sys.executable, "-m", module] + (extra_args or [])
    log.info("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    if result.returncode != 0:
        log.error("Step FAILED: %s (exit code %d)", module, result.returncode)
        return False
    log.info("Step OK: %s", module)
    return True


# ---------------------------------------------------------------------------
# Status display
# ---------------------------------------------------------------------------

def print_status():
    print("\n" + "=" * 55)
    print("  BDT Distress Warning — Pipeline Status")
    print("=" * 55)

    # Bronze (MinIO)
    bronze_ok = bronze_exists_in_minio()
    print(f"  Bronze (MinIO)   : {'✅ data found' if bronze_ok else '❌ empty or unreachable'}")

    # Silver (MinIO)
    silver_minio = silver_exists_in_minio()
    print(f"  Silver (MinIO)   : {'✅ data found' if silver_minio else '❌ not built yet'}")

    # Silver (local cache)
    sf = _parquet_files(SILVER_LOCAL)
    if sf:
        age = _age_days(_newest_mtime(sf))
        fresh = age < CACHE_MAX_AGE_DAYS
        print(f"  Silver (cache)   : ✅ {len(sf)} files, {age:.1f}d old {'(fresh)' if fresh else '(STALE)'}")
    else:
        print(f"  Silver (cache)   : ❌ {SILVER_LOCAL}")

    # Gold (local cache)
    gf = _parquet_files(GOLD_LOCAL)
    if gf:
        age = _age_days(_newest_mtime(gf))
        fresh = age < CACHE_MAX_AGE_DAYS
        print(f"  Gold (cache)     : ✅ {len(gf)} files, {age:.1f}d old {'(fresh)' if fresh else '(STALE)'}")
    else:
        print(f"  Gold (cache)     : ❌ {GOLD_LOCAL}")

    # Composite (local)
    cf = COMPOSITE_LOCAL / "composite.parquet"
    if cf.exists():
        age = _age_days(cf.stat().st_mtime)
        fresh = age < CACHE_MAX_AGE_DAYS
        print(f"  Composite score  : ✅ {age:.1f}d old {'(fresh)' if fresh else '(STALE)'}")
    else:
        print(f"  Composite score  : ❌ not built — run --step composite")

    print("=" * 55)
    if gf:
        print("  ✅ Dashboard is ready:  streamlit run src/dashboard/app.py")
    else:
        print("  ⚠️  Run `python run_pipeline.py` to build the data first.")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# Main orchestration logic
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Smart pipeline runner — skips steps already completed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py               # auto: run only what's missing
  python run_pipeline.py --dashboard   # run pipeline then launch dashboard
  python run_pipeline.py --status      # check what data exists
  python run_pipeline.py --force       # rebuild everything from scratch
  python run_pipeline.py --step gold   # run only the gold step
  python run_pipeline.py --no-bronze   # skip bronze fetch (use existing MinIO data)
        """,
    )
    parser.add_argument("--status",     action="store_true",  help="Show pipeline status and exit")
    parser.add_argument("--force",      action="store_true",  help="Re-run all steps even if data exists")
    parser.add_argument("--dashboard",  action="store_true",  help="Launch Streamlit after pipeline")
    parser.add_argument("--no-bronze",  action="store_true",  help="Skip bronze ingestion entirely")
    parser.add_argument("--with-text",  action="store_true",  help="Also run bronze_text + silver_text (LLM)")
    parser.add_argument("--step",       choices=["bronze", "bronze_text", "silver", "gold", "silver_text", "composite"],
                        help="Run only this one step")
    args = parser.parse_args()

    if args.status:
        print_status()
        return

    log.info("=" * 55)
    log.info("BDT Distress Warning — Pipeline Runner")
    log.info("=" * 55)

    # --- Run a single named step if requested ---
    if args.step:
        step_map = {
            "bronze":      ("src.ingestion.fetch_bronze",      []),
            "bronze_text": ("src.ingestion.fetch_bronze_text", []),
            "silver":      ("src.silver.build_silver",         []),
            "gold":        ("src.gold.build_gold",             []),
            "silver_text": ("src.silver.build_silver_text",    []),
            "composite":   ("src.composite.build_composite",   []),
        }
        module, extra = step_map[args.step]
        ok = run_step(module, extra)
        sys.exit(0 if ok else 1)

    # --- Smart auto-run ---
    success = True

    # STEP 1: Bronze ingestion
    if not args.no_bronze:
        bronze_ok = bronze_exists_in_minio()
        if args.force or not bronze_ok:
            log.info("--- Step 1/4: Bronze ingestion ---")
            success = run_step("src.ingestion.fetch_bronze") and success
        else:
            log.info("--- Step 1/4: Bronze ✅ already in MinIO — skipping ---")
    else:
        log.info("--- Step 1/4: Bronze skipped (--no-bronze) ---")

    # STEP 1b: Bronze text (optional)
    if args.with_text:
        log.info("--- Step 1b: Bronze text ingestion ---")
        success = run_step("src.ingestion.fetch_bronze_text") and success

    # STEP 2: Silver build
    if args.force or not silver_is_fresh():
        # Also skip if silver is already in MinIO and we just need local cache
        if not args.force and silver_exists_in_minio() and _parquet_files(SILVER_LOCAL):
            log.info("--- Step 2/4: Silver ✅ cache exists — skipping ---")
        else:
            log.info("--- Step 2/4: Silver build ---")
            success = run_step("src.silver.build_silver") and success
    else:
        log.info("--- Step 2/4: Silver ✅ cache fresh — skipping ---")

    # STEP 3: Gold build
    if args.force or not gold_is_fresh():
        log.info("--- Step 3/4: Gold build ---")
        success = run_step("src.gold.build_gold") and success
    else:
        log.info("--- Step 3/4: Gold ✅ cache fresh — skipping ---")

    # STEP 4: Silver text / LLM (optional)
    if args.with_text:
        log.info("--- Step 4/5: Silver text (LLM) ---")
        success = run_step("src.silver.build_silver_text") and success

    # STEP 5: Composite score
    if args.force or not composite_is_fresh():
        log.info("--- Step 5/5: Composite distress score ---")
        success = run_step("src.composite.build_composite") and success
    else:
        log.info("--- Step 5/5: Composite ✅ cache fresh — skipping ---")

    # --- Summary ---
    log.info("=" * 55)
    if success:
        log.info("Pipeline complete ✅")
        print_status()
    else:
        log.error("Pipeline finished with errors ❌  — check logs above")
        sys.exit(1)

    # --- Launch dashboard ---
    if args.dashboard:
        log.info("Launching Streamlit dashboard...")
        subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "src/dashboard/app.py"],
            cwd=str(PROJECT_ROOT),
        )


if __name__ == "__main__":
    main()
