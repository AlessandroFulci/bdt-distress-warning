"""
composite.build_composite
=========================
Builds a unified composite distress score by combining three signal sources:

  Component          Weight  Source
  ─────────────────────────────────────────────────────────────
  Z-Score            40 %    Gold layer  (Altman Z-Score)
  LLM text signals   40 %    Silver text (sentiment, risk, flags)
  Trend signals      20 %    Gold layer  (QoQ growth, decline flags)

All three components are normalised to [0, 1] where:
  0 = no distress signal    1 = maximum distress signal

Final composite_score = 0.40 * z_component
                      + 0.40 * llm_component   (0.50 neutral when missing)
                      + 0.20 * trend_component

Composite zones (mirroring Altman's intuition scaled to 0-1):
  composite_score > 0.60  →  distress
  0.35 – 0.60             →  grey
  < 0.35                  →  safe

AfterEmerging classification  (rule-based keyword matching):
  liquidated  — liquidation outcome
  acquired    — acquired, merged, or sold post-emergence
  survived    — continued operating independently
  refiled     — re-entered bankruptcy
  unknown     — no text or unmatched pattern

Output written to:  data/cache/composite_scores/composite.parquet
Also uploaded to MinIO gold bucket as:  composite_scores/composite.parquet

Usage
-----
  docker compose exec app python -m src.composite.build_composite
  python -m src.composite.build_composite           # outside Docker (local dev)
"""

from __future__ import annotations

import glob
import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT    = Path(__file__).resolve().parent.parent.parent
GOLD_LOCAL      = PROJECT_ROOT / "data" / "cache" / "gold_distress"
TEXT_LOCAL      = PROJECT_ROOT / "data" / "cache" / "silver_text"
COMPOSITE_LOCAL = PROJECT_ROOT / "data" / "cache" / "composite_scores"
LOGS_DIR        = PROJECT_ROOT / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Composite weights
W_ZSCORE = 0.40
W_LLM    = 0.40
W_TREND  = 0.20

# Composite zone thresholds
ZONE_DISTRESS = 0.60
ZONE_SAFE     = 0.35

log = logging.getLogger(__name__)


# =============================================================================
# Helpers
# =============================================================================

def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOGS_DIR / "composite_build.log"),
            logging.StreamHandler(),
        ],
    )


# =============================================================================
# Step 1 — Load gold data
# =============================================================================

def load_gold() -> pd.DataFrame:
    files = glob.glob(str(GOLD_LOCAL / "**" / "*.parquet"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No gold Parquet files found at {GOLD_LOCAL}. "
            "Run `python -m src.gold.build_gold` first."
        )

    import duckdb
    con = duckdb.connect()
    df = con.execute(
        f"SELECT * FROM read_parquet('{GOLD_LOCAL}/**/*.parquet', "
        "hive_partitioning=true, union_by_name=true)"
    ).df()
    con.close()

    df["period_end"] = pd.to_datetime(df["period_end"], errors="coerce")
    log.info("Gold: %d rows × %d columns", len(df), len(df.columns))
    return df


# =============================================================================
# Step 2 — Load silver text features
# =============================================================================

def load_text_features() -> pd.DataFrame:
    """Load LLM text features from local cache. Returns empty DF if not present."""
    local_files = glob.glob(str(TEXT_LOCAL / "**" / "*.parquet"), recursive=True)
    # Also check flat single file
    flat = TEXT_LOCAL / "part-0.parquet"
    if not local_files and flat.exists():
        local_files = [str(flat)]

    if not local_files:
        log.warning("No silver text features found at %s — LLM component will be neutral (0.50).", TEXT_LOCAL)
        return pd.DataFrame()

    import duckdb
    path_glob = str(TEXT_LOCAL / "**" / "*.parquet")
    con = duckdb.connect()
    try:
        df = con.execute(
            f"SELECT * FROM read_parquet('{path_glob}', union_by_name=true)"
        ).df()
    except Exception:
        # Fallback: read the flat file directly
        df = pd.concat([pd.read_parquet(f) for f in local_files], ignore_index=True)
    con.close()

    log.info("Silver text: %d rows from %d companies", len(df), df["cik"].nunique())
    return df


# =============================================================================
# Step 3 — Z-Score component  [0=safe, 1=distress]
# =============================================================================

def compute_z_component(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise Altman Z-Score to a distress signal in [0, 1].

    Mapping (inverse — high Z = low distress):
      Z ≤ 0   →  1.0  (deep distress or negative book value)
      0 < Z < 5 → 1 − Z/5  (linear: Z=1.81 → 0.638, Z=2.99 → 0.402)
      Z ≥ 5   →  0.0  (unambiguously safe)
      NULL    →  0.50 (unknown / missing financials)
    """
    z = df["altman_z_score"]
    component = np.where(
        z.isna(),        0.50,
        np.where(
            z <= 0,      1.00,
            np.where(
                z >= 5,  0.00,
                1.0 - (z / 5.0),
            )
        )
    ).clip(0.0, 1.0)
    df = df.copy()
    df["z_component"] = component.astype(float)
    return df


# =============================================================================
# Step 4 — LLM component  [0=safe, 1=distress]
# =============================================================================

_RISK_MAP = {"low": 0.0, "medium": 0.33, "high": 0.67, "critical": 1.0}


def _build_llm_per_company(text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate LLM signals per company (cik), producing a single llm_component
    score in [0, 1] per company.

    Sub-scores (all 0-1, then weighted):
      sentiment_distress  40%  → (1 − sentiment_score) / 2
      risk_score          30%  → risk_level mapped to {low:0, medium:0.33, high:0.67, critical:1}
      flag_score          30%  → weighted binary flags (going_concern heaviest)
    """
    if text_df.empty:
        return pd.DataFrame(columns=["cik", "llm_component",
                                     "avg_sentiment_score", "avg_risk_level",
                                     "any_going_concern", "any_liquidity_risk",
                                     "any_restructuring", "n_docs_analysed"])

    df = text_df.copy()
    df["cik"] = df["cik"].astype(str).str.lstrip("0").str.zfill(1)  # normalise

    # Sentiment distress: sentiment_score in [-1, +1] → distress in [0, 1]
    # score=-1 (very negative) → distress=1.0;  score=+1 → distress=0.0
    if "sentiment_score" in df.columns:
        df["_sent_distress"] = ((1.0 - df["sentiment_score"].clip(-1, 1)) / 2.0)
    else:
        df["_sent_distress"] = 0.5

    # Risk level → numeric
    if "risk_level" in df.columns:
        df["_risk_num"] = df["risk_level"].str.lower().map(_RISK_MAP).fillna(0.33)
    else:
        df["_risk_num"] = 0.33

    # Binary distress flags (ensure they exist; default 0)
    flag_cols = {
        "going_concern":           0.35,
        "going_concern_explicit":  0.25,  # MDA-specific
        "liquidity_risk":          0.20,
        "covenant_breach_mentioned": 0.10,
        "restructuring":           0.10,
    }
    for col in flag_cols:
        if col not in df.columns:
            df[col] = 0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Weighted flag score per row (capped at 1)
    df["_flag_score"] = sum(
        df[col].clip(0, 1) * w for col, w in flag_cols.items()
    ).clip(0, 1)

    # Aggregate per CIK
    agg = df.groupby("cik").agg(
        _sent_distress=("_sent_distress", "mean"),
        _risk_num=("_risk_num", "mean"),
        _flag_score=("_flag_score", "max"),        # max: if ANY doc has a flag, it counts
        avg_sentiment_score=("sentiment_score", "mean") if "sentiment_score" in df.columns else ("_sent_distress", "mean"),
        any_going_concern=("going_concern", "max"),
        any_liquidity_risk=("liquidity_risk", "max"),
        any_restructuring=("restructuring", "max"),
        n_docs_analysed=("cik", "count"),
    ).reset_index()

    # Final LLM component: 40% sentiment + 30% risk + 30% flags
    agg["llm_component"] = (
        0.40 * agg["_sent_distress"]
      + 0.30 * agg["_risk_num"]
      + 0.30 * agg["_flag_score"]
    ).clip(0.0, 1.0)

    # Average risk level label for display
    agg["avg_risk_level"] = agg["_risk_num"].apply(
        lambda v: "low" if v < 0.2 else ("medium" if v < 0.5 else ("high" if v < 0.85 else "critical"))
    )

    log.info("LLM component built for %d companies (mean=%.3f)",
             len(agg), agg["llm_component"].mean())
    return agg[["cik", "llm_component", "avg_sentiment_score", "avg_risk_level",
                "any_going_concern", "any_liquidity_risk", "any_restructuring",
                "n_docs_analysed"]]


# =============================================================================
# Step 5 — Trend component  [0=no trend signal, 1=strong decline]
# =============================================================================

def compute_trend_component(df: pd.DataFrame) -> pd.DataFrame:
    """
    Combines QoQ growth rates and 3-quarter decline flags into a
    trend distress component in [0, 1].

      revenue_declining_3q      40%  (consecutive 3Q revenue decline)
      net_income_declining_3q   30%  (consecutive 3Q profit decline)
      revenue_growth < -10%     20%  (sharp single-period revenue drop)
      net_income_growth < -15%  10%  (sharp single-period profit drop)
    """
    df = df.copy()

    flag_rev = df.get("revenue_declining_3q", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)
    flag_ni  = df.get("net_income_declining_3q", pd.Series(0, index=df.index)).fillna(0).clip(0, 1)

    rev_drop = (
        df.get("revenue_growth_qoq", pd.Series(np.nan, index=df.index))
          .fillna(0).lt(-0.10).astype(float)
    )
    ni_drop  = (
        df.get("net_income_growth_qoq", pd.Series(np.nan, index=df.index))
          .fillna(0).lt(-0.15).astype(float)
    )

    df["trend_component"] = (
        0.40 * flag_rev
      + 0.30 * flag_ni
      + 0.20 * rev_drop
      + 0.10 * ni_drop
    ).clip(0.0, 1.0)

    return df


# =============================================================================
# Step 6 — AfterEmerging classification (rule-based keyword matching)
# =============================================================================

_AFTER_RULES: list[tuple[str, str]] = [
    (r"liquidat",                    "liquidated"),
    (r"acqui|purchas|merger|sold|sale|absorbed", "acquired"),
    (r"second.{0,10}bankrupt|refil",  "refiled"),
    (r"emerged|operating|surviv|continu|still|independent", "survived"),
]


def classify_after_emerging(text: str | float) -> str:
    """Classify the LoPucki AfterEmerging free-text into a canonical outcome."""
    if not isinstance(text, str) or not text.strip():
        return "unknown"
    t = text.lower()
    for pattern, label in _AFTER_RULES:
        if re.search(pattern, t):
            return label
    return "unknown"


# =============================================================================
# Step 7 — Join + final composite
# =============================================================================

def build_composite(gold_df: pd.DataFrame, text_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce the composite distress score table with one row per
    (company, period_end) — same granularity as the gold layer.
    """
    # --- Z-Score component ---
    df = compute_z_component(gold_df)

    # --- Trend component ---
    df = compute_trend_component(df)

    # --- LLM component (per company, joined by cik) ---
    llm_per_co = _build_llm_per_company(text_df)

    # Normalise CIK for join (both sides: strip leading zeros → bare string)
    df["_cik_join"] = df["cik"].astype(str).str.lstrip("0").str.zfill(1)
    llm_per_co["_cik_join"] = llm_per_co["cik"].astype(str).str.lstrip("0").str.zfill(1)

    df = df.merge(
        llm_per_co[["_cik_join", "llm_component", "avg_sentiment_score",
                    "avg_risk_level", "any_going_concern", "any_liquidity_risk",
                    "any_restructuring", "n_docs_analysed"]],
        on="_cik_join", how="left",
    )
    df.drop(columns=["_cik_join"], inplace=True)

    # Fill missing LLM component with 0.50 (neutral — no text data available)
    df["llm_component"]       = df["llm_component"].fillna(0.50)
    df["avg_sentiment_score"] = df["avg_sentiment_score"].fillna(0.0)
    df["avg_risk_level"]      = df["avg_risk_level"].fillna("unknown")
    df["any_going_concern"]   = df["any_going_concern"].fillna(0).astype(int)
    df["any_liquidity_risk"]  = df["any_liquidity_risk"].fillna(0).astype(int)
    df["any_restructuring"]   = df["any_restructuring"].fillna(0).astype(int)
    df["n_docs_analysed"]     = df["n_docs_analysed"].fillna(0).astype(int)

    # --- Composite score ---
    df["composite_score"] = (
        W_ZSCORE * df["z_component"]
      + W_LLM    * df["llm_component"]
      + W_TREND  * df["trend_component"]
    ).clip(0.0, 1.0)

    # --- Composite zone ---
    df["composite_zone"] = df["composite_score"].apply(
        lambda s: "distress" if s > ZONE_DISTRESS else ("safe" if s < ZONE_SAFE else "grey")
        if pd.notna(s) else None
    )

    # --- AfterEmerging classification ---
    if "brd_after_emerging" in df.columns:
        df["after_emerging_class"] = df["brd_after_emerging"].apply(classify_after_emerging)
    else:
        df["after_emerging_class"] = "unknown"

    log.info("Composite table: %d rows × %d columns", len(df), len(df.columns))
    log.info("Composite zone distribution:\n%s",
             df["composite_zone"].value_counts().to_string())
    log.info("Composite score stats:\n%s",
             df["composite_score"].describe().to_string())

    return df


# =============================================================================
# Step 8 — Write output
# =============================================================================

def write_composite(df: pd.DataFrame):
    COMPOSITE_LOCAL.mkdir(parents=True, exist_ok=True)
    out_path = COMPOSITE_LOCAL / "composite.parquet"
    df.to_parquet(out_path, index=False, engine="pyarrow")
    log.info("Wrote %d rows → %s", len(df), out_path)


def upload_composite_to_minio(df: pd.DataFrame):
    """Upload composite Parquet to MinIO gold bucket."""
    try:
        import io
        import pyarrow as pa
        import pyarrow.parquet as pq
        from minio import Minio

        client = Minio(
            os.getenv("MINIO_ENDPOINT", "localhost:9000"),
            access_key=os.getenv("MINIO_ACCESS_KEY", "minioadmin"),
            secret_key=os.getenv("MINIO_SECRET_KEY", "minioadmin"),
            secure=os.getenv("MINIO_SECURE", "false") == "true",
        )
        gold_bucket = os.getenv("MINIO_GOLD_BUCKET", "gold")
        if not client.bucket_exists(gold_bucket):
            client.make_bucket(gold_bucket)

        buf = io.BytesIO()
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), buf, compression="snappy")
        data = buf.getvalue()
        client.put_object(
            gold_bucket, "composite_scores/composite.parquet",
            data=io.BytesIO(data), length=len(data),
            content_type="application/octet-stream",
        )
        log.info("Uploaded composite → MinIO %s/composite_scores/composite.parquet", gold_bucket)
    except Exception as exc:
        log.warning("MinIO upload skipped: %s", exc)


# =============================================================================
# Main
# =============================================================================

def main():
    _setup_logging()
    log.info("=" * 70)
    log.info("Composite distress score build — start")
    log.info("  Weights: Z-Score %.0f%%  LLM %.0f%%  Trend %.0f%%",
             W_ZSCORE * 100, W_LLM * 100, W_TREND * 100)
    log.info("=" * 70)

    log.info("Step 1/4: Loading gold data...")
    gold_df = load_gold()

    log.info("Step 2/4: Loading LLM text features...")
    text_df = load_text_features()
    llm_available = not text_df.empty
    if not llm_available:
        log.info("  ⚠️  No LLM data — llm_component will be 0.50 (neutral) for all rows")

    log.info("Step 3/4: Building composite scores...")
    composite_df = build_composite(gold_df, text_df)

    log.info("Step 4/4: Writing output...")
    write_composite(composite_df)
    upload_composite_to_minio(composite_df)

    log.info("=" * 70)
    log.info("Composite build complete!")
    log.info("  Rows:            %d", len(composite_df))
    log.info("  Companies:       %d", composite_df["cik"].nunique())
    log.info("  LLM data used:   %s", "yes" if llm_available else "no (neutral fallback)")
    log.info("  Output:          %s", COMPOSITE_LOCAL / "composite.parquet")
    log.info("=" * 70)

    # Quick distress preview
    top_distress = (
        composite_df[composite_df["composite_zone"] == "distress"]
        .sort_values("composite_score", ascending=False)
        [["company_name", "period_end", "composite_score", "z_component",
          "llm_component", "trend_component", "composite_zone"]]
        .drop_duplicates(subset="company_name")
        .head(10)
    )
    if not top_distress.empty:
        log.info("Top distress companies:\n%s", top_distress.to_string(index=False))


if __name__ == "__main__":
    main()
