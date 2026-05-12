"""
silver.build_silver_text
========================
LLM-powered silver pipeline for textual financial distress signals.

Reads 10-K/10-Q text sections (MD&A, Risk Factors, Business) from the
MinIO bronze bucket, processes each through a locally-running LLM via
Ollama, and writes structured NLP features to the silver bucket as Parquet.

Supported models (via Ollama)
-----------------------------
  llama3.2:3b    — fast, CPU-friendly (~2GB RAM)   ← recommended default
  llama3.2:8b    — better quality, needs ~6GB RAM
  minicpm3:4b    — compact, multilingual-friendly  (~3GB RAM)
  minicpm-v:8b   — vision-capable variant

Model recommendation for this use case
---------------------------------------
  • LLaMA 3.2 3B  → best for machines with ≤8 GB RAM; good English financial text
  • LLaMA 3.2 8B  → best quality for distress signal extraction on 16GB+ RAM
  • MiniCPM-3 4B  → good alternative if you need multilingual (IFRS filers)
                    or if you have <6GB VRAM and want better quality than 3B

Ollama setup
------------
  1. Install: https://ollama.com
  2. Pull model:
       ollama pull llama3.2:3b          # or minicpm3:4b
  3. Run server (usually auto-starts):
       ollama serve
  4. Verify: curl http://localhost:11434/api/tags

Features extracted per document
---------------------------------
  sentiment_score    float  -1.0 (very negative) … +1.0 (very positive)
  sentiment_label    str    negative | neutral | positive
  distress_keywords  int    count of distress-related terms in raw text
  risk_level         str    low | medium | high | critical
  going_concern      int    1 if going-concern language detected, else 0
  liquidity_risk     int    1 if liquidity / cash / covenant risk mentioned
  restructuring      int    1 if restructuring / layoff / asset sale mentioned
  llm_summary        str    1-sentence LLM summary of the section
  llm_raw            str    full JSON string from LLM (for debugging)

Output schema (silver bucket)
-------------------------------
  edgar_text_features/
    section=<mda|risk|business>/
      part-0.parquet

Usage
-----
  python -m src.silver.build_silver_text
  python -m src.silver.build_silver_text --model minicpm3:4b --dry-run --limit 20

Environment variables
---------------------
  MINIO_ENDPOINT, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_SECURE
  MINIO_BRONZE_BUCKET  (default: bronze)
  MINIO_SILVER_BUCKET  (default: silver)
  OLLAMA_BASE_URL      (default: http://localhost:11434)
  OLLAMA_MODEL         (default: llama3.2:3b)
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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
from minio import Minio
from minio.error import S3Error
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
load_dotenv()

log = logging.getLogger(__name__)


# =============================================================================
# Config & constants
# =============================================================================

SECTIONS     = ["mda", "risk", "business"]
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
OLLAMA_BASE   = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LOCAL_TEXT_CACHE = PROJECT_ROOT / "data" / "cache" / "silver_text"

# Simple keyword lists for fast pre-LLM feature extraction
DISTRESS_KEYWORDS = [
    "going concern", "bankruptcy", "default", "insolvency", "insolvent",
    "restructuring", "debt covenant", "covenant violation", "liquidity risk",
    "unable to repay", "material doubt", "substantial doubt", "delisting",
    "impairment", "write-off", "write-down", "negative working capital",
    "cash burn", "runway", "forbearance", "receivership", "chapter 11",
    "chapter 7", "administration", "wind down", "wind-down", "cessation",
    "going-concern", "noncompliance", "breach of covenant", "waiver",
]

POSITIVE_KEYWORDS = [
    "record revenue", "strong growth", "exceeded expectations", "profitability",
    "cash flow positive", "expanding", "growth momentum", "robust demand",
    "market leadership", "cost reduction", "efficiency gains",
]


# =============================================================================
# Keyword-based feature extraction  (fast, no LLM)
# =============================================================================

def extract_keyword_features(text: str) -> dict[str, Any]:
    """
    Count distress keywords and derive binary flags.
    This runs before the LLM to avoid wasting inference on empty sections.
    """
    lower = text.lower()
    distress_count = sum(1 for kw in DISTRESS_KEYWORDS if kw in lower)
    positive_count = sum(1 for kw in POSITIVE_KEYWORDS if kw in lower)

    going_concern  = int("going concern" in lower or "going-concern" in lower
                         or "substantial doubt" in lower or "material doubt" in lower)
    liquidity_risk = int(
        "liquidity" in lower or "cash covenant" in lower
        or "covenant" in lower or "cash burn" in lower
        or "unable to repay" in lower
    )
    restructuring  = int(
        "restructuring" in lower or "layoff" in lower
        or "reduction in force" in lower or "rif" in lower
        or "asset sale" in lower or "divestiture" in lower
    )

    return {
        "distress_keywords": distress_count,
        "positive_keywords": positive_count,
        "going_concern":     going_concern,
        "liquidity_risk":    liquidity_risk,
        "restructuring":     restructuring,
    }


# =============================================================================
# LLM prompts  — section-specific, calibrated for financial distress detection
# =============================================================================

SYSTEM_PROMPT = """You are a CFA-level credit analyst specialising in bankruptcy prediction \
and financial distress. You analyse SEC 10-K filing language to surface early warning signals \
that supplement quantitative models like the Altman Z-Score.

Your job is to detect what management is NOT saying as much as what they ARE saying — \
hedging language, omissions, tone shifts, and buried disclosures matter.

Respond ONLY with a valid JSON object. No prose, no markdown fences, no explanation outside the JSON."""


# ---------------------------------------------------------------------------
# Calibration examples embedded in each prompt so small models have anchors
# ---------------------------------------------------------------------------

_SENTIMENT_SCALE = """
Sentiment score calibration (use these as anchors):
  -1.0  "The company has substantial doubt about its ability to continue as a going concern."
  -0.7  "We face significant liquidity constraints and may be unable to meet debt obligations."
  -0.4  "Revenue declined 18%% year-over-year; management expects continued pressure."
   0.0  "Results were in line with prior year with no material changes to liquidity position."
  +0.4  "Cash flow from operations increased 12%%; debt levels remain manageable."
  +0.7  "Record revenue and strong free cash flow generation; net debt reduced by 30%%."
  +1.0  "Exceptional growth, debt-free balance sheet, industry-leading margins."
""".strip()


def _make_mda_prompt(text: str) -> str:
    """
    MD&A prompt: focuses on management's candour about liquidity, covenant
    compliance, debt maturity, and tone relative to financial reality.
    """
    snippet = text[:4000].replace('"', "'")
    return f"""Analyse the following MD&A section from a US public company's 10-K filing.

{_SENTIMENT_SCALE}

TEXT:
\"\"\"{snippet}\"\"\"

Extract these signals and return ONLY this JSON object:
{{
  "sentiment_score":             <float -1.0 to +1.0 using the scale above>,
  "sentiment_label":             <"negative" | "neutral" | "positive">,
  "management_tone":             <"evasive" | "cautious" | "neutral" | "confident" | "optimistic">,
  "liquidity_signal":            <"critical" | "stressed" | "adequate" | "strong">,
  "going_concern_explicit":      <true if the phrase 'going concern' or 'substantial doubt' or 'material doubt' appears, else false>,
  "covenant_breach_mentioned":   <true if debt covenant violation or waiver is disclosed, else false>,
  "debt_restructuring_mentioned":<true if debt restructuring, forbearance, or refinancing difficulty is mentioned, else false>,
  "cash_burn_mentioned":         <true if the company discusses negative cash flow, cash burn rate, or limited cash runway, else false>,
  "key_distress_signal":         <one sentence: the single most critical distress indicator found, or "none detected">,
  "summary":                     <one sentence, max 40 words: overall financial health assessment from this MD&A>
}}

Scoring rules:
- management_tone "evasive": vague language, excessive qualifiers, avoids quantifying problems
- liquidity_signal "critical": going-concern language OR < 3 months implied cash runway
- liquidity_signal "stressed": covenant risk, refinancing urgency, or declining cash with no clear plan
- Return ONLY the JSON, nothing else."""


def _make_risk_prompt(text: str) -> str:
    """
    Risk Factors prompt: focuses on the severity and specificity of disclosed
    risks, especially liquidity, debt maturity, and concentration risks.
    """
    snippet = text[:4000].replace('"', "'")
    return f"""Analyse the following Risk Factors section from a US public company's 10-K filing.

{_SENTIMENT_SCALE}

TEXT:
\"\"\"{snippet}\"\"\"

Extract these signals and return ONLY this JSON object:
{{
  "sentiment_score":               <float -1.0 to +1.0; more negative = more severe disclosed risks>,
  "risk_level":                    <"low" | "medium" | "high" | "critical">,
  "liquidity_risk_disclosed":      <true if inability to fund operations, meet debt payments, or access capital is a stated risk>,
  "going_concern_risk_disclosed":  <true if ability to continue as a going concern is listed as a risk>,
  "debt_maturity_risk":            <true if near-term debt maturity the company may struggle to refinance is mentioned>,
  "customer_concentration_risk":   <true if loss of one or a few major customers is listed as a material risk>,
  "covenant_violation_risk":       <true if risk of breaching financial covenants is disclosed>,
  "severity_language":             <"speculative" | "possible" | "probable" | "certain" — based on modal verbs used (could/may vs will/has)>,
  "top_risk_category":             <"liquidity" | "debt_maturity" | "market" | "operational" | "legal" | "regulatory" | "competitive" | "macro" | "other">,
  "key_distress_signal":           <one sentence: the most severe risk factor disclosed, or "no critical risks">,
  "summary":                       <one sentence, max 40 words: overall risk profile assessment>
}}

Scoring rules:
- risk_level "critical": going-concern risk OR liquidity/debt risk that could prevent continued operations
- risk_level "high": multiple severe risks OR one near-term debt/liquidity risk
- severity_language "certain": uses "will", "has", "is unable to" — not conditional
- Return ONLY the JSON, nothing else."""


def _make_business_prompt(text: str) -> str:
    """
    Business Description prompt: focuses on revenue model sustainability,
    concentration risk, competitive moat, and operational fragility.
    """
    snippet = text[:4000].replace('"', "'")
    return f"""Analyse the following Business Description section from a US public company's 10-K filing.

{_SENTIMENT_SCALE}

TEXT:
\"\"\"{snippet}\"\"\"

Extract these signals and return ONLY this JSON object:
{{
  "sentiment_score":              <float -1.0 to +1.0; reflects strength and sustainability of the business model>,
  "business_model_viability":     <"strong" | "adequate" | "fragile" | "critical">,
  "competitive_position":         <"dominant" | "strong" | "adequate" | "weak" | "deteriorating">,
  "revenue_concentration_risk":   <true if the business depends heavily on one product, geography, or customer type>,
  "high_capex_or_debt_dependent": <true if the business requires continuous heavy capital investment or debt to operate>,
  "regulatory_dependency":        <true if a key operating license, government contract, or regulatory approval is essential and at risk>,
  "key_vulnerability":            <one sentence: the main structural weakness or dependency in this business, or "none identified">,
  "summary":                      <one sentence, max 40 words: overall business model strength assessment>
}}

Scoring rules:
- business_model_viability "fragile": single product, single geography, or single customer dependency
- business_model_viability "critical": business requires external funding to continue AND has no clear path to profitability
- competitive_position "deteriorating": language implies losing market share or pricing power
- Return ONLY the JSON, nothing else."""


def _make_user_prompt(section: str, text: str) -> str:
    """Route to the correct section-specific prompt."""
    if section == "mda":
        return _make_mda_prompt(text)
    elif section == "risk":
        return _make_risk_prompt(text)
    elif section == "business":
        return _make_business_prompt(text)
    else:
        # Fallback for unknown sections
        snippet = text[:3000].replace('"', "'")
        return f"""Analyse this SEC filing section and return ONLY this JSON:
{{
  "sentiment_score": <float -1.0 to +1.0>,
  "sentiment_label": <"negative"|"neutral"|"positive">,
  "risk_level": <"low"|"medium"|"high"|"critical">,
  "key_distress_signal": <one sentence or "none detected">,
  "summary": <one sentence>
}}
TEXT: \"\"\"{snippet}\"\"\""""


# =============================================================================
# Ollama client
# =============================================================================

class OllamaClient:
    """
    Thin wrapper around the Ollama HTTP API.
    Uses /api/generate (single-turn completion) — compatible with all models.
    """

    def __init__(self, base_url: str = OLLAMA_BASE, model: str = DEFAULT_MODEL,
                 timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.model    = model
        self.timeout  = timeout
        self._session = requests.Session()

    def health_check(self) -> bool:
        """Return True if Ollama is reachable and the model is available."""
        try:
            r = self._session.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False
            tags = [m["name"] for m in r.json().get("models", [])]
            # Accept prefix match (e.g. "llama3.2:3b" matches "llama3.2:3b-instruct-q4_0")
            return any(t.startswith(self.model.split(":")[0]) for t in tags) or self.model in tags
        except Exception:
            return False

    def generate(self, system: str, user: str, temperature: float = 0.1) -> str:
        """
        Call Ollama generate endpoint. Returns the model's text response.
        Raises RuntimeError on failure.
        """
        payload = {
            "model":  self.model,
            "prompt": f"<|system|>\n{system}\n<|user|>\n{user}\n<|assistant|>",
            "stream": False,
            "options": {
                "temperature":   temperature,
                "num_predict":   512,   # enough for richer section-specific JSON
                "top_p":         0.9,
                "repeat_penalty": 1.1,
            },
        }
        try:
            r = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout,
            )
            r.raise_for_status()
            return r.json()["response"]
        except Exception as exc:
            raise RuntimeError(f"Ollama generate failed: {exc}") from exc

    def chat(self, system: str, user: str, temperature: float = 0.1) -> str:
        """
        Use /api/chat (messages format) — works better with instruct models.
        Falls back to generate() if chat endpoint returns 404.
        """
        payload = {
            "model":  self.model,
            "stream": False,
            "options": {"temperature": temperature, "num_predict": 512},
            "messages": [
                {"role": "system",    "content": system},
                {"role": "user",      "content": user},
            ],
        }
        try:
            r = self._session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout,
            )
            if r.status_code == 404:
                return self.generate(system, user, temperature)
            r.raise_for_status()
            return r.json()["message"]["content"]
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Ollama chat failed: {exc}") from exc


# =============================================================================
# LLM feature extraction
# =============================================================================

def _parse_llm_json(raw: str) -> dict:
    """Extract a JSON object from potentially noisy LLM output."""
    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {}


def _safe_bool(val: Any, default: bool = False) -> int:
    """Coerce LLM boolean output to 0/1 int."""
    if isinstance(val, bool):
        return int(val)
    if isinstance(val, int):
        return int(bool(val))
    if isinstance(val, str):
        return int(val.lower() in ("true", "yes", "1"))
    return int(default)


def _safe_enum(val: Any, choices: list[str], default: str) -> str:
    s = str(val).lower().strip() if val is not None else default
    return s if s in choices else default


def _parse_section_fields(section: str, parsed: dict) -> dict[str, Any]:
    """
    Extract section-specific fields from the parsed LLM JSON.
    Returns a flat dict with all fields for this section, with safe defaults.
    """
    # Fields shared by all sections
    score = max(-1.0, min(1.0, float(parsed.get("sentiment_score", 0.0))))
    # Derive sentiment_label from score if LLM didn't return it explicitly
    llm_label = parsed.get("sentiment_label")
    if llm_label in ("negative", "neutral", "positive"):
        label = llm_label
    elif score <= -0.2:
        label = "negative"
    elif score >= 0.2:
        label = "positive"
    else:
        label = "neutral"

    base: dict[str, Any] = {
        "sentiment_score":     score,
        "sentiment_label":     label,
        "key_distress_signal": str(parsed.get("key_distress_signal", "none detected"))[:300],
        "llm_summary":         str(parsed.get("summary", ""))[:400],
    }

    if section == "mda":
        base.update({
            "management_tone":              _safe_enum(parsed.get("management_tone"),
                                                ["evasive","cautious","neutral","confident","optimistic"], "neutral"),
            "liquidity_signal":             _safe_enum(parsed.get("liquidity_signal"),
                                                ["critical","stressed","adequate","strong"], "adequate"),
            "going_concern_explicit":       _safe_bool(parsed.get("going_concern_explicit")),
            "covenant_breach_mentioned":    _safe_bool(parsed.get("covenant_breach_mentioned")),
            "debt_restructuring_mentioned": _safe_bool(parsed.get("debt_restructuring_mentioned")),
            "cash_burn_mentioned":          _safe_bool(parsed.get("cash_burn_mentioned")),
            # derive risk_level from liquidity_signal for dashboard compatibility
            "risk_level": {
                "critical": "critical", "stressed": "high",
                "adequate": "medium",   "strong": "low",
            }.get(str(parsed.get("liquidity_signal", "adequate")).lower(), "medium"),
        })

    elif section == "risk":
        base.update({
            "risk_level":                   _safe_enum(parsed.get("risk_level"),
                                                ["low","medium","high","critical"], "medium"),
            "liquidity_risk_disclosed":     _safe_bool(parsed.get("liquidity_risk_disclosed")),
            "going_concern_risk_disclosed": _safe_bool(parsed.get("going_concern_risk_disclosed")),
            "debt_maturity_risk":           _safe_bool(parsed.get("debt_maturity_risk")),
            "customer_concentration_risk":  _safe_bool(parsed.get("customer_concentration_risk")),
            "covenant_violation_risk":      _safe_bool(parsed.get("covenant_violation_risk")),
            "severity_language":            _safe_enum(parsed.get("severity_language"),
                                                ["speculative","possible","probable","certain"], "possible"),
            "top_risk_category":            _safe_enum(parsed.get("top_risk_category"),
                                                ["liquidity","debt_maturity","market","operational",
                                                 "legal","regulatory","competitive","macro","other"], "other"),
        })

    elif section == "business":
        base.update({
            "business_model_viability":     _safe_enum(parsed.get("business_model_viability"),
                                                ["strong","adequate","fragile","critical"], "adequate"),
            "competitive_position":         _safe_enum(parsed.get("competitive_position"),
                                                ["dominant","strong","adequate","weak","deteriorating"], "adequate"),
            "revenue_concentration_risk":   _safe_bool(parsed.get("revenue_concentration_risk")),
            "high_capex_or_debt_dependent": _safe_bool(parsed.get("high_capex_or_debt_dependent")),
            "regulatory_dependency":        _safe_bool(parsed.get("regulatory_dependency")),
            "key_vulnerability":            str(parsed.get("key_vulnerability", "none identified"))[:300],
            # derive risk_level for dashboard compatibility
            "risk_level": {
                "critical": "critical", "fragile": "high",
                "adequate": "medium",   "strong": "low",
            }.get(str(parsed.get("business_model_viability", "adequate")).lower(), "medium"),
        })

    else:
        base["risk_level"] = _safe_enum(parsed.get("risk_level"),
                                        ["low","medium","high","critical"], "medium")
    return base


def extract_llm_features(
    text: str,
    section: str,
    client: OllamaClient,
) -> dict[str, Any]:
    """
    Run the LLM on one section and return structured, section-specific features.
    Falls back to safe defaults on any error.
    """
    defaults: dict[str, Any] = {
        "sentiment_score":   0.0,
        "sentiment_label":   "neutral",
        "risk_level":        "medium",
        "key_distress_signal": "",
        "llm_summary":       "",
        "llm_raw":           "",
    }

    try:
        prompt   = _make_user_prompt(section, text)
        response = client.chat(SYSTEM_PROMPT, prompt)
        parsed   = _parse_llm_json(response)

        features = _parse_section_fields(section, parsed)
        features["llm_raw"] = response[:1000]
        return features

    except Exception as exc:
        log.warning("LLM extraction failed: %s", exc)
        defaults["llm_raw"] = str(exc)[:200]
        return defaults


# =============================================================================
# MinIO helpers
# =============================================================================

def _get_minio(endpoint, access_key, secret_key, secure) -> Minio:
    return Minio(endpoint, access_key=access_key, secret_key=secret_key, secure=secure)


def _list_text_objects(client: Minio, bucket: str, prefix: str) -> list[str]:
    objects = client.list_objects(bucket, prefix=prefix, recursive=True)
    return [o.object_name for o in objects if o.object_name.endswith(".txt")]


def _read_text_object(client: Minio, bucket: str, name: str) -> str:
    resp = client.get_object(bucket, name)
    try:
        return resp.read().decode("utf-8", errors="replace")
    finally:
        resp.close()
        resp.release_conn()


def _parse_object_name(name: str) -> dict:
    """
    Extract cik, accession, section from object name like:
      edgar_filings_text/ingestion_date=2024-01-01/CIK0000012345_00012345_mda.txt
    """
    basename = os.path.basename(name).replace(".txt", "")
    parts = basename.split("_")
    cik_raw  = parts[0] if len(parts) > 0 else ""
    acc      = parts[1] if len(parts) > 1 else ""
    section  = parts[2] if len(parts) > 2 else ""
    # strip leading "CIK" and leading zeros
    cik = cik_raw.lstrip("CIK").lstrip("0") or "0"
    # Try to pull ingestion_date from path
    m = re.search(r"ingestion_date=(\d{4}-\d{2}-\d{2})", name)
    ingestion_date = m.group(1) if m else ""
    return {
        "cik":            cik,
        "accession":      acc,
        "section":        section,
        "ingestion_date": ingestion_date,
        "object_name":    name,
    }


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
    text_prefix:   str  = "edgar_filings_text/",
    model:         str  = DEFAULT_MODEL,
    ollama_url:    str  = OLLAMA_BASE,
    dry_run:       bool = False,
    limit:         Optional[int] = None,
    skip_llm:      bool = False,
    workers:       int  = 3,
) -> pd.DataFrame:
    """
    Full LLM text pipeline:
      1. List bronze text objects
      2. For each: read → keyword features → LLM features
      3. Write Parquet to silver bucket (partitioned by section)
    """
    log.info("=" * 60)
    log.info("Silver text pipeline (LLM) — start")
    log.info("  Model   : %s  via %s", model, ollama_url)
    log.info("  Bronze  : %s/%s", bronze_bucket, text_prefix)
    log.info("  Silver  : %s", silver_bucket)
    log.info("  Dry run : %s  |  Skip LLM: %s", dry_run, skip_llm)
    log.info("=" * 60)

    # ------------------------------------------------------------------
    # 1. Init clients
    # ------------------------------------------------------------------
    minio = _get_minio(endpoint, access_key, secret_key, secure)

    ollama = OllamaClient(base_url=ollama_url, model=model)
    if not skip_llm:
        if ollama.health_check():
            log.info("Ollama health check OK — model '%s' ready", model)
        else:
            log.warning(
                "Ollama health check FAILED for model '%s'. "
                "Run: ollama pull %s && ollama serve\n"
                "Continuing with keyword-only features (--skip-llm mode).",
                model, model,
            )
            skip_llm = True

    # ------------------------------------------------------------------
    # 2. List bronze text objects
    # ------------------------------------------------------------------
    log.info("Listing bronze text objects under prefix '%s'...", text_prefix)
    objects = _list_text_objects(minio, bronze_bucket, text_prefix)
    log.info("Found %d text objects", len(objects))

    # ------------------------------------------------------------------
    # 2b. Resume: skip already-processed (cik, accession, section) tuples
    # ------------------------------------------------------------------
    import glob as _glob
    existing_keys: set[tuple[str, str, str]] = set()
    cache_files = _glob.glob(str(LOCAL_TEXT_CACHE / "**" / "*.parquet"), recursive=True)
    if not cache_files:
        # also check single flat file
        flat = LOCAL_TEXT_CACHE / "part-0.parquet"
        if flat.exists():
            cache_files = [str(flat)]
    if cache_files:
        try:
            existing_df = pd.concat(
                [pd.read_parquet(f, columns=["cik", "accession", "section"]) for f in cache_files],
                ignore_index=True,
            )
            existing_keys = set(
                zip(existing_df["cik"], existing_df["accession"], existing_df["section"])
            )
            log.info("Resume mode: %d already-processed documents found — skipping them",
                     len(existing_keys))
        except Exception as exc:
            log.warning("Could not load existing cache for resume: %s", exc)

    def _already_done(obj_name: str) -> bool:
        meta = _parse_object_name(obj_name)
        return (meta["cik"], meta["accession"], meta["section"]) in existing_keys

    before = len(objects)
    objects = [o for o in objects if not _already_done(o)]
    if before != len(objects):
        log.info("Skipped %d already-processed objects — %d remaining",
                 before - len(objects), len(objects))

    if limit:
        objects = objects[:limit]
        log.info("Limited to %d objects", limit)

    # ------------------------------------------------------------------
    # 3. Process each object
    # ------------------------------------------------------------------
    # 3. Process each object — parallel workers for speed
    # ------------------------------------------------------------------
    def _process_one(obj_name: str) -> dict | None:
        meta = _parse_object_name(obj_name)
        section = meta["section"]
        if section not in SECTIONS:
            return None
        try:
            text = _read_text_object(minio, bronze_bucket, obj_name)
        except Exception as exc:
            log.warning("Read failed %s: %s", obj_name, exc)
            return None
        if not text.strip():
            return None

        kw_feats = extract_keyword_features(text)

        if skip_llm:
            llm_feats = {
                "sentiment_score": 0.0,
                "sentiment_label": "neutral",
                "risk_level":      "medium",
                "llm_summary":     "",
                "llm_raw":         "skipped",
            }
        else:
            llm_feats = extract_llm_features(text, section, ollama)

        return {
            "cik":            meta["cik"],
            "accession":      meta["accession"],
            "section":        section,
            "ingestion_date": meta["ingestion_date"],
            "text_length":    len(text),
            **kw_feats,
            **llm_feats,
            "processed_at":   datetime.now(timezone.utc).isoformat(),
        }

    n_workers = 1 if skip_llm else max(1, workers)
    log.info("Processing %d objects with %d worker(s)...", len(objects), n_workers)

    records = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_process_one, obj): obj for obj in objects}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="LLM processing", unit="doc"):
            result = fut.result()
            if result:
                records.append(result)

    if not records:
        log.warning("No records produced — nothing to write.")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    log.info("Produced %d new records across %d companies",
             len(df), df["cik"].nunique())

    # Merge with existing cache so we keep previously processed records
    LOCAL_TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    existing_cache = LOCAL_TEXT_CACHE / "part-0.parquet"
    if existing_cache.exists() and not df.empty:
        try:
            old_df = pd.read_parquet(existing_cache)
            df = pd.concat([old_df, df], ignore_index=True).drop_duplicates(
                subset=["cik", "accession", "section"], keep="last"
            )
            log.info("Merged with existing cache → %d total records", len(df))
        except Exception as exc:
            log.warning("Could not merge with existing cache: %s", exc)

    if dry_run:
        log.info("[DRY RUN] Skipping write. Sample output:")
        print(df.head(3).to_string())
        return df

    # ------------------------------------------------------------------
    # 4. Write Parquet to silver bucket, partitioned by section
    # ------------------------------------------------------------------
    if not minio.bucket_exists(silver_bucket):
        minio.make_bucket(silver_bucket)
        log.info("Created bucket: %s", silver_bucket)

    for section, group in df.groupby("section"):
        object_name = f"edgar_text_features/section={section}/part-0.parquet"
        buf = io.BytesIO()
        pq.write_table(
            pa.Table.from_pandas(group.reset_index(drop=True), preserve_index=False),
            buf, compression="snappy",
        )
        data = buf.getvalue()
        minio.put_object(
            silver_bucket, object_name,
            data=io.BytesIO(data), length=len(data),
            content_type="application/octet-stream",
        )
        log.info("Wrote  %s  (%d rows, %d KB)",
                 object_name, len(group), len(data) // 1024)

    # Write merged result to local cache
    LOCAL_TEXT_CACHE.mkdir(parents=True, exist_ok=True)
    df.to_parquet(LOCAL_TEXT_CACHE / "part-0.parquet", index=False, engine="pyarrow")
    log.info("Local cache updated → %s (%d total records)", LOCAL_TEXT_CACHE, len(df))

    log.info("=" * 60)
    log.info("Silver text pipeline — complete")
    log.info("=" * 60)
    return df


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "LLM silver pipeline: bronze text → NLP features → silver Parquet.\n\n"
            "Recommended models:\n"
            "  llama3.2:3b   — fast, CPU-friendly (default)\n"
            "  llama3.2:8b   — best quality, needs 16GB+ RAM\n"
            "  minicpm3:4b   — compact, good for IFRS/multilingual filers\n\n"
            "Quick start:\n"
            "  ollama pull llama3.2:3b\n"
            "  python -m src.silver.build_silver_text\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--endpoint",      default=os.getenv("MINIO_ENDPOINT",      "localhost:9000"))
    p.add_argument("--access-key",    default=os.getenv("MINIO_ACCESS_KEY",    "minioadmin"))
    p.add_argument("--secret-key",    default=os.getenv("MINIO_SECRET_KEY",    "minioadmin"))
    p.add_argument("--secure",        action="store_true",
                   default=os.getenv("MINIO_SECURE", "false").lower() == "true")
    p.add_argument("--bronze-bucket", default=os.getenv("MINIO_BRONZE_BUCKET", "bronze"))
    p.add_argument("--silver-bucket", default=os.getenv("MINIO_SILVER_BUCKET", "silver"))
    p.add_argument("--text-prefix",   default="edgar_filings_text/")
    p.add_argument("--model",         default=DEFAULT_MODEL,
                   help="Ollama model tag, e.g. llama3.2:3b or minicpm3:4b")
    p.add_argument("--ollama-url",    default=OLLAMA_BASE)
    p.add_argument("--dry-run",       action="store_true",
                   help="Parse and print without writing to MinIO")
    p.add_argument("--skip-llm",      action="store_true",
                   help="Keyword features only — no Ollama calls")
    p.add_argument("--limit",         type=int, default=None,
                   help="Process only first N objects (for testing)")
    p.add_argument("--workers",       type=int, default=3,
                   help="Parallel Ollama workers (default 3; use 1 on weak CPU)")
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
        text_prefix=args.text_prefix,
        model=args.model,
        ollama_url=args.ollama_url,
        dry_run=args.dry_run,
        limit=args.limit,
        skip_llm=args.skip_llm,
        workers=args.workers,
    )
