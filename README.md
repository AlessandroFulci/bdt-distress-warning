# BDT Financial Distress Early Warning System

A big data pipeline that ingests SEC 10-K filings, computes Altman Z-Scores, runs LLM analysis on filing text, and visualises financial distress signals in an interactive Streamlit dashboard.

---

## Architecture

```
SEC EDGAR ──► Bronze (MinIO) ──► Silver (Parquet) ──► Gold (DuckDB) ──► Dashboard
                 raw text         keyword + LLM         Z-Scores         Streamlit
                                    features            distress zones
```

| Layer | Contents | Storage |
|-------|----------|---------|
| Bronze | Raw 10-K filing text — MD&A, Risk Factors, Business Description | MinIO `bronze` bucket |
| Silver | Keyword features + LLM sentiment, risk signals, going-concern flags | MinIO `silver` + `data/cache/silver_text/` |
| Gold | Altman Z-Score, distress zones, QoQ growth trends, BRD enrichment | MinIO `gold` + `data/cache/gold_distress/` |

---

## Quick Start (Docker)

### Prerequisites
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

### 1. Clone and configure

```bash
git clone <repo-url>
cd bdt-distress-warning
```

Edit `.env`:

```env
SEC_USER_AGENT=Your Name your@email.com
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BRONZE_BUCKET=bronze
MINIO_SILVER_BUCKET=silver
MINIO_GOLD_BUCKET=gold
MINIO_SECURE=false
```

### 2. Start all services

```bash
docker compose up -d
```

Starts MinIO (object storage), Ollama (LLM server), and the Streamlit app. On first start, Ollama auto-pulls `llama3.2:3b`. Wait for it:

```bash
docker compose logs -f ollama-init
# Done when you see: Model ready.
```

Also pull the faster 1b model for text analysis:

```bash
docker compose exec ollama ollama pull llama3.2:1b
```

### 3. Build the company universe

> Requires `data/reference/lopucki_brd_2023.csv` — see [LoPucki BRD](#lopucki-bankruptcy-research-database) section.

```bash
docker compose exec app python -m src.build_universe
```

### 4. Run the data pipeline

```bash
# Fetch 10-K filing text from SEC EDGAR → MinIO bronze
docker compose exec app python -m src.ingestion.fetch_bronze_text --max-companies 20

# Keyword-only features for all documents (fast, ~10 seconds)
docker compose exec app python -m src.silver.build_silver_text --skip-llm

# LLM features for a batch (resumable — safe to re-run, skips already-processed)
docker compose exec app python -m src.silver.build_silver_text \
    --model llama3.2:1b --workers 3 --limit 100

# Build gold layer (Altman Z-Score, distress zones, BRD enrichment)
docker compose exec app python -m src.gold.build_gold
```

### 5. Open the dashboard

**http://localhost:8501**

---

## Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Streamlit dashboard | http://localhost:8501 | — |
| MinIO console | http://localhost:9001 | `minioadmin` / `minioadmin` |
| Ollama API | http://localhost:11434 | — |

---

## Data Persistence

All processed data survives `docker compose down` and restarts. **Just run `docker compose up -d` to resume.**

| Data | Location | Survives restart | Survives `down -v` |
|------|----------|-----------------|-------------------|
| Gold Parquet | `data/cache/gold_distress/` | ✅ | ✅ |
| Silver text (LLM features) | `data/cache/silver_text/` | ✅ | ✅ |
| Bronze 10-K text | MinIO `minio_data` volume | ✅ | ❌ |
| Ollama model weights | `ollama_data` volume | ✅ | ❌ |

> **Never use `docker compose down -v`** unless you want a full reset. Use `docker compose down` (no `-v`) to safely stop and restart later.

If you do run `docker compose down -v`, re-run these to restore:
```bash
docker compose exec ollama ollama pull llama3.2:1b
docker compose exec app python -m src.ingestion.fetch_bronze_text --max-companies 20
docker compose exec app python -m src.silver.build_silver_text --model llama3.2:1b --workers 3
docker compose exec app python -m src.gold.build_gold
```

---

## Project Structure

```
bdt-distress-warning/
├── config/
│   └── company_universe.csv        # 736 companies: S&P 500 + LoPucki bankruptcies
├── data/
│   ├── cache/                      # Auto-generated Parquet cache (gitignored)
│   │   ├── gold_distress/          # Gold layer partitioned by distress_label
│   │   ├── silver_financials/      # Silver financial facts
│   │   └── silver_text/            # LLM text features (resumable)
│   └── reference/
│       └── lopucki_brd_2023.csv    # LoPucki Bankruptcy Research Database
├── logs/                           # Pipeline run logs
├── src/
│   ├── build_universe.py           # Build company universe (S&P 500 + LoPucki BRD)
│   ├── ingestion/
│   │   └── fetch_bronze_text.py    # Fetch 10-K sections from SEC EDGAR → MinIO
│   ├── silver/
│   │   └── build_silver_text.py    # LLM pipeline: bronze text → NLP features
│   ├── gold/
│   │   └── build_gold.py           # Altman Z-Score + distress zones via DuckDB
│   └── dashboard/
│       └── app.py                  # Streamlit dashboard
├── run_pipeline.py                 # Smart orchestrator (checks cache freshness)
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## LLM Text Pipeline

The silver text pipeline sends each 10-K section to a local LLM (via Ollama) and extracts structured distress signals.

### Section-specific signals extracted

| Section | Key fields |
|---------|-----------|
| **MD&A** | `management_tone`, `liquidity_signal`, `going_concern_explicit`, `covenant_breach_mentioned`, `debt_restructuring_mentioned` |
| **Risk Factors** | `risk_level`, `liquidity_risk_disclosed`, `going_concern_risk_disclosed`, `debt_maturity_risk`, `severity_language`, `top_risk_category` |
| **Business** | `business_model_viability`, `competitive_position`, `revenue_concentration_risk`, `regulatory_dependency` |

### Recommended models

| Model | Size | Speed (CPU) | Use case |
|-------|------|-------------|----------|
| `llama3.2:1b` | 1.3 GB | Fast (~10s/doc) | Large batches, limited RAM |
| `llama3.2:3b` | 2.0 GB | Medium (~30s/doc) | Better quality |

### Resumable processing

The pipeline automatically skips already-processed documents. Re-run the same command to continue from where you left off:

```bash
# Run 100 docs now, come back and run another 100 later
docker compose exec app python -m src.silver.build_silver_text \
    --model llama3.2:1b --workers 3 --limit 100
```

---

## Pipeline Reference

### Individual steps

```bash
# Build company universe
docker compose exec app python -m src.build_universe

# Fetch 10-K text (bronze)
docker compose exec app python -m src.ingestion.fetch_bronze_text \
    --max-companies 20 --max-filings 3

# Keyword-only silver (instant, all 5000+ docs)
docker compose exec app python -m src.silver.build_silver_text --skip-llm

# LLM silver (resumable batches)
docker compose exec app python -m src.silver.build_silver_text \
    --model llama3.2:1b --workers 3 --limit 100

# Gold layer
docker compose exec app python -m src.gold.build_gold
```

### Smart orchestrator

```bash
docker compose exec app python run_pipeline.py --status   # check what's stale
docker compose exec app python run_pipeline.py            # run only stale layers
docker compose exec app python run_pipeline.py --force    # rebuild everything
docker compose exec app python run_pipeline.py --step gold
```

---

## Dashboard Features

| Section | Description |
|---------|-------------|
| **Pipeline Overview** | Row counts, zone distribution, data freshness |
| **Altman Z-Score** | Distribution, per-company timeline, distress zones |
| **Trend Analysis** | QoQ revenue/income growth, 3-quarter decline flags |
| **LLM Text Analysis** | Sentiment scores by section, risk levels, going-concern signals |

### Altman Z-Score Zones

| Zone | Z-Score | Interpretation |
|------|---------|---------------|
| 🔴 Distress | Z < 1.81 | High bankruptcy risk |
| 🟡 Grey | 1.81 – 2.99 | Uncertain — monitor closely |
| 🟢 Safe | Z > 2.99 | Low bankruptcy risk |

---

## LoPucki Bankruptcy Research Database

Distressed company labels and bankruptcy outcome data come from the [LoPucki BRD](https://lopucki.law.ufl.edu/). The pipeline enriches each distressed company with 27 BRD columns covering:

- **Outcome** — disposition, emerged, days in bankruptcy, fresh-start accounting
- **Financing** — DIP loan, §363 sale, prepackaged filing
- **Governance** — CEO replaced, trustee appointed
- **Financials at filing** — assets, sales, EBIT (USD millions)
- **Post-emergence** — `AfterEmerging` description, emergence date, post-emergence assets/sales/EBIT/EBITDA

To set up:
1. Download `Florida-UCLA-LoPucki Bankruptcy Research Database 1-12-2023.csv`
2. Place at `data/reference/lopucki_brd_2023.csv`
3. Run `docker compose exec app python -m src.build_universe`

---

## Docker Management

```bash
docker compose ps                                          # check service status
docker compose logs -f app                                 # follow app logs
docker compose logs -f ollama                              # follow Ollama logs
docker compose restart app                                 # restart one service
docker compose down                                        # stop (data preserved)
docker compose down -v                                     # stop + wipe all volumes
docker compose build app && docker compose up -d app       # rebuild after code changes
```

---

## Troubleshooting

**Dashboard shows no data**
Run the pipeline first. The dashboard reads from `data/cache/` which starts empty.

**Ollama 404 error when running LLM pipeline**
The model isn't pulled yet. Run:
```bash
docker compose exec ollama ollama pull llama3.2:1b
```

**Ollama container unhealthy on first start**
Normal — startup takes ~30s. Check with `docker compose logs ollama`. If it's running, restart the stack: `docker compose down && docker compose up -d`.

**Port conflicts (9000, 9001, 11434, 8501)**
Stop locally running services first:
```bash
pkill ollama
```
Or edit the host ports in `docker-compose.yml`.

**SEC EDGAR rate limiting**
Use `--max-companies` to process in smaller batches.

**`NoSuchBucket` error outside Docker**
MinIO buckets are created automatically by the `minio-init` service in Docker. Outside Docker, create them manually via the MinIO console at http://localhost:9001.
