# BDT Distress Warning System

Corporate Financial Distress Early Warning from SEC EDGAR Filings 
---

## Overview

This system processes SEC EDGAR company filings and financial statements to detect early signals of financial distress. It combines structured XBRL financial data with the LoPucki Bankruptcy Research Database to compute Altman Z-Scores, classify companies into distress zones, and flag deteriorating trends.

The pipeline follows a **Bronze → Silver → Gold medallion architecture** backed by MinIO object storage.

---

## Architecture

```
SEC EDGAR API
      │
      ▼
┌─────────────┐
│   BRONZE    │  Raw JSON (company facts + submissions)
│   MinIO     │  Immutable, resumable ingestion
└──────┬──────┘
       │  PySpark + PyArrow
       ▼
┌─────────────┐
│   SILVER    │  Standardised financial table (Parquet)
│   MinIO     │  One row per company per reporting period
└──────┬──────┘
       │  DuckDB
       ▼
┌─────────────┐
│    GOLD     │  Altman Z-Scores, distress zones, trend flags
│   MinIO     │  Partitioned by distress_label
└──────┬──────┘
       │  Streamlit
       ▼
  Dashboard
```

---

## Setup

### Prerequisites

- Python 3.11+
- Docker (for MinIO)
- Conda environment recommended

### 1. Clone the repo

```bash
git clone https://github.com/AlessandroFulci/bdt-distress-warning.git
cd bdt-distress-warning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start MinIO

```bash
docker run -p 9000:9000 -p 9001:9001 \
  -e "MINIO_ROOT_USER=minioadmin" \
  -e "MINIO_ROOT_PASSWORD=minioadmin" \
  quay.io/minio/minio server /data --console-address ":9001"
```

MinIO console available at [http://localhost:9001](http://localhost:9001)

### 4. Configure environment

Create a `.env` file in the project root:

```env
SEC_USER_AGENT=Your Name your@email.com
MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_BRONZE_BUCKET=bronze
MINIO_SILVER_BUCKET=silver
MINIO_GOLD_BUCKET=gold
MINIO_SECURE=false
```

> The SEC requires a valid User-Agent header identifying the requester.

### 5. Verify setup

```bash
python src/test_setup.py
```

---

## Running the Pipeline

Run each step in order:

```bash
# Bronze — ingest raw EDGAR data (~20-30 min, resumable)
python src/ingestion/fetch_bronze.py

# Silver — parse and standardise into Parquet
python src/silver/build_silver.py

# Gold — compute distress scores with DuckDB
python src/gold/build_gold.py
```

### Launch the dashboard

```bash
streamlit run src/dashboard/app.py
```

---

## Data Sources

| Source | Description |
|--------|-------------|
| [SEC EDGAR XBRL Company Facts](https://data.sec.gov/api/xbrl/companyfacts/) | Structured financial data for every public company |
| [SEC EDGAR Submissions](https://data.sec.gov/submissions/) | Filing metadata per company |
| [Wikipedia S&P 500](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies) | Current S&P 500 constituents (healthy label) |
| [LoPucki Bankruptcy Research Database](https://lopucki.law.ufl.edu) | Verified Chapter 11 bankruptcies 2010–2022 (distress label) |

**Universe:** ~736 companies — S&P 500 constituents + LoPucki bankruptcies (Chapter 11, assets ≥ $500M, year ≥ 2010)

---

## Gold Layer Features

| Feature | Description |
|---------|-------------|
| `altman_z_score` | Altman Z-Score (5-component model) |
| `distress_zone` | `distress` / `grey` / `safe` classification |
| `x1_wc_to_assets` | Working capital / total assets |
| `x2_re_to_assets` | Retained earnings / total assets |
| `x3_ebit_to_assets` | EBIT / total assets |
| `x4_equity_to_liab` | Book equity / total liabilities |
| `x5_rev_to_assets` | Revenue / total assets |
| `revenue_growth_qoq` | Quarter-over-quarter revenue growth |
| `net_income_growth_qoq` | Quarter-over-quarter net income growth |
| `asset_growth_qoq` | Quarter-over-quarter asset growth |
| `revenue_declining_3q` | Flag: 3+ consecutive quarters of revenue decline |
| `net_income_declining_3q` | Flag: 3+ consecutive quarters of net income decline |
| `distress_label` | Ground truth: 1 = distressed (LoPucki), 0 = healthy |

**Altman Z-Score zones:**
- Z < 1.81 → Distress zone
- 1.81 ≤ Z ≤ 2.99 → Grey zone
- Z > 2.99 → Safe zone

---

## Project Structure

```
bdt-distress-warning/
├── config/
│   └── company_universe.csv      # ~736 companies with distress labels
├── data/
│   └── reference/
│       └── lopucki_brd_2023.csv  # LoPucki Bankruptcy Research Database
├── src/
│   ├── ingestion/
│   │   ├── fetch_bronze.py       # Bronze ingestion pipeline
│   │   ├── sec_client.py         # SEC EDGAR HTTP client (rate-limited)
│   │   └── minio_client.py       # MinIO bronze store with lineage metadata
│   ├── silver/
│   │   └── build_silver.py       # PySpark + PyArrow silver pipeline
│   ├── gold/
│   │   └── build_gold.py         # DuckDB gold analytics pipeline
│   ├── dashboard/
│   │   └── app.py                # Streamlit dashboard
│   ├── build_universe.py         # Build company universe CSV
│   └── test_setup.py             # Verify SEC + MinIO connectivity
├── logs/                         # Ingestion and build logs (git-ignored)
├── requirements.txt
└── .env                          # Local credentials (git-ignored)
```

---

## Team

| Name | Layer |
|------|-------|
| Alessandro Fulci | Bronze (ingestion) |
| Nary | Silver + Gold + Dashboard |
