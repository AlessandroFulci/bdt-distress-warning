"""
Build the company universe CSV by combining:
  1. S&P 500 constituents from Wikipedia (healthy companies, label=0)
  2. Verified Chapter 11 bankruptcies from the LoPucki Bankruptcy
     Research Database (distressed companies, label=1)

Enriches distressed companies with 15 BRD columns that are directly
relevant to financial distress analysis (outcome, financing, industry,
governance, severity).

Output: config/company_universe.csv
"""
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("ERROR: SEC_USER_AGENT not set in .env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR   = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)
LOPUCKI_PATH = PROJECT_ROOT / "data" / "reference" / "lopucki_brd_2023.csv"
OUTPUT_PATH  = CONFIG_DIR / "company_universe.csv"

# Filters for the LoPucki bankruptcy subset
MIN_BANKRUPTCY_YEAR = 2010   # XBRL became mandatory ~2009-2010 for large filers
MIN_ASSETS_M        = 500.0  # USD millions — focus on substantial public companies


# ---------------------------------------------------------------------------
# BRD column parsing helpers
# All BRD columns come in as object (string). We cast to usable types here.
# ---------------------------------------------------------------------------

def _parse_date(series: pd.Series) -> pd.Series:
    """Parse M/D/YYYY or similar date strings to YYYY-MM-DD strings."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.strftime("%Y-%m-%d").where(parsed.notna(), other="")


def _yes_no_to_int(series: pd.Series, true_values: list[str] | None = None) -> pd.Series:
    """Convert yes/no/not applicable strings → 1/0 int. Handles NaN safely."""
    tv = true_values or ["yes", "y", "true", "1", "replaced", "refiled"]
    return series.fillna("").astype(str).str.strip().str.lower().isin(tv).astype(int)


def _parse_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _clean_category(series: pd.Series) -> pd.Series:
    """Strip, lower, replace spaces with underscores for categorical columns."""
    return series.fillna("").astype(str).str.strip()


# ---------------------------------------------------------------------------
# Step 1: Fetch S&P 500 constituents from Wikipedia
# ---------------------------------------------------------------------------
def fetch_sp500() -> pd.DataFrame:
    print("[1/2] Fetching S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": USER_AGENT})
    df = tables[0]

    df = df.rename(columns={"Symbol": "ticker", "Security": "company_name", "CIK": "cik"})
    df = df[["ticker", "company_name", "cik"]].copy()
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"]    = df["cik"].astype(str).str.zfill(10)
    df["source"] = "sp500"

    # Label columns
    df["distress_label"]      = 0
    df["distress_event_date"] = ""
    df["distress_event_type"] = ""

    # BRD-specific columns (empty for healthy companies)
    for col in _BRD_ENRICHMENT_COLS:
        df[col] = np.nan if col in _BRD_FLOAT_COLS else ""

    print(f"      Got {len(df)} S&P 500 companies")
    return df


# ---------------------------------------------------------------------------
# BRD enrichment column definitions
# ---------------------------------------------------------------------------

# Columns we extract from LoPucki and carry through to the universe CSV
_BRD_ENRICHMENT_COLS = [
    # Outcome
    "brd_chapter",             # 7 or 11
    "brd_disposition",         # emerged | liquidated | converted | dismissed | …
    "brd_emerged",             # 1 if company emerged from bankruptcy, else 0
    "brd_days_in_bk",          # number of days spent in bankruptcy
    "brd_fresh_start",         # 1 if fresh-start accounting applied on emergence
    "brd_refiled",             # 1 if company refiled for bankruptcy after dismissal
    # Financing & process
    "brd_dip_loan",            # 1 if debtor-in-possession financing obtained
    "brd_sale_363",            # 1 if assets sold via §363 sale
    "brd_prepackaged",         # 1 if pre-packaged/pre-negotiated bankruptcy
    "brd_voluntary",           # 1 if voluntary filing, 0 if involuntary
    # Governance
    "brd_ceo_replaced",        # 1 if CEO was replaced during bankruptcy
    "brd_trustee_appointed",   # 1 if a Chapter 11 trustee was appointed (severe)
    # Financials at filing
    "brd_assets_before_m",     # total assets in USD millions at last 10-K before filing
    "brd_sales_before_m",      # total revenue in USD millions at last 10-K before filing
    "brd_ebit_before_m",       # EBIT in USD millions at last 10-K before filing
    # Industry
    "brd_sic_description",     # SIC industry description string
    "brd_sic_division",        # SIC division (e.g. "Manufacturing")
    # Post-emergence outcome
    "brd_after_emerging",      # free-text: what happened after bankruptcy (acquired, merged, liquidated, etc.)
    "brd_date_emerging",       # date the company emerged from bankruptcy
    "brd_year_emerged",        # year of emergence
    "brd_days_emerge_to_refile", # days between emergence and re-filing (if refiled)
    "brd_name_emerging",       # company name at emergence (may differ from filing name)
    # Post-emergence financials (USD millions, from first 10-K after emergence)
    "brd_assets_emerging_m",   # total assets at emergence
    "brd_sales_emerging_m",    # revenue at emergence
    "brd_ebit_emerging_m",     # EBIT at emergence
    "brd_ebitda_emerging_m",   # EBITDA at emergence
    "brd_net_income_emerging_m", # net income at emergence
]

_BRD_FLOAT_COLS = {
    "brd_assets_before_m", "brd_sales_before_m", "brd_ebit_before_m", "brd_days_in_bk",
    "brd_assets_emerging_m", "brd_sales_emerging_m", "brd_ebit_emerging_m",
    "brd_ebitda_emerging_m", "brd_net_income_emerging_m", "brd_year_emerged",
    "brd_days_emerge_to_refile",
}


# ---------------------------------------------------------------------------
# Step 2: Load & enrich bankruptcies from LoPucki BRD
# ---------------------------------------------------------------------------
def fetch_bankruptcies() -> pd.DataFrame:
    print("[2/2] Loading bankruptcies from LoPucki BRD...")
    if not LOPUCKI_PATH.exists():
        sys.exit(
            f"ERROR: LoPucki CSV not found at {LOPUCKI_PATH}\n"
            "Please copy 'Florida-UCLA-LoPucki Bankruptcy Research "
            "Database 1-12-2023.csv' to that location."
        )

    raw = pd.read_csv(LOPUCKI_PATH, low_memory=False, encoding="latin-1")
    print(f"      Loaded {len(raw)} total cases (1980-present)")

    # ---- Parse YearFiled (numeric or derive from DateFiled) ----
    if "YearFiled" in raw.columns:
        raw["YearFiled"] = pd.to_numeric(raw["YearFiled"], errors="coerce")
    else:
        raw["YearFiled"] = pd.to_datetime(raw["DateFiled"], errors="coerce").dt.year

    # ---- Parse AssetsBefore ----
    raw["AssetsBefore"] = _parse_float(raw["AssetsBefore"])

    # ---- Parse Chapter ----
    raw["Chapter"] = pd.to_numeric(raw["Chapter"], errors="coerce")

    # ---- Apply filters ----
    df = raw[
        (raw["Chapter"] == 11) &
        (raw["CikBefore"].notna()) &
        (raw["YearFiled"] >= MIN_BANKRUPTCY_YEAR) &
        (raw["AssetsBefore"] >= MIN_ASSETS_M)
    ].copy()
    print(
        f"      After filters (Ch.11, valid CIK, "
        f"year>={MIN_BANKRUPTCY_YEAR}, assets>=${MIN_ASSETS_M:.0f}M): "
        f"{len(df)} cases"
    )

    # ---- Parse disposition string ----
    def _map_disposition(val: str) -> str:
        v = str(val).lower().strip()
        if "emerged"   in v or "plan confirmed" in v: return "emerged"
        if "liquidat"  in v or "chapter 7"      in v: return "liquidated"
        if "converted" in v:                           return "converted"
        if "dismissed" in v:                           return "dismissed"
        if "sold"      in v or "363 sale"        in v: return "363_sale"
        if "pending"   in v:                           return "pending"
        return "other"

    # ---- Build enriched rows ----
    rows = []
    for _, r in df.iterrows():
        cik = str(int(float(r["CikBefore"]))).zfill(10)

        rows.append({
            "ticker":       "",
            "company_name": r["NameCorp"],
            "cik":          cik,
            "source":       "lopucki_brd",

            # Core label fields
            "distress_label":      1,
            "distress_event_date": _parse_date(pd.Series([r["DateFiled"]]))[0],
            "distress_event_type": f"chapter_{int(r['Chapter'])}",

            # Outcome
            "brd_chapter":     int(r["Chapter"]),
            "brd_disposition": _map_disposition(r.get("Disposition", "")),
            "brd_emerged":     _yes_no_to_int(pd.Series([r.get("Emerge", "no")]))[0],
            "brd_days_in_bk":  _parse_float(pd.Series([r.get("DaysIn", np.nan)]))[0],
            "brd_fresh_start": _yes_no_to_int(pd.Series([r.get("FreshStartAccounting", "no")]))[0],
            "brd_refiled":     _yes_no_to_int(pd.Series([r.get("Refile", "no")]),
                                               true_values=["refiled", "yes"])[0],

            # Financing & process
            "brd_dip_loan":    _yes_no_to_int(pd.Series([r.get("DipLoan", "no")]))[0],
            "brd_sale_363":    _yes_no_to_int(pd.Series([r.get("Sale363", "no")]))[0],
            "brd_prepackaged": _yes_no_to_int(
                pd.Series([r.get("Prepackaged", "no")]),
                true_values=["yes", "prepackaged", "pre-packaged", "prenegotiated"]
            )[0],
            "brd_voluntary":   _yes_no_to_int(
                pd.Series([r.get("Voluntary", "voluntary")]),
                true_values=["voluntary"]
            )[0],

            # Governance
            "brd_ceo_replaced":     _yes_no_to_int(
                pd.Series([r.get("CeoReplaced", "noreplace")]),
                true_values=["replaced"]
            )[0],
            "brd_trustee_appointed": _yes_no_to_int(pd.Series([r.get("Trustee", "no")]))[0],

            # Financials at filing (USD millions)
            "brd_assets_before_m": _parse_float(pd.Series([r.get("AssetsBefore", np.nan)]))[0],
            "brd_sales_before_m":  _parse_float(pd.Series([r.get("SalesBefore",  np.nan)]))[0],
            "brd_ebit_before_m":   _parse_float(pd.Series([r.get("EbitBefore",   np.nan)]))[0],

            # Industry
            "brd_sic_description": "" if pd.isna(r.get("SICDescription")) else str(r.get("SICDescription", "")).strip(),
            "brd_sic_division":    "" if pd.isna(r.get("SICDivision"))    else str(r.get("SICDivision",    "")).strip(),

            # Post-emergence outcome
            "brd_after_emerging":       "" if pd.isna(r.get("AfterEmerging"))  else str(r.get("AfterEmerging",  "")).strip(),
            "brd_date_emerging":        _parse_date(pd.Series([r.get("DateEmerging", "")]))[0],
            "brd_year_emerged":         _parse_float(pd.Series([r.get("YearEmerged", np.nan)]))[0],
            "brd_days_emerge_to_refile":_parse_float(pd.Series([r.get("DaysEmergeToRefile", np.nan)]))[0],
            "brd_name_emerging":        "" if pd.isna(r.get("NameEmerging"))   else str(r.get("NameEmerging",   "")).strip(),

            # Post-emergence financials (USD millions)
            "brd_assets_emerging_m":    _parse_float(pd.Series([r.get("AssetsEmerging",   np.nan)]))[0],
            "brd_sales_emerging_m":     _parse_float(pd.Series([r.get("SalesEmerging",    np.nan)]))[0],
            "brd_ebit_emerging_m":      _parse_float(pd.Series([r.get("EbitEmerging",     np.nan)]))[0],
            "brd_ebitda_emerging_m":    _parse_float(pd.Series([r.get("EbitdaEmerging",   np.nan)]))[0],
            "brd_net_income_emerging_m":_parse_float(pd.Series([r.get("NetIncomeEmerging",np.nan)]))[0],
        })

    result = pd.DataFrame(rows)
    print(f"\n      BRD enrichment summary (distressed companies):")
    print(f"        Emerged:           {result['brd_emerged'].sum()} / {len(result)}")
    print(f"        DIP financing:     {result['brd_dip_loan'].sum()} / {len(result)}")
    print(f"        §363 sale:         {result['brd_sale_363'].sum()} / {len(result)}")
    print(f"        CEO replaced:      {result['brd_ceo_replaced'].sum()} / {len(result)}")
    print(f"        Trustee appointed: {result['brd_trustee_appointed'].sum()} / {len(result)}")
    print(f"        Prepackaged:       {result['brd_prepackaged'].sum()} / {len(result)}")
    disp = result['brd_disposition'].value_counts().to_dict()
    print(f"        Dispositions:      {disp}")
    return result


# ---------------------------------------------------------------------------
# Combine + save
# ---------------------------------------------------------------------------
def main():
    sp500    = fetch_sp500()
    bankrupt = fetch_bankruptcies()

    print("\nCombining and deduplicating...")
    universe = pd.concat([sp500, bankrupt], ignore_index=True)

    # If a company appears in both (e.g. was in S&P 500 then went bankrupt),
    # keep the bankruptcy row so the label is correct
    universe = universe.sort_values("distress_label", ascending=False)
    before   = len(universe)
    universe = universe.drop_duplicates(subset="cik", keep="first")
    after    = len(universe)
    if before != after:
        print(f"  Deduplicated {before - after} CIK overlaps")

    universe = universe.sort_values(
        ["distress_label", "company_name"], ascending=[False, True]
    ).reset_index(drop=True)

    universe.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(universe)} companies → {OUTPUT_PATH}")
    print(f"  Columns:               {len(universe.columns)}")
    print(f"  Healthy (S&P 500):     {(universe['distress_label'] == 0).sum()}")
    print(f"  Distressed (LoPucki):  {(universe['distress_label'] == 1).sum()}")
    print(f"\nNew BRD columns added: {_BRD_ENRICHMENT_COLS}")


if __name__ == "__main__":
    main()
