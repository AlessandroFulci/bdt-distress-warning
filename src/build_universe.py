"""
Build the company universe CSV by combining:
  1. S&P 500 constituents from Wikipedia (healthy companies, label=0)
  2. Verified Chapter 11 bankruptcies from the LoPucki Bankruptcy
     Research Database (distressed companies, label=1)

Output: config/company_universe.csv
"""
import os
import sys
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("ERROR: SEC_USER_AGENT not set in .env")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)
LOPUCKI_PATH = PROJECT_ROOT / "data" / "reference" / "lopucki_brd_2023.csv"
OUTPUT_PATH = CONFIG_DIR / "company_universe.csv"

# Filters for the LoPucki bankruptcy subset
MIN_BANKRUPTCY_YEAR = 2010   # XBRL became mandatory ~2009-2010 for large filers
MIN_ASSETS_M = 500.0         # USD millions, focus on substantial public companies


# ---------------------------------------------------------------------------
# Step 1: Fetch S&P 500 constituents from Wikipedia
# ---------------------------------------------------------------------------
def fetch_sp500():
    print("[1/2] Fetching S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    tables = pd.read_html(url, storage_options={"User-Agent": USER_AGENT})
    df = tables[0]  # first table is the constituents list

    df = df.rename(columns={"Symbol": "ticker", "Security": "company_name", "CIK": "cik"})
    df = df[["ticker", "company_name", "cik"]].copy()

    # Wikipedia uses '.' in some tickers (BRK.B), SEC uses '-' (BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["source"] = "sp500"
    df["distress_label"] = 0
    df["distress_event_date"] = ""
    df["distress_event_type"] = ""

    print(f"      Got {len(df)} S&P 500 companies")
    return df


# ---------------------------------------------------------------------------
# Step 2: Load verified bankruptcies from LoPucki BRD
# ---------------------------------------------------------------------------
def fetch_bankruptcies():
    print("[2/2] Loading bankruptcies from LoPucki BRD...")
    if not LOPUCKI_PATH.exists():
        sys.exit(
            f"ERROR: LoPucki CSV not found at {LOPUCKI_PATH}\n"
            f"Please copy the file 'Florida-UCLA-LoPucki Bankruptcy Research "
            f"Database 1-12-2023.csv' to that location."
        )

    df = pd.read_csv(LOPUCKI_PATH, low_memory=False, encoding="latin-1")
    print(f"      Loaded {len(df)} total cases (1980-2022)")

    # Apply filters
    n0 = len(df)
    df = df[df["Chapter"] == "11"]
    df = df[df["CikBefore"].notna()]
    df = df[df["YearFiled"] >= MIN_BANKRUPTCY_YEAR]
    df = df[df["AssetsBefore"] >= MIN_ASSETS_M]
    print(
        f"      After filters (Ch.11, valid CIK, year>={MIN_BANKRUPTCY_YEAR}, "
        f"assets>=${MIN_ASSETS_M:.0f}M): {len(df)} cases"
    )

    # Reformat into the universe schema
    rows = []
    for _, r in df.iterrows():
        cik = str(int(r["CikBefore"])).zfill(10)
        rows.append({
            "ticker": "",  # LoPucki doesn't track tickers
            "company_name": r["NameCorp"],
            "cik": cik,
            "source": "lopucki_brd",
            "distress_label": 1,
            "distress_event_date": r["DateFiled"],
            "distress_event_type": "chapter_11",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combine + save
# ---------------------------------------------------------------------------
def main():
    sp500 = fetch_sp500()
    bankrupt = fetch_bankruptcies()

    print("\nCombining and deduplicating...")
    universe = pd.concat([sp500, bankrupt], ignore_index=True)

    # If a company appears in both, keep the bankruptcy row (it has the label)
    universe = universe.sort_values("distress_label", ascending=False)
    before = len(universe)
    universe = universe.drop_duplicates(subset="cik", keep="first")
    after = len(universe)
    if before != after:
        print(f"  Deduplicated {before - after} CIK overlaps "
              f"(companies present in both S&P 500 and LoPucki)")

    universe = universe.sort_values(
        ["distress_label", "company_name"],
        ascending=[False, True]
    ).reset_index(drop=True)

    universe.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(universe)} companies to {OUTPUT_PATH}")
    print(f"  Healthy (S&P 500):     {(universe['distress_label'] == 0).sum()}")
    print(f"  Distressed (LoPucki):  {(universe['distress_label'] == 1).sum()}")


if __name__ == "__main__":
    main()
