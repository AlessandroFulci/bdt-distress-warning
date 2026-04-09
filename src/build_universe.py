"""
Build the company universe CSV: S&P 500 constituents + 30 historical bankruptcies.
Output: config/company_universe.csv
"""
import os
import sys
import time
import json
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

USER_AGENT = os.getenv("SEC_USER_AGENT")
if not USER_AGENT:
    sys.exit("ERROR: SEC_USER_AGENT not set in .env")

HEADERS = {"User-Agent": USER_AGENT}
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = CONFIG_DIR / "company_universe.csv"


# ---------------------------------------------------------------------------
# Step 1: Fetch S&P 500 constituents from Wikipedia
# ---------------------------------------------------------------------------
def fetch_sp500():
    print("[1/3] Fetching S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    # Wikipedia blocks default pandas/urllib UA; pass our SEC UA which is fine
    tables = pd.read_html(url, storage_options={"User-Agent": USER_AGENT})
    df = tables[0]  # first table is the constituents list

    # Wikipedia column names: 'Symbol', 'Security', 'CIK'
    df = df.rename(columns={"Symbol": "ticker", "Security": "company_name", "CIK": "cik"})
    df = df[["ticker", "company_name", "cik"]].copy()

    # Wikipedia uses '.' in some tickers (e.g. BRK.B), SEC uses '-' (BRK-B)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    # Pad CIK to 10 digits
    df["cik"] = df["cik"].astype(str).str.zfill(10)
    df["source"] = "sp500"
    df["distress_label"] = 0
    df["distress_event_date"] = ""
    df["distress_event_type"] = ""

    print(f"      Got {len(df)} S&P 500 companies")
    return df


# ---------------------------------------------------------------------------
# Step 2: Load SEC ticker -> CIK master file
# ---------------------------------------------------------------------------
def load_sec_ticker_map():
    print("[2/3] Loading SEC ticker->CIK master file...")
    url = "https://www.sec.gov/files/company_tickers.json"
    r = requests.get(url, headers=HEADERS, timeout=30)
    r.raise_for_status()
    data = r.json()
    # Format: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
    rows = []
    for entry in data.values():
        rows.append({
            "ticker": entry["ticker"],
            "cik": str(entry["cik_str"]).zfill(10),
            "company_name": entry["title"],
        })
    df = pd.DataFrame(rows)
    print(f"      Got {len(df)} ticker->CIK mappings")
    return df


# ---------------------------------------------------------------------------
# Step 3: Curated list of 30 historical bankruptcies
# ---------------------------------------------------------------------------
BANKRUPTCIES = [
    # (company_name_hint, ticker, event_date, event_type)
    ("Lehman Brothers Holdings",     "LEHMQ", "2008-09-15", "chapter_11"),
    ("Washington Mutual",            "WAMUQ", "2008-09-26", "chapter_11"),
    ("General Motors",               "GM",    "2009-06-01", "chapter_11"),
    ("CIT Group",                    "CIT",   "2009-11-01", "chapter_11"),
    ("Borders Group",                "BGPIQ", "2011-02-16", "chapter_11"),
    ("MF Global Holdings",           "MFGLQ", "2011-10-31", "chapter_11"),
    ("Eastman Kodak",                "KODK",  "2012-01-19", "chapter_11"),
    ("Hostess Brands",               "TWNK",  "2012-11-16", "chapter_11"),
    ("RadioShack",                   "RSHCQ", "2015-02-05", "chapter_11"),
    ("Caesars Entertainment Operating Co", "CZR", "2015-01-15", "chapter_11"),
    ("SunEdison",                    "SUNEQ", "2016-04-21", "chapter_11"),
    ("Peabody Energy",               "BTU",   "2016-04-13", "chapter_11"),
    ("Toys R Us",                    "TOYS",  "2017-09-18", "chapter_11"),
    ("Sears Holdings",               "SHLDQ", "2018-10-15", "chapter_11"),
    ("PG&E Corporation",             "PCG",   "2019-01-29", "chapter_11"),
    ("Forever 21",                   "FRVR",  "2019-09-29", "chapter_11"),
    ("Hertz Global Holdings",        "HTZ",   "2020-05-22", "chapter_11"),
    ("J C Penney",                   "JCP",   "2020-05-15", "chapter_11"),
    ("Frontier Communications",      "FTR",   "2020-04-14", "chapter_11"),
    ("Chesapeake Energy",            "CHK",   "2020-06-28", "chapter_11"),
    ("Neiman Marcus",                "NMG",   "2020-05-07", "chapter_11"),
    ("Washington Prime Group",       "WPG",   "2021-06-13", "chapter_11"),
    ("Revlon",                       "REV",   "2022-06-15", "chapter_11"),
    ("Cineworld",                    "CNNWQ", "2022-09-07", "chapter_11"),
    ("Bed Bath & Beyond",            "BBBYQ", "2023-04-23", "chapter_11"),
    ("Silicon Valley Bank (SVB Financial)", "SIVBQ", "2023-03-17", "chapter_11"),
    ("WeWork",                       "WE",    "2023-11-06", "chapter_11"),
    ("Rite Aid",                     "RAD",   "2023-10-15", "chapter_11"),
    ("Spirit Airlines",              "SAVE",  "2024-11-18", "chapter_11"),
    ("Party City Holdco",            "PRTY",  "2023-01-17", "chapter_11"),
]


def resolve_bankruptcies(sec_map):
    print("[3/3] Resolving CIKs for 30 bankruptcies...")
    rows = []
    unresolved = []

    for name_hint, ticker, event_date, event_type in BANKRUPTCIES:
        # Try exact ticker match first
        match = sec_map[sec_map["ticker"] == ticker]
        if len(match) == 0:
            # Fallback: fuzzy name match (case-insensitive substring)
            hint_lower = name_hint.lower().split()[0]  # first word of hint
            match = sec_map[sec_map["company_name"].str.lower().str.contains(hint_lower, na=False)]
            if len(match) == 0:
                unresolved.append((name_hint, ticker))
                continue
            match = match.head(1)

        row = match.iloc[0]
        rows.append({
            "ticker": row["ticker"],
            "cik": row["cik"],
            "company_name": row["company_name"],
            "source": "bankruptcy",
            "distress_label": 1,
            "distress_event_date": event_date,
            "distress_event_type": event_type,
        })

    df = pd.DataFrame(rows)
    print(f"      Resolved {len(df)}/{len(BANKRUPTCIES)} bankruptcies")
    if unresolved:
        print(f"      WARNING: could not resolve {len(unresolved)}:")
        for name, tk in unresolved:
            print(f"        - {name} ({tk})")
        print("      Note: delisted bankrupt firms often vanish from the SEC ticker file.")
        print("      We'll handle these manually in a follow-up step if needed.")
    return df


# ---------------------------------------------------------------------------
# Combine + save
# ---------------------------------------------------------------------------
def main():
    sp500 = fetch_sp500()
    time.sleep(1)  # be polite
    sec_map = load_sec_ticker_map()
    time.sleep(1)
    bankrupt = resolve_bankruptcies(sec_map)

    print("\nCombining and deduplicating...")
    universe = pd.concat([sp500, bankrupt], ignore_index=True)

    # If a company appears in both, keep the bankruptcy row (it has the label)
    universe = universe.sort_values("distress_label", ascending=False)
    universe = universe.drop_duplicates(subset="cik", keep="first")
    universe = universe.sort_values(["distress_label", "ticker"], ascending=[False, True])
    universe = universe.reset_index(drop=True)

    universe.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(universe)} companies to {OUTPUT_PATH}")
    print(f"  Healthy (S&P 500):  {(universe['distress_label'] == 0).sum()}")
    print(f"  Distressed:         {(universe['distress_label'] == 1).sum()}")


if __name__ == "__main__":
    main()
