"""
Resolve CIKs for delisted/bankrupt companies that don't appear in the
SEC active ticker file. Uses EDGAR's full-text company search.
"""
import os
import re
import sys
import time
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
USER_AGENT = os.getenv("SEC_USER_AGENT")
HEADERS = {"User-Agent": USER_AGENT}

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UNIVERSE_PATH = PROJECT_ROOT / "config" / "company_universe.csv"

# Companies still missing — we'll search EDGAR by company name
# Format: (search_name, fallback_ticker, event_date, event_type)
MISSING = [
    ("Borders Group",     "BGPIQ", "2011-02-16", "chapter_11"),
    ("Hostess Brands",    "TWNK",  "2012-11-16", "chapter_11"),
    ("RadioShack",        "RSHCQ", "2015-02-05", "chapter_11"),
    ("SunEdison",         "SUNEQ", "2016-04-21", "chapter_11"),
    ("Toys R Us",         "TOYS",  "2017-09-18", "chapter_11"),
    ("Sears Holdings",    "SHLDQ", "2018-10-15", "chapter_11"),
    ("Forever 21",        "FRVR",  "2019-09-29", "chapter_11"),
    ("Revlon",            "REV",   "2022-06-15", "chapter_11"),
    ("Cineworld",         "CNNWQ", "2022-09-07", "chapter_11"),
    ("WeWork",            "WE",    "2023-11-06", "chapter_11"),
    ("Party City Holdco", "PRTY",  "2023-01-17", "chapter_11"),
]


def search_edgar_by_name(company_name):
    """
    Query EDGAR's company browser by name. Returns list of (cik, name) candidates.
    """
    url = "https://www.sec.gov/cgi-bin/browse-edgar"
    params = {
        "action": "getcompany",
        "company": company_name,
        "type": "10-K",        # filter to companies that filed 10-Ks
        "dateb": "",
        "owner": "include",
        "count": "20",
        "output": "atom",
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=15)
    if r.status_code != 200:
        return []

    # Parse CIKs and names from the Atom feed
    text = r.text
    # CIKs appear in URLs like /cgi-bin/browse-edgar?action=getcompany&CIK=0000123456
    cik_pattern = re.compile(r"CIK=(\d{10})")
    name_pattern = re.compile(r"<name>([^<]+)</name>")
    ciks = cik_pattern.findall(text)
    names = name_pattern.findall(text)

    # Dedupe while preserving order; first <name> entry is the feed itself, skip it
    seen = set()
    results = []
    for i, cik in enumerate(ciks):
        if cik in seen:
            continue
        seen.add(cik)
        # The feed has one <name> per entry; offset by 1 for the feed-level <name>
        name = names[i + 1] if i + 1 < len(names) else ""
        results.append((cik, name))
    return results


def main():
    universe = pd.read_csv(UNIVERSE_PATH, dtype={"cik": str})
    print(f"Current universe: {len(universe)} companies\n")

    new_rows = []
    still_missing = []

    for search_name, ticker, event_date, event_type in MISSING:
        print(f"Searching: {search_name}")
        candidates = search_edgar_by_name(search_name)
        time.sleep(0.2)  # SEC rate limit (max 10 req/s)

        if not candidates:
            print(f"  No results")
            still_missing.append(search_name)
            continue

        print(f"  Found {len(candidates)} candidate(s):")
        for i, (cik, name) in enumerate(candidates[:5]):
            print(f"    [{i}] CIK={cik}  {name}")

        # Auto-pick the first candidate (usually the right one for well-known names)
        cik, name = candidates[0]
        print(f"  -> Using [0]: CIK={cik}  {name}\n")

        new_rows.append({
            "ticker": ticker,
            "cik": cik,
            "company_name": name,
            "source": "bankruptcy",
            "distress_label": 1,
            "distress_event_date": event_date,
            "distress_event_type": event_type,
        })

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Append, dedupe by CIK (prefer rows with distress_label = 1)
        combined = pd.concat([universe, new_df], ignore_index=True)
        combined = combined.sort_values("distress_label", ascending=False)
        combined = combined.drop_duplicates(subset="cik", keep="first")
        combined = combined.sort_values(["distress_label", "ticker"], ascending=[False, True])
        combined = combined.reset_index(drop=True)

        combined.to_csv(UNIVERSE_PATH, index=False)
        print(f"\nUpdated universe saved: {len(combined)} companies")
        print(f"  Healthy (S&P 500):  {(combined['distress_label'] == 0).sum()}")
        print(f"  Distressed:         {(combined['distress_label'] == 1).sum()}")

    if still_missing:
        print(f"\nStill could not resolve: {still_missing}")
        print("These will need manual lookup via EDGAR website.")


if __name__ == "__main__":
    main()
