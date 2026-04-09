"""
SEC EDGAR HTTP client.

Handles polite scraping per SEC fair-access rules:
  - Required User-Agent header identifying the requester
  - Rate limit: max 10 requests/second (we use 5 to be safe)
  - Automatic retries with exponential backoff on transient failures

This module is intentionally agnostic of storage. It returns raw bytes
and lets the caller decide what to do with them.
"""
import os
import time
import logging
from threading import Lock
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class RateLimiter:
    """Simple thread-safe token-style rate limiter (max N requests per second)."""

    def __init__(self, max_per_second: float):
        self.min_interval = 1.0 / max_per_second
        self.last_call = 0.0
        self.lock = Lock()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.monotonic()


class SECClient:
    """HTTP client for SEC EDGAR endpoints."""

    BASE_DATA = "https://data.sec.gov"

    def __init__(self, user_agent: str, max_per_second: float = 5.0):
        if not user_agent:
            raise ValueError("SEC requires a User-Agent identifying the requester")
        self.user_agent = user_agent
        self.rate_limiter = RateLimiter(max_per_second)
        self.session = self._build_session()

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept-Encoding": "gzip, deflate",
            "Host": "data.sec.gov",
        })
        # Retry on 429 (rate limit), 500, 502, 503, 504
        retry = Retry(
            total=5,
            backoff_factor=1.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
            respect_retry_after_header=True,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        return session

    def _get(self, url: str, timeout: int = 30) -> Optional[requests.Response]:
        self.rate_limiter.wait()
        try:
            r = self.session.get(url, timeout=timeout)
            return r
        except requests.RequestException as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    def fetch_company_facts(self, cik: str) -> Optional[bytes]:
        """
        Fetch the XBRL company facts JSON for a given CIK.
        CIK must be a 10-digit zero-padded string.
        Returns raw bytes (JSON) or None on failure.
        """
        url = f"{self.BASE_DATA}/api/xbrl/companyfacts/CIK{cik}.json"
        r = self._get(url)
        if r is None or r.status_code != 200:
            code = r.status_code if r is not None else "no_response"
            logger.info(f"company_facts CIK={cik} status={code}")
            return None
        return r.content

    def fetch_submissions(self, cik: str) -> Optional[bytes]:
        """
        Fetch the submissions metadata JSON for a given CIK.
        Lists every filing the company has made.
        """
        url = f"{self.BASE_DATA}/submissions/CIK{cik}.json"
        r = self._get(url)
        if r is None or r.status_code != 200:
            code = r.status_code if r is not None else "no_response"
            logger.info(f"submissions CIK={cik} status={code}")
            return None
        return r.content
