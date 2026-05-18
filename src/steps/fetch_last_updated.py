"""
fetch_last_updated.py — Fetch release_date from Google Play for all apps in apps.jsonl.

Stores 'release_date' (YYYY-MM-DD, parsed from the 'released' field on Google Play) into
apps.jsonl and apps_raw.jsonl. Run once before temporal_split.py.

Using 'released' (original launch date) rather than 'updated' (last update) because:
- 'updated' changes every time we fetch — not reproducible
- 'released' is stable and reflects when the app first appeared on the market
"""
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from tqdm import tqdm
from fetch_app_metadata import AppMetadataFetcher
from config import CFG

# google_play_scraper returns 'released' as e.g. "Mar 21, 2019"
_RELEASED_FMT = "%b %d, %Y"


def parse_released(released_str: str) -> str | None:
    try:
        return datetime.strptime(released_str.strip(), _RELEASED_FMT).strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_release_dates(jsonl_path: str) -> dict[str, str]:
    """Returns {app_id: 'YYYY-MM-DD'} using Google Play 'released' field."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    fetcher = AppMetadataFetcher()
    date_map = {}

    # Skip apps already fetched
    for row in rows:
        existing = row.get("release_date")
        if existing:
            date_map[row["app_id"]] = existing

    to_fetch = [r for r in rows if r["app_id"] not in date_map]
    print(f"Apps to fetch: {len(to_fetch)} / {len(rows)}")

    for row in tqdm(to_fetch, desc="Fetching release dates"):
        app_id = row["app_id"]
        play_data = fetcher.fetch_play_store_metadata(app_id)
        if play_data:
            released_str = play_data.get("released", "")
            date_str = parse_released(released_str) if released_str else None
            if date_str:
                date_map[app_id] = date_str
                tqdm.write(f"  {app_id}: {date_str}")
            else:
                tqdm.write(f"  {app_id}: no released field")
        else:
            tqdm.write(f"  {app_id}: fetch failed")
        time.sleep(CFG.api_request_delay)

    return date_map


def inject_dates(jsonl_path: str, date_map: dict[str, str]) -> int:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    updated = 0
    for row in rows:
        app_id = row["app_id"]
        if app_id in date_map and row.get("release_date") != date_map[app_id]:
            row["release_date"] = date_map[app_id]
            updated += 1

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return updated


def main():
    print("=== Fetching release_date (original launch date) from Google Play ===\n")
    date_map = fetch_release_dates(CFG.dataset_path)

    print(f"\nFetched {len(date_map)} release dates")

    for path in [CFG.dataset_path, CFG.raw_dataset_path]:
        n = inject_dates(path, date_map)
        print(f"Updated {n} rows in {path}")

    # Distribution around D_cut
    d_cut = datetime.strptime(CFG.temporal_d_cut, "%Y-%m-%d")
    before = sum(1 for d in date_map.values() if datetime.strptime(d, "%Y-%m-%d") <= d_cut)
    after  = sum(1 for d in date_map.values() if datetime.strptime(d, "%Y-%m-%d") >  d_cut)
    missing = len([r for r in open(CFG.dataset_path)]) - len(date_map)
    print(f"\nD_cut={CFG.temporal_d_cut}: train (<=)={before}, test (>)={after}, no date={missing}")


if __name__ == "__main__":
    main()
