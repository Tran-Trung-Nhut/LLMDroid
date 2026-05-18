"""
fetch_last_updated.py — Fetch last_updated date from Google Play for all apps in apps.jsonl.

Adds 'last_updated' field (YYYY-MM-DD) to each row and updates both apps.jsonl and apps_raw.jsonl.
Run once before temporal_split.py.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

from tqdm import tqdm
from fetch_app_metadata import AppMetadataFetcher
from config import CFG


def unix_to_date(ts) -> str | None:
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return None


def fetch_dates(jsonl_path: str) -> dict[str, str]:
    """Fetch last_updated for all app_ids in a JSONL file. Returns {app_id: date_str}."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    fetcher = AppMetadataFetcher()
    date_map = {}

    # Pre-fill from existing field to skip already-fetched apps
    for row in rows:
        existing = row.get("last_updated")
        if existing:
            date_map[row["app_id"]] = existing

    to_fetch = [r for r in rows if r["app_id"] not in date_map]
    print(f"Apps to fetch: {len(to_fetch)} / {len(rows)}")

    for row in tqdm(to_fetch, desc="Fetching dates"):
        app_id = row["app_id"]
        play_data = fetcher.fetch_play_store_metadata(app_id)
        if play_data:
            ts = play_data.get("updated")
            date_str = unix_to_date(ts) if ts else None
            if date_str:
                date_map[app_id] = date_str
                tqdm.write(f"  {app_id}: {date_str}")
            else:
                tqdm.write(f"  {app_id}: no date")
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
        if app_id in date_map and row.get("last_updated") != date_map[app_id]:
            row["last_updated"] = date_map[app_id]
            updated += 1

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return updated


def main():
    print("=== Fetching last_updated dates from Google Play ===\n")
    date_map = fetch_dates(CFG.dataset_path)

    print(f"\nFetched {len(date_map)} dates total")

    for path in [CFG.dataset_path, CFG.raw_dataset_path]:
        n = inject_dates(path, date_map)
        print(f"Updated {n} rows in {path}")

    # Summary
    d_cut = datetime.strptime(CFG.temporal_d_cut, "%Y-%m-%d")
    before = sum(1 for d in date_map.values() if datetime.strptime(d, "%Y-%m-%d") <= d_cut)
    after  = sum(1 for d in date_map.values() if datetime.strptime(d, "%Y-%m-%d") >  d_cut)
    missing = 298 - len(date_map)
    print(f"\nD_cut={CFG.temporal_d_cut}: train (<=)={before}, test (>)={after}, no date={missing}")


if __name__ == "__main__":
    main()
