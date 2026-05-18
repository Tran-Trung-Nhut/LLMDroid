import csv
import json
from pathlib import Path
import pandas as pd


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def write_predictions_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def load_label_map(csv_path: str) -> dict:
    """Return {pkg_name: label} from a CSV with columns pkg_name, label."""
    result = {}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            result[row["pkg_name"]] = int(row["label"])
    return result
