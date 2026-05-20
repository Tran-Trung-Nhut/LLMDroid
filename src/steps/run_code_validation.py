"""
run_code_validation.py — Table 2: N_code=80 code-level validation with AI Discriminator.

Downloads APKs from Androzoo, decompiles with apktool (bundled), runs AI Discriminator,
and writes data/code_validation.csv for cohen_kappa_validation.py.

Setup:
    pip install requests
    # Androzoo metadata (~3 GB):
    wget https://androzoo.uni.lu/static/lists/latest.csv.gz
    gunzip latest.csv.gz && mv latest.csv data/androzoo_latest.csv

    export ANDROZOO_API_KEY=your_key_here
    python src/steps/run_code_validation.py
"""
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR.parent) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR.parent))

ANDROZOO_DOWNLOAD_URL = "https://androzoo.uni.lu/api/download"
ANDROZOO_CSV          = "data/androzoo_latest.csv"
APPS_CSV              = "data/code_validation_apps.csv"
OUT_CSV               = "data/code_validation.csv"
APK_DIR               = Path("data/apks")
DECOMPILE_DIR         = Path("data/decompiled")
CHECKPOINT            = Path("data/code_validation_checkpoint.json")

AI_DISC_CLI  = _PROJECT_ROOT / "AIApp-custom" / "identification" / "ai_discriminator_cli.py"
AI_DISC_BIN  = f"python {AI_DISC_CLI}"
APKTOOL_JAR  = _PROJECT_ROOT / "AIApp-custom" / "identification" / "apktool_2.5.0.jar"


# ── Androzoo helpers ──────────────────────────────────────────────────────────

def load_pkg2sha(target_pkgs: set) -> dict:
    """Scan Androzoo metadata CSV for target packages; keep latest version per pkg."""
    pkg2sha, pkg2date = {}, {}
    print(f"Scanning {ANDROZOO_CSV} for {len(target_pkgs)} packages...")
    with open(ANDROZOO_CSV, encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            pkg = row.get("pkg_name", "").strip()
            if pkg not in target_pkgs:
                continue
            date = row.get("dex_date", "")
            if pkg not in pkg2date or date > pkg2date[pkg]:
                pkg2date[pkg] = date
                pkg2sha[pkg]  = row["sha256"].strip()
    print(f"  Found: {len(pkg2sha)}/{len(target_pkgs)}")
    return pkg2sha


def download_apk(api_key: str, sha256: str, out_path: Path) -> bool:
    for attempt in range(4):
        try:
            r = requests.get(ANDROZOO_DOWNLOAD_URL,
                             params={"apikey": api_key, "sha256": sha256},
                             stream=True, timeout=180)
            if r.status_code == 200:
                out_path.write_bytes(r.content)
                return True
            print(f"    HTTP {r.status_code}")
        except Exception as e:
            print(f"    attempt {attempt+1} error: {e}")
        time.sleep(2 ** attempt)
    return False


# ── Decompile ─────────────────────────────────────────────────────────────────

def decompile(apk_path: Path, out_dir: Path) -> bool:
    if out_dir.exists() and any(out_dir.iterdir()):
        return True
    out_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["java", "-jar", str(APKTOOL_JAR), "d", "-f",
         str(apk_path), "-o", str(out_dir)],
        capture_output=True, timeout=300,
    )
    return result.returncode == 0


# ── AI Discriminator ──────────────────────────────────────────────────────────

def run_ai_discriminator(decompiled_dir: Path) -> int:
    result = subprocess.run(
        AI_DISC_BIN.split() + ["--dir", str(decompiled_dir)],
        capture_output=True, text=True, timeout=600,
    )
    for line in reversed(result.stdout.strip().splitlines()):
        if line.strip() in ("0", "1"):
            return int(line.strip())
    return -1


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT.exists():
        return json.loads(CHECKPOINT.read_text())
    return {}


def save_checkpoint(done: dict):
    CHECKPOINT.write_text(json.dumps(done, indent=2))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    api_key = os.environ.get("ANDROZOO_API_KEY", "")
    if not api_key:
        print("[error] export ANDROZOO_API_KEY=your_key")
        sys.exit(1)

    if not Path(ANDROZOO_CSV).exists():
        print(f"[error] {ANDROZOO_CSV} not found.")
        print("  Download: wget https://androzoo.uni.lu/static/lists/latest.csv.gz")
        print("  Then: gunzip latest.csv.gz && mv latest.csv data/androzoo_latest.csv")
        sys.exit(1)

    apps = []
    with open(APPS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            apps.append({"pkg": row["pkg_name"], "listing_label": int(row["listing_label"])})
    print(f"Target: {len(apps)} apps")

    target_pkgs = {a["pkg"] for a in apps}
    pkg2sha = load_pkg2sha(target_pkgs)

    APK_DIR.mkdir(parents=True, exist_ok=True)
    DECOMPILE_DIR.mkdir(parents=True, exist_ok=True)

    done = load_checkpoint()
    results = []

    for i, app in enumerate(apps, 1):
        pkg   = app["pkg"]
        label = app["listing_label"]
        print(f"\n[{i}/{len(apps)}] {pkg} (label={label})")

        if pkg in done:
            print("  [checkpoint] skipping")
            results.append(done[pkg])
            continue

        row = {"pkg_name": pkg, "listing_label": label, "ai_discriminator_label": -1}

        sha256 = pkg2sha.get(pkg)
        if not sha256:
            print("  [skip] not found in Androzoo")
            row["note"] = "not_in_androzoo"
            done[pkg] = row
            save_checkpoint(done)
            results.append(row)
            continue

        apk_path = APK_DIR / f"{pkg}.apk"
        if not apk_path.exists():
            print(f"  Downloading ({sha256[:12]}...)...", end=" ", flush=True)
            ok = download_apk(api_key, sha256, apk_path)
            print("ok" if ok else "FAILED")
            if not ok:
                row["note"] = "download_failed"
                done[pkg] = row
                save_checkpoint(done)
                results.append(row)
                continue
        else:
            print("  APK already exists")

        dec_dir = DECOMPILE_DIR / pkg
        print(f"  Decompiling...", end=" ", flush=True)
        ok = decompile(apk_path, dec_dir)
        print("ok" if ok else "FAILED")
        if not ok:
            row["note"] = "decompile_failed"
            done[pkg] = row
            save_checkpoint(done)
            results.append(row)
            continue

        print(f"  AI Discriminator...", end=" ", flush=True)
        row["ai_discriminator_label"] = run_ai_discriminator(dec_dir)
        print(row["ai_discriminator_label"])

        done[pkg] = row
        save_checkpoint(done)
        results.append(row)

    valid = [r for r in results if r.get("ai_discriminator_label", -1) != -1]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pkg_name", "listing_label", "ai_discriminator_label"])
        writer.writeheader()
        writer.writerows([{k: r.get(k, -1) for k in writer.fieldnames} for r in valid])

    n_total = len(results)
    n_androzoo = sum(1 for r in results if r.get("note") != "not_in_androzoo")
    n_done = len(valid)
    print(f"\nSummary:")
    print(f"  Total targets          : {n_total}")
    print(f"  Found on Androzoo      : {n_androzoo}")
    print(f"  AI Discriminator ran   : {n_done}")
    print(f"  Output: {OUT_CSV}")
    print(f"\nNext: python src/steps/cohen_kappa_validation.py")


if __name__ == "__main__":
    main()
