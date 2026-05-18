#!/usr/bin/env python3
import importlib
import os
import sys
import traceback
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
os.chdir(_PROJECT_ROOT)
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from config import CFG


def _section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def _run(label: str, module_path: str) -> bool:
    """Import and call main() of a steps module. Returns True on success."""
    _section(label)
    try:
        mod = importlib.import_module(module_path)
        importlib.reload(mod)
        mod.main()
        print(f"\n[OK] {label}")
        return True
    except Exception:
        print(f"\n[FAIL] {label}")
        traceback.print_exc()
        return False


def main():
    results = {}

    # 6.1 — Statistical tests (Table 9)
    results["6.1 Statistical tests"] = _run(
        "6.1 — Statistical tests (Table 9)",
        "steps.statistical_tests",
    )

    # 6.2 — Branch complementarity (Table 10 + Figure 4)
    results["6.2 Branch complementarity"] = _run(
        "6.2 — Branch complementarity (Table 10 + Figure 4)",
        "steps.branch_complementarity",
    )

    # 6.3 — Disagreement accuracy (Table 11)
    results["6.3 Disagreement accuracy"] = _run(
        "6.3 — Disagreement accuracy (Table 11)",
        "steps.disagreement_accuracy",
    )

    # 6.4 — Per-category performance (Table 13) — requires step 5c
    results["6.4 Per-category"] = _run(
        "6.4 — Per-category performance (Table 13)",
        "steps.per_category",
    )

    # 6.5 — Prior-corrected precision (Table 14 + Figure 5) — requires step 5c
    results["6.5 Prior correction"] = _run(
        "6.5 — Prior-corrected precision (Table 14 + Figure 5)",
        "steps.prior_correction",
    )

    # 6.6 — Temporal split (Table 15) — requires last_updated in apps.jsonl
    results["6.6 Temporal split"] = _run(
        "6.6 — Temporal split (Table 15)",
        "steps.temporal_split",
    )

    # 6.7 — Inference latency (Table 16)
    results["6.7 Latency benchmark"] = _run(
        "6.7 — Inference latency (Table 16)",
        "steps.latency_benchmark",
    )

    # 6.8 — Probability calibration (Table 17)
    results["6.8 Calibration"] = _run(
        "6.8 — Probability calibration (Table 17)",
        "steps.calibration",
    )

    # 6.9 — Robustness (Table 18) — extract trunc50 features first if needed
    trunc50_path = Path(CFG.features_test_trunc50_dir) / "text" / "features.npz"
    if not trunc50_path.exists():
        _run(
            "6.9 — Extract truncated text features (trunc50 pre-step)",
            "steps.extract_text_features_trunc50",
        )
    results["6.9 Robustness"] = _run(
        "6.9 — Robustness to missing modalities (Table 18)",
        "steps.robustness",
    )

    # Summary
    _section("SUMMARY")
    ok = [k for k, v in results.items() if v]
    fail = [k for k, v in results.items() if not v]
    print(f"Passed: {len(ok)}/{len(results)}")
    for k in ok:
        print(f"  [OK]   {k}")
    for k in fail:
        print(f"  [FAIL] {k}")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
