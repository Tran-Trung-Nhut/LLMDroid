import importlib
import sys
import traceback


def section(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print('=' * 60)


def run_step(label: str, module_path: str) -> bool:
    section(label)
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


def print_summary(results: dict) -> None:
    section("SUMMARY")
    ok   = [k for k, v in results.items() if v]
    fail = [k for k, v in results.items() if not v]
    print(f"Passed: {len(ok)}/{len(results)}")
    for k in ok:
        print(f"  [OK]   {k}")
    for k in fail:
        print(f"  [FAIL] {k}")
    if fail:
        sys.exit(1)
