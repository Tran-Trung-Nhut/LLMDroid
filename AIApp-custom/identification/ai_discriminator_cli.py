#!/usr/bin/env python3
"""
Wrapper CLI cho AI Discriminator (Li et al.) để tương thích với run_code_validation.py.

Usage:
    python ai_discriminator_cli.py --dir <decompiled_dir>

Output:
    1 (stdout) nếu phát hiện AI framework/model
    0 (stdout) nếu không phát hiện
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from identification_config import Config
from deep_model_identify import ai_model_format_identity
from extract_so_identity import main_identify_so
from keywords_identity import main_keywords_identity

Cf = Config()


def check_directory(decompiled_dir: str) -> bool:
    found = False

    try:
        result = ai_model_format_identity(decompiled_dir)
        if result:
            found = True
    except Exception:
        pass

    if not found:
        try:
            result = main_identify_so(decompiled_dir)
            if result:
                found = True
        except Exception:
            pass

    if not found:
        for root, _, files in os.walk(decompiled_dir):
            for fname in files:
                fpath = os.path.join(root, fname)
                try:
                    content = open(fpath, "r", errors="ignore").read()
                    for keyword in Cf.ai_code_key_words:
                        if keyword in content:
                            found = True
                            break
                except Exception:
                    pass
                if found:
                    break
            if found:
                break

    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True, help="Path to decompiled APK directory")
    args = parser.parse_args()

    if not Path(args.dir).exists():
        print(0)
        sys.exit(0)

    detected = check_directory(args.dir)
    print(1 if detected else 0)


if __name__ == "__main__":
    main()
