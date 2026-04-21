#!/usr/bin/env python3
"""Analyze inference results against manual labels and produce plots/metrics.

Usage:
  python src/analyze_inference_results.py --inference-dir inference_results \
      --manual-file data/test_set_manual.csv --out-dir inference_analysis

The script expects inference CSVs to have (at least) columns:
  - `app_id` (package name), `y_prob` (probability for positive class), `prediction_label` (0/1)

And the manual labels file to have (at least) columns:
  - `pkg_name` (package name), `label` (0/1)

The script will create a subfolder per inference file under the output folder
and save confusion matrix, ROC, PR curve, probability histogram, and a
classification report + summary metrics.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List

import pandas as pd

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:  # pragma: no cover - handled at runtime
    plt = None
    sns = None

from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_curve,
    classification_report,
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def analyze_one(
    inf_path: str,
    manual_df: pd.DataFrame,
    out_dir: str,
    id_col: str = "app_id",
    manual_id: str = "pkg_name",
    manual_label: str = "label",
    prob_col: str = "y_prob",
    pred_col: str = "prediction_label",
):
    basename = os.path.splitext(os.path.basename(inf_path))[0]
    out_sub = os.path.join(out_dir, basename)
    ensure_dir(out_sub)

    df_inf = pd.read_csv(inf_path)

    # fallback: if id_col not present, try first column
    if id_col not in df_inf.columns:
        id_col = df_inf.columns[0]

    # Merge predictions with manual labels
    merged = df_inf.merge(manual_df, left_on=id_col, right_on=manual_id, how="inner")

    matched = len(merged)
    total_inf = len(df_inf)
    unmatched_inf_df = df_inf[~df_inf[id_col].isin(merged[id_col])]
    missing_manual_df = manual_df[~manual_df[manual_id].isin(merged[manual_id])]

    if matched == 0:
        summary = {
            "file": basename,
            "total_inference": int(total_inf),
            "matched": 0,
            "unmatched_inference": int(len(unmatched_inf_df)),
            "missing_manual": int(len(missing_manual_df)),
        }
        with open(os.path.join(out_sub, f"{basename}_summary.json"), "w") as fh:
            json.dump(summary, fh, indent=2)
        unmatched_inf_df.to_csv(os.path.join(out_sub, f"{basename}_unmatched_inference.csv"), index=False)
        missing_manual_df.to_csv(os.path.join(out_sub, f"{basename}_missing_in_manual.csv"), index=False)
        print(f"No matches for {basename} (check id columns). Saved unmatched lists.")
        return summary

    y_true = merged[manual_label].astype(int)
    y_pred = merged[pred_col].astype(int)
    y_prob = merged[prob_col].astype(float)
    roc_auc = None
    ap = None

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if plt is None or sns is None:
        print("matplotlib or seaborn not installed. Skipping plots.")
    else:
        sns.set(style="whitegrid")
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"{basename} Confusion Matrix")
        plt.tight_layout()
        plt.savefig(os.path.join(out_sub, f"{basename}_confusion_matrix.png"))
        plt.close()

        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(6, 4))
            plt.plot(fpr, tpr, label=f"AUC={roc_auc:.4f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"{basename} ROC")
            plt.legend(loc="lower right")
            plt.tight_layout()
            plt.savefig(os.path.join(out_sub, f"{basename}_roc.png"))
            plt.close()
        except Exception as e:
            print("ROC plotting failed:", e)

        # Precision-Recall
        try:
            precision, recall, _ = precision_recall_curve(y_true, y_prob)
            ap = average_precision_score(y_true, y_prob)
            plt.figure(figsize=(6, 4))
            plt.plot(recall, precision, label=f"AP={ap:.4f}")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"{basename} Precision-Recall")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(out_sub, f"{basename}_pr.png"))
            plt.close()
        except Exception as e:
            print("PR plotting failed:", e)

        # Probability histogram
        plt.figure(figsize=(6, 4))
        plt.hist(y_prob, bins=50, alpha=0.7)
        plt.title(f"{basename} Probability Distribution")
        plt.xlabel("y_prob")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_sub, f"{basename}_yprob_hist.png"))
        plt.close()

    # Classification report + CSV
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(os.path.join(out_sub, f"{basename}_classification_report.csv"))

    # Save unmatched / missing lists
    unmatched_inf_df.to_csv(os.path.join(out_sub, f"{basename}_unmatched_inference.csv"), index=False)
    missing_manual_df.to_csv(os.path.join(out_sub, f"{basename}_missing_in_manual.csv"), index=False)

    # Summary metrics
    summary: Dict[str, object] = {
        "file": basename,
        "total_inference": int(total_inf),
        "matched": int(matched),
        "unmatched_inference": int(len(unmatched_inf_df)),
        "missing_manual": int(len(missing_manual_df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc) if roc_auc is not None else None,
        "average_precision": float(ap) if ap is not None else None,
    }

    with open(os.path.join(out_sub, f"{basename}_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return summary


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Analyze inference results against manual labels.")
    parser.add_argument("--inference-dir", required=True, help="Folder with inference CSVs")
    parser.add_argument("--manual-file", required=True, help="Manual labels CSV")
    parser.add_argument("--out-dir", required=True, help="Output folder for plots and metrics")
    parser.add_argument("--id-col", default="app_id", help="ID column name in inference CSVs (default: app_id)")
    parser.add_argument("--manual-id", default="pkg_name", help="ID column name in manual CSV (default: pkg_name)")
    parser.add_argument("--manual-label", default="label", help="Label column name in manual CSV (default: label)")
    parser.add_argument("--prob-col", default="y_prob", help="Probability column name in inference CSVs (default: y_prob)")
    parser.add_argument("--pred-col", default="prediction_label", help="Prediction label column name in inference CSVs (default: prediction_label)")
    args = parser.parse_args(args=argv)

    if not os.path.isdir(args.inference_dir):
        parser.error(f"Inference directory not found: {args.inference_dir}")
    if not os.path.isfile(args.manual_file):
        parser.error(f"Manual labels file not found: {args.manual_file}")

    if plt is None or sns is None:
        print("Warning: matplotlib or seaborn not available. Install them to get plots:")
        print("  pip install matplotlib seaborn")

    manual_df = pd.read_csv(args.manual_file)

    ensure_dir(args.out_dir)

    inference_files = sorted(glob.glob(os.path.join(args.inference_dir, "*.csv")))
    if not inference_files:
        print("No CSV files found in", args.inference_dir)
        return

    summaries: List[Dict[str, object]] = []
    for inf in inference_files:
        print("Analyzing", inf)
        try:
            s = analyze_one(
                inf,
                manual_df,
                args.out_dir,
                id_col=args.id_col,
                manual_id=args.manual_id,
                manual_label=args.manual_label,
                prob_col=args.prob_col,
                pred_col=args.pred_col,
            )
            summaries.append(s)
        except Exception as e:
            print(f"Failed analyzing {inf}: {e}")

    pd.DataFrame(summaries).to_csv(os.path.join(args.out_dir, "summary_metrics.csv"), index=False)
    print("Done. Results saved to", args.out_dir)


if __name__ == "__main__":
    main()
