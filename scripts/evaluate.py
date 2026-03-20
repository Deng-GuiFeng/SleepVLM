#!/usr/bin/env python3
"""
Evaluate sleep-staging inference results from a JSONL file.

Reads a JSONL produced by the inference pipeline (one JSON object per epoch)
and computes:
  - Per-subject metrics: accuracy, macro-F1, Cohen's kappa, per-class F1,
    confusion matrix, and mean rules IoU.
  - Overall (aggregate) metrics across all subjects.

Results are saved as JSON and a human-readable summary is printed to stdout.

Usage:

  python scripts/evaluate.py \
      --results_jsonl runs/inference/results.jsonl \
      --output_dir    runs/inference/eval_output

Expected JSONL schema (per line):
  {
    "custom_id": "MASS-SS1/01-01-0001#100_N2",
    "sub_id": "MASS-SS1/01-01-0001",
    "label": 2,                    // ground-truth stage index
    "pred": 2,                     // predicted stage index (-1 on parse failure)
    "rules_iou": 0.75,             // optional
    "applicable_rules": "N2.1",    // optional (predicted)
    "gt_applicable_rules": [...]   // optional (ground truth)
    ...
  }
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sleepvlm.evaluation.metrics import (
    compute_overall_metrics,
    compute_rules_iou,
    compute_subject_metrics,
    save_metrics_json,
)

# Stage index to human-readable name.
_IDX_TO_STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute sleep-staging evaluation metrics from a JSONL "
                    "results file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_jsonl", type=str, required=True,
        help="Path to the inference results JSONL file.",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Directory where evaluation output files will be written.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Load results
# ---------------------------------------------------------------------------

def load_results(jsonl_path: str) -> pd.DataFrame:
    """Read the JSONL file and return a DataFrame.

    Each line must be a valid JSON object with at least ``sub_id``, ``label``,
    and ``pred`` keys.  Malformed lines are skipped with a warning.
    """
    records: List[dict] = []
    with open(jsonl_path, "r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[warn] Skipping malformed JSON at line {line_no}: {exc}")

    if not records:
        print("[error] No valid records found in the JSONL file.")
        sys.exit(1)

    df = pd.DataFrame(records)

    # Validate required columns.
    for col in ("sub_id", "label", "pred"):
        if col not in df.columns:
            print(f"[error] Required column '{col}' missing from JSONL.")
            sys.exit(1)

    return df


# ---------------------------------------------------------------------------
# Compute per-subject metrics
# ---------------------------------------------------------------------------

def compute_per_subject(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Return a list of per-subject metric dicts."""
    subject_results: List[Dict[str, Any]] = []

    for sub_id, group in df.groupby("sub_id"):
        y_true = group["label"].values
        y_pred = group["pred"].values

        metrics = compute_subject_metrics(y_true, y_pred)

        # Compute mean rules IoU for this subject if available.
        mean_iou = float("nan")
        if "rules_iou" in group.columns:
            valid_labels = [0, 1, 2, 3, 4]
            valid_mask = np.isin(y_pred, valid_labels) & np.isin(y_true, valid_labels)
            iou_vals = group.loc[valid_mask, "rules_iou"].values
            if len(iou_vals) > 0:
                mean_iou = float(np.nanmean(iou_vals))

        subject_results.append({
            "subject": str(sub_id),
            "n_epochs": len(group),
            **metrics,
            "mean_rules_iou": mean_iou,
        })

    return subject_results


# ---------------------------------------------------------------------------
# Print a human-readable summary
# ---------------------------------------------------------------------------

def print_summary(
    overall: Dict[str, Any],
    subject_results: List[Dict[str, Any]],
    n_total: int,
) -> None:
    """Print a concise summary to stdout."""
    sep = "=" * 60
    print()
    print(sep)
    print("Evaluation Summary")
    print(sep)
    print(f"  Total epochs  : {n_total}")
    print(f"  Valid epochs  : {overall.get('n_valid', 'N/A')}")
    print(f"  Invalid epochs: {overall.get('n_invalid', 'N/A')}")
    print(f"  Subjects      : {len(subject_results)}")
    print()
    print(f"  Accuracy : {overall['accuracy']:.4f}")
    print(f"  Macro-F1 : {overall['macro_f1']:.4f}")
    print(f"  Kappa    : {overall['kappa']:.4f}")

    if "mean_rules_iou" in overall and not np.isnan(overall["mean_rules_iou"]):
        print(f"  Rules IoU: {overall['mean_rules_iou']:.4f}")

    print()
    print("  Per-class F1:")
    pcf1 = overall.get("per_class_f1", {})
    for cls_name in ["W", "N1", "N2", "N3", "REM"]:
        val = pcf1.get(cls_name, float("nan"))
        print(f"    {cls_name:>3s}: {val:.4f}")

    print()
    print("  Confusion matrix (rows=true, cols=pred):")
    cm = overall.get("confusion_matrix", [])
    header = "        " + "  ".join(f"{_IDX_TO_STAGE.get(i, '?'):>5s}" for i in range(5))
    print(header)
    for i, row in enumerate(cm):
        label = _IDX_TO_STAGE.get(i, "?")
        row_str = "  ".join(f"{int(v):5d}" for v in row)
        print(f"    {label:>3s}: {row_str}")

    print(sep)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.results_jsonl):
        print(f"[error] File not found: {args.results_jsonl}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data.
    print(f"Loading results from {args.results_jsonl} ...")
    df = load_results(args.results_jsonl)
    print(f"  Loaded {len(df)} records across {df['sub_id'].nunique()} subjects.")

    # Ensure sub_id has a center prefix (e.g. "MASS-SS1/01-01-0001").
    # If sub_id is a bare ID like "01-01-0001", infer the center.
    SUBSET_MAP = {
        "01-01": "MASS-SS1", "01-02": "MASS-SS2", "01-03": "MASS-SS3",
        "01-04": "MASS-SS4", "01-05": "MASS-SS5",
    }
    if "/" not in str(df["sub_id"].iloc[0]):
        df["sub_id"] = df["sub_id"].apply(
            lambda sid: f"{SUBSET_MAP.get(str(sid)[:5], 'unknown')}/{sid}"
        )
        print(f"  Inferred center prefix -> {df['sub_id'].iloc[0].split('/')[0]}")

    # Recompute rules_iou from raw fields if the column is absent but the
    # source data provides both predicted and ground-truth rules.
    if "rules_iou" not in df.columns:
        if "applicable_rules" in df.columns and "gt_applicable_rules" in df.columns:
            print("  Computing rules IoU from applicable_rules fields ...")
            df["rules_iou"] = df.apply(
                lambda row: compute_rules_iou(
                    row.get("applicable_rules"),
                    row.get("gt_applicable_rules"),
                ),
                axis=1,
            )

    # Per-subject metrics.
    print("Computing per-subject metrics ...")
    subject_results = compute_per_subject(df)

    # Overall metrics.
    print("Computing overall metrics ...")
    overall = compute_overall_metrics(df)

    # Save per-center JSON files (one per dataset/center).
    print("Saving per-center JSON files ...")
    save_metrics_json(df, args.output_dir)

    # Save a combined results JSON with both per-subject and overall metrics.
    combined_output = {
        "num_subjects": len(subject_results),
        "num_epochs": len(df),
        "overall": overall,
        "subjects": subject_results,
    }
    combined_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(combined_path, "w", encoding="utf-8") as fh:
        json.dump(combined_output, fh, ensure_ascii=False, indent=2)
    print(f"  Combined results saved to {combined_path}")

    # Print human-readable summary.
    print_summary(overall, subject_results, n_total=len(df))


if __name__ == "__main__":
    main()
