"""
Metrics computation for sleep-staging evaluation.

Provides per-subject metrics (accuracy, macro-F1, Cohen's kappa,
per-class F1, confusion matrix), aggregation across subjects,
rule-set IoU, and JSON serialisation utilities.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
)

# Canonical label set used by all metric helpers.
_LABELS: List[int] = [0, 1, 2, 3, 4]
_LABEL_NAMES: List[str] = ["W", "N1", "N2", "N3", "REM"]


# ---------------------------------------------------------------------------
# Per-subject metrics
# ---------------------------------------------------------------------------

def compute_subject_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Any]:
    """Compute classification metrics for a single subject.

    Only epochs where both *y_true* and *y_pred* belong to the valid label set
    ``{0, 1, 2, 3, 4}`` are considered.  Epochs with ``label == -1`` (unscorable)
    or ``pred == -1`` (parse failure) are excluded automatically.

    Parameters
    ----------
    y_true : array-like of int
        Ground-truth sleep-stage indices.
    y_pred : array-like of int
        Predicted sleep-stage indices.

    Returns
    -------
    dict
        Keys: ``accuracy``, ``macro_f1``, ``kappa``, ``per_class_f1`` (dict
        mapping stage name to F1), ``confusion_matrix`` (list of lists),
        ``n_valid``, ``n_invalid``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    valid = np.isin(y_pred, _LABELS) & np.isin(y_true, _LABELS)
    n_valid = int(valid.sum())
    n_invalid = int((~valid).sum())

    if n_valid == 0:
        nan5 = {name: float("nan") for name in _LABEL_NAMES}
        return {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "kappa": float("nan"),
            "per_class_f1": nan5,
            "confusion_matrix": np.full((5, 5), np.nan).tolist(),
            "n_valid": 0,
            "n_invalid": n_invalid,
        }

    yt = y_true[valid]
    yp = y_pred[valid]

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(
        f1_score(yt, yp, average="macro", labels=_LABELS, zero_division=0)
    )
    kappa = float(cohen_kappa_score(yt, yp))
    per_class = f1_score(
        yt, yp, average=None, labels=_LABELS, zero_division=0
    )
    conf_mat = confusion_matrix(yt, yp, labels=_LABELS)

    per_class_dict = {
        name: float(per_class[i]) for i, name in enumerate(_LABEL_NAMES)
    }

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_f1": per_class_dict,
        "confusion_matrix": conf_mat.tolist(),
        "n_valid": n_valid,
        "n_invalid": n_invalid,
    }


# ---------------------------------------------------------------------------
# Overall (aggregate) metrics
# ---------------------------------------------------------------------------

def compute_overall_metrics(results_df: pd.DataFrame) -> Dict[str, Any]:
    """Aggregate classification metrics across all subjects.

    The DataFrame is expected to contain at least the columns ``label`` (int,
    ground-truth index) and ``pred`` (int, predicted index).  Rows where either
    value falls outside ``{0, 1, 2, 3, 4}`` are excluded.

    Optionally, if a ``rules_iou`` column is present its mean (ignoring NaN) is
    included in the result under the key ``mean_rules_iou``.

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results table (one row per epoch).

    Returns
    -------
    dict
        Same structure as :func:`compute_subject_metrics` plus
        ``mean_rules_iou`` when available.
    """
    df = results_df.copy()
    valid_mask = df["pred"].isin(_LABELS) & df["label"].isin(_LABELS)
    df_valid = df[valid_mask]

    if len(df_valid) == 0:
        nan5 = {name: float("nan") for name in _LABEL_NAMES}
        result: Dict[str, Any] = {
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
            "kappa": float("nan"),
            "per_class_f1": nan5,
            "confusion_matrix": np.full((5, 5), np.nan).tolist(),
            "n_valid": 0,
            "n_invalid": len(df),
        }
        if "rules_iou" in df.columns:
            result["mean_rules_iou"] = float("nan")
        return result

    yt = df_valid["label"].values
    yp = df_valid["pred"].values

    acc = float(accuracy_score(yt, yp))
    macro_f1 = float(
        f1_score(yt, yp, average="macro", labels=_LABELS, zero_division=0)
    )
    kappa = float(cohen_kappa_score(yt, yp))
    per_class = f1_score(
        yt, yp, average=None, labels=_LABELS, zero_division=0
    )
    conf_mat = confusion_matrix(yt, yp, labels=_LABELS)

    per_class_dict = {
        name: float(per_class[i]) for i, name in enumerate(_LABEL_NAMES)
    }

    result = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "kappa": kappa,
        "per_class_f1": per_class_dict,
        "confusion_matrix": conf_mat.tolist(),
        "n_valid": int(valid_mask.sum()),
        "n_invalid": int((~valid_mask).sum()),
    }

    if "rules_iou" in df_valid.columns:
        iou_vals = df_valid["rules_iou"].values
        result["mean_rules_iou"] = (
            float(np.nanmean(iou_vals)) if len(iou_vals) > 0 else float("nan")
        )

    return result


# ---------------------------------------------------------------------------
# Rule-set IoU
# ---------------------------------------------------------------------------

def compute_rules_iou(
    pred_rules: Optional[Union[str, List[str]]],
    gt_rules: Optional[List[str]],
) -> float:
    """Compute Intersection-over-Union between predicted and ground-truth rule sets.

    Each argument may be ``None`` (treated as empty), a comma-separated string,
    or a list of rule-name strings.

    Edge cases:

    * Both sets empty -> 1.0 (perfect match).
    * Exactly one set empty -> 0.0.

    Parameters
    ----------
    pred_rules : str, list of str, or None
        Predicted applicable rules.
    gt_rules : list of str or None
        Ground-truth applicable rules.

    Returns
    -------
    float
        IoU value in [0.0, 1.0].
    """
    # Normalise predictions to a set.
    if pred_rules is None:
        pred_set: set = set()
    elif isinstance(pred_rules, str):
        pred_set = {r.strip() for r in pred_rules.split(",") if r.strip()}
    elif isinstance(pred_rules, list):
        pred_set = {str(r).strip() for r in pred_rules if r}
    else:
        pred_set = set()

    # Normalise ground truth to a set.
    if gt_rules is None:
        gt_set: set = set()
    elif isinstance(gt_rules, list):
        gt_set = {str(r).strip() for r in gt_rules if r}
    else:
        gt_set = set()

    # Both empty is considered a perfect match.
    if len(pred_set) == 0 and len(gt_set) == 0:
        return 1.0

    intersection = len(pred_set & gt_set)
    union = len(pred_set | gt_set)

    if union == 0:
        return 0.0

    return intersection / union


# ---------------------------------------------------------------------------
# JSON serialisation
# ---------------------------------------------------------------------------

def save_metrics_json(
    results_df: pd.DataFrame,
    output_dir: str,
) -> None:
    """Save per-subject and overall metrics to JSON files, grouped by center.

    The ``sub_id`` column is expected to follow the pattern
    ``"<center>/<subject_id>"`` (e.g. ``"MASS-SS1/01-01-0001"``).  One JSON
    file is written per unique center, with the structure::

        {
          "num_subjects": 53,
          "num_epochs": 50700,
          "subjects": [ { "subject_id": ..., "metrics": {...} }, ... ],
          "overall": { ... }
        }

    Parameters
    ----------
    results_df : pd.DataFrame
        Full results table with at least ``sub_id``, ``label``, and ``pred``.
    output_dir : str
        Directory where the JSON files will be written.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = results_df.copy()
    df["center"] = df["sub_id"].apply(lambda x: x.split("/")[0])
    df["subject_id"] = df["sub_id"].apply(lambda x: x.split("/")[-1])

    for center, center_df in df.groupby("center"):
        center_result: Dict[str, Any] = {
            "num_subjects": int(center_df["sub_id"].nunique()),
            "num_epochs": 0,
            "subjects": [],
            "overall": {},
        }

        center_y_true: list = []
        center_y_pred: list = []

        for sub_id, sub_df in center_df.groupby("sub_id"):
            subject_id = str(sub_id).split("/")[-1]
            metrics = compute_subject_metrics(
                sub_df["label"].values, sub_df["pred"].values
            )

            if metrics["n_valid"] == 0:
                continue

            subject_record = {
                "subject_id": subject_id,
                "subject_path": str(sub_id),
                "num_epochs": metrics["n_valid"],
                "metrics": {
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "kappa": metrics["kappa"],
                    "per_class_f1": metrics["per_class_f1"],
                    "confusion_matrix": metrics["confusion_matrix"],
                },
            }
            center_result["subjects"].append(subject_record)
            center_result["num_epochs"] += metrics["n_valid"]

            # Collect for center-wide aggregation.
            valid = np.isin(sub_df["pred"].values, _LABELS) & np.isin(
                sub_df["label"].values, _LABELS
            )
            center_y_true.extend(sub_df["label"].values[valid].tolist())
            center_y_pred.extend(sub_df["pred"].values[valid].tolist())

        # Compute center-level overall metrics.
        if len(center_y_true) > 0:
            yt = np.array(center_y_true)
            yp = np.array(center_y_pred)

            acc = float(accuracy_score(yt, yp))
            macro_f1 = float(
                f1_score(yt, yp, average="macro", labels=_LABELS, zero_division=0)
            )
            kappa = float(cohen_kappa_score(yt, yp))
            per_class = f1_score(
                yt, yp, average=None, labels=_LABELS, zero_division=0
            )
            conf_mat = confusion_matrix(yt, yp, labels=_LABELS)

            center_result["overall"] = {
                "accuracy": acc,
                "macro_f1": macro_f1,
                "kappa": kappa,
                "per_class_f1": {
                    name: float(per_class[i])
                    for i, name in enumerate(_LABEL_NAMES)
                },
                "confusion_matrix": conf_mat.tolist(),
            }

        output_path = os.path.join(output_dir, f"{center}_metrics.json")
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(center_result, fh, ensure_ascii=False, indent=2)
