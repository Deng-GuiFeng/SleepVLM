"""
Inference pipeline for the SleepVLM model.

Handles image encoding, vLLM API calls, sample collection from image
directories, and parallelised batch inference across multiple API
endpoints.

The expected directory layout for each subject is::

    <img_dir>/SS*/images/<subject_id>/
        0_W.png
        1_N1.png
        2_N2.png
        ...

Each filename follows the pattern ``<epoch_index>_<stage>.<ext>``.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
from typing import Any, Dict, List, Optional

import numpy as np
import requests
from tqdm import tqdm

from sleepvlm.evaluation.metrics import compute_rules_iou
from sleepvlm.evaluation.parse_output import STAGE_MAP, parse_model_output

# Re-export the stage map for convenience.
__all__ = [
    "to_base64_data_url",
    "call_vllm_api",
    "process_sample",
    "collect_samples",
    "run_inference",
]

# ---------------------------------------------------------------------------
# Filename helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp"}


def _is_image_file(path: str) -> bool:
    """Return True if *path* has a recognised image extension."""
    ext = os.path.splitext(path)[1].lower()
    return ext in _IMAGE_EXTENSIONS


def _parse_filename(fname: str):
    """Extract (epoch_index, stage) from a filename like ``42_N2.png``.

    Returns ``None`` when the filename does not match the expected pattern.
    """
    match = re.match(r"^(\d+)_([^.]+)\.[A-Za-z0-9]+$", fname)
    if not match:
        return None
    try:
        idx = int(match.group(1))
    except ValueError:
        return None
    stage = match.group(2)
    return idx, stage


# ---------------------------------------------------------------------------
# Image encoding
# ---------------------------------------------------------------------------

def to_base64_data_url(path: str) -> str:
    """Convert an image file to a base64-encoded data URL.

    The MIME type is inferred from the file extension.  Falls back to
    ``image/png`` when the type cannot be determined.

    Parameters
    ----------
    path : str
        Path to the image file on disk.

    Returns
    -------
    str
        A ``data:<mime>;base64,<payload>`` URL suitable for embedding in
        OpenAI-compatible chat messages.
    """
    mime, _ = mimetypes.guess_type(path)
    mime = mime or "image/png"
    with open(path, "rb") as fh:
        b64 = base64.b64encode(fh.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


# ---------------------------------------------------------------------------
# vLLM API interaction
# ---------------------------------------------------------------------------

def call_vllm_api(
    messages: List[Dict[str, Any]],
    base_url: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int] = None,
) -> str:
    """Send a chat-completion request to a local vLLM server.

    Parameters
    ----------
    messages : list of dict
        OpenAI-compatible message list (system + user).
    base_url : str
        Base URL of the vLLM server, e.g. ``"http://127.0.0.1:6002/v1"``.
    model_name : str
        Model identifier (typically the checkpoint directory path).
    temperature : float
        Sampling temperature.
    top_p : float
        Nucleus-sampling cumulative probability threshold.
    max_tokens : int
        Maximum number of tokens to generate.
    seed : int, optional
        Random seed for reproducibility.  When ``None`` the server's default
        randomness is used.

    Returns
    -------
    str
        The text content of the first completion choice.
    """
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer EMPTY",
    }

    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed

    resp = requests.post(
        f"{base_url}/chat/completions",
        headers=headers,
        data=json.dumps(payload),
    )
    resp.raise_for_status()
    data = resp.json()

    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        content = str(data)
    return content


# ---------------------------------------------------------------------------
# Single-sample processing
# ---------------------------------------------------------------------------

def process_sample(
    sample: Dict[str, Any],
    base_url: str,
    system_prompt: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    all_api_urls: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Process a single sample: build messages, call API, parse, compute IoU.

    If the primary ``base_url`` returns a server error (5xx), the request is
    automatically retried on other URLs from ``all_api_urls``.

    The message payload contains three images arranged as a sliding window:

    * **Preceding epoch N-1** (context)
    * **Target epoch N** (the epoch to classify)
    * **Subsequent epoch N+1** (context)

    Parameters
    ----------
    sample : dict
        Must contain ``custom_id``, ``sub_id``, ``image_paths`` (with keys
        ``preceding``, ``current``, ``subsequent``), ``stage``, ``label``,
        and optionally ``gt_applicable_rules``.
    base_url : str
        vLLM server URL.
    system_prompt : str
        System-level prompt text.
    model_name : str
        Model identifier for the API.
    temperature, top_p, max_tokens, seed :
        Generation hyper-parameters forwarded to :func:`call_vllm_api`.

    Returns
    -------
    dict
        Result dictionary with keys: ``custom_id``, ``sub_id``,
        ``image_paths``, ``stage``, ``label``, ``output``, ``pred``,
        ``pred_stage``, ``reasoning_text``, ``applicable_rules``,
        ``gt_applicable_rules``, ``rules_iou``, ``parse_error``.
    """
    # NOTE: The non-breaking hyphen U+2011 is used in "N-1" to match the
    # training data formatting exactly.
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Preceding epoch N\u20111: "},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_base64_data_url(
                            sample["image_paths"]["preceding"]
                        ),
                    },
                },
                {"type": "text", "text": "**Target epoch N**: "},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_base64_data_url(
                            sample["image_paths"]["current"]
                        ),
                    },
                },
                {"type": "text", "text": "Subsequent epoch N+1: "},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": to_base64_data_url(
                            sample["image_paths"]["subsequent"]
                        ),
                    },
                },
            ],
        },
    ]

    gt_applicable_rules = sample.get("gt_applicable_rules")

    # Build the list of URLs to try: primary first, then fallbacks.
    urls_to_try = [base_url]
    if all_api_urls:
        for u in all_api_urls:
            if u != base_url and u not in urls_to_try:
                urls_to_try.append(u)

    try:
        out_text = None
        last_err = None
        for try_url in urls_to_try:
            try:
                out_text = call_vllm_api(
                    messages, try_url, model_name, temperature, top_p,
                    max_tokens, seed,
                )
                break  # Success
            except requests.exceptions.HTTPError as e:
                if e.response is not None and e.response.status_code >= 500:
                    last_err = e
                    continue  # Retry on next server
                raise  # Client error (4xx), don't retry
        if out_text is None:
            raise last_err or RuntimeError("All API endpoints failed")
        sleep_stage, reasoning_text, applicable_rules, parse_error = (
            parse_model_output(out_text)
        )
        pred = STAGE_MAP.get(sleep_stage, -1) if sleep_stage is not None else -1
        rules_iou = compute_rules_iou(applicable_rules, gt_applicable_rules)

        return {
            "custom_id": sample["custom_id"],
            "sub_id": sample["sub_id"],
            "image_paths": sample["image_paths"],
            "stage": sample["stage"],
            "label": sample["label"],
            "output": out_text,
            "pred": pred,
            "pred_stage": sleep_stage,
            "reasoning_text": reasoning_text,
            "applicable_rules": applicable_rules,
            "gt_applicable_rules": gt_applicable_rules,
            "rules_iou": rules_iou,
            "parse_error": parse_error,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "custom_id": sample["custom_id"],
            "sub_id": sample["sub_id"],
            "image_paths": sample["image_paths"],
            "stage": sample["stage"],
            "label": sample["label"],
            "output": f"Request failed: {exc}",
            "pred": -1,
            "pred_stage": None,
            "reasoning_text": None,
            "applicable_rules": None,
            "gt_applicable_rules": gt_applicable_rules,
            "rules_iou": float("nan"),
            "parse_error": f"Request exception: {exc}",
        }


def _process_sample_star(args):
    """Unpack a positional-argument tuple and forward to :func:`process_sample`.

    This wrapper is required because :mod:`multiprocess` ``Pool.imap*``
    only passes a single argument to the worker function.
    """
    return process_sample(*args)


# ---------------------------------------------------------------------------
# Sample collection
# ---------------------------------------------------------------------------

def collect_samples(
    subjects: List[str],
    img_dir: str,
    annotation_data: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Collect all test samples from subject image directories.

    For each subject the function reads the image directory, sorts epochs by
    index, and builds a sliding window of three consecutive epochs (preceding,
    current, subsequent).  The first and last epochs of each subject are used
    only as context and are never the classification target.

    Parameters
    ----------
    subjects : list of str
        Subject identifiers in the form ``"<center>/<subject_id>"``
        (e.g. ``"MASS-SS1/01-01-0001"``).
    img_dir : str
        Root directory containing subject image sub-directories.
    annotation_data : dict, optional
        Mapping from annotation key to a dict with ``"applicable_rules"``
        (list of str) and ``"stage"`` (str).  When provided, the ground-truth
        rules are attached to each sample.

    Returns
    -------
    list of dict
        Each element is a sample dict ready for :func:`process_sample`.
    """
    if annotation_data is None:
        annotation_data = {}

    samples: List[Dict[str, Any]] = []

    for sub_id in tqdm(subjects, desc="Collecting samples"):
        sub_img_dir = os.path.join(img_dir, sub_id)
        if not os.path.isdir(sub_img_dir):
            continue

        # Discover image files and parse their filenames.
        epochs = []
        for fname in os.listdir(sub_img_dir):
            if not _is_image_file(fname):
                continue
            parsed = _parse_filename(fname)
            if parsed is None:
                continue
            idx, stage = parsed
            epochs.append(
                {
                    "idx": idx,
                    "stage": stage,
                    "path": os.path.join(sub_img_dir, fname),
                    "sub_id": sub_id,
                }
            )

        if len(epochs) < 3:
            continue

        epochs.sort(key=lambda e: e["idx"])

        # Build 3-epoch sliding windows; only interior epochs are targets.
        n = len(epochs)
        for pos in range(1, n - 1):
            preceding = epochs[pos - 1]
            current = epochs[pos]
            subsequent = epochs[pos + 1]

            current_stage = current["stage"]
            custom_id = f"{sub_id}#{current['idx']}_{current_stage}"

            # Look up ground-truth annotation (key uses bare subject id).
            subject_id_only = sub_id.split("/")[-1]
            annotation_key = (
                f"{subject_id_only}#{current['idx']}_{current_stage}"
            )
            annotation_item = annotation_data.get(annotation_key)
            gt_applicable_rules = (
                annotation_item["applicable_rules"]
                if annotation_item
                else None
            )

            samples.append(
                {
                    "custom_id": custom_id,
                    "sub_id": sub_id,
                    "image_paths": {
                        "preceding": preceding["path"],
                        "current": current["path"],
                        "subsequent": subsequent["path"],
                    },
                    "stage": current_stage,
                    "label": STAGE_MAP.get(current_stage, -1),
                    "gt_applicable_rules": gt_applicable_rules,
                }
            )

    return samples


# ---------------------------------------------------------------------------
# Parallel inference
# ---------------------------------------------------------------------------

def run_inference(
    samples: List[Dict[str, Any]],
    api_urls: List[str],
    system_prompt: str,
    model_name: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: Optional[int],
    num_threads: int,
) -> List[Dict[str, Any]]:
    """Run parallel inference over all samples using a multiprocess pool.

    Samples are distributed across *api_urls* in round-robin fashion so that
    the load is balanced across multiple vLLM server instances.

    Parameters
    ----------
    samples : list of dict
        Sample dicts as produced by :func:`collect_samples`.
    api_urls : list of str
        vLLM server base URLs (e.g. ``["http://127.0.0.1:6002/v1", ...]``).
    system_prompt : str
        System prompt text.
    model_name : str
        Model identifier.
    temperature, top_p, max_tokens, seed :
        Generation hyper-parameters.
    num_threads : int
        Number of parallel worker processes.

    Returns
    -------
    list of dict
        Result dicts sorted in the same order as the input *samples*.
    """
    # Import here to keep the top-level import lightweight; multiprocess can
    # be slow to load on some systems.
    import multiprocess as mp  # noqa: WPS433

    # Build task tuples with round-robin URL assignment.
    tasks = [
        (
            sample,
            api_urls[idx % len(api_urls)],
            system_prompt,
            model_name,
            temperature,
            top_p,
            max_tokens,
            seed,
        )
        for idx, sample in enumerate(samples)
    ]

    # Build a lookup from custom_id to original position for sorting.
    order_map = {s["custom_id"]: i for i, s in enumerate(samples)}

    results: List[Dict[str, Any]] = []
    with mp.Pool(processes=num_threads) as pool:
        for result in tqdm(
            pool.imap_unordered(_process_sample_star, tasks),
            total=len(tasks),
            desc="Inference",
        ):
            results.append(result)

    # Restore the original sample order for deterministic output.
    results.sort(key=lambda r: order_map.get(r["custom_id"], float("inf")))

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_annotations_from_massex(annotation_dir: str) -> Dict[str, Any]:
    """Load ground-truth annotations from MASS-EX CSV files.

    Returns a dict keyed by ``"<subject_id>#<epoch>_<stage>"`` with value
    ``{"applicable_rules": [...]}``.
    """
    import csv

    data: Dict[str, Any] = {}
    for subdir in ("fine", "coarse"):
        csv_dir = os.path.join(annotation_dir, subdir)
        if not os.path.isdir(csv_dir):
            continue
        for fname in os.listdir(csv_dir):
            if not fname.endswith(".csv"):
                continue
            fpath = os.path.join(csv_dir, fname)
            with open(fpath, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    cid = row.get("custom_id", "")
                    rules_str = row.get("applicable_rules", "")
                    if not cid:
                        continue
                    rules = (
                        [r.strip() for r in rules_str.split(",") if r.strip()]
                        if rules_str
                        else []
                    )
                    data[cid] = {"applicable_rules": rules}
    return data


def main() -> None:
    """CLI entry point for batch inference."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run SleepVLM inference on a test set via vLLM."
    )
    parser.add_argument("--model_name", type=str, required=True,
                        help="Model name/path (must match vLLM --served-model-name)")
    parser.add_argument("--system_prompt", type=str, required=True,
                        help="Path to the system prompt file")
    parser.add_argument("--split_json", type=str, default="split.json",
                        help="Path to split.json")
    parser.add_argument("--split", type=str, default="test",
                        choices=["test", "val"],
                        help="Which split to evaluate (default: test)")
    parser.add_argument("--img_dir", type=str, default="data/MASS",
                        help="Root data directory containing SS*/images/")
    parser.add_argument("--annotation_dir", type=str, default="MASS-EX/annotations",
                        help="Path to MASS-EX annotations directory")
    parser.add_argument("--output_dir", type=str, default="outputs/eval",
                        help="Output directory for results")
    parser.add_argument("--base_port", type=int, default=6002)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_threads", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=1e-6)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--max_tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    # Load split configuration.
    with open(args.split_json, "r", encoding="utf-8") as f:
        split_config = json.load(f)

    if args.split == "test":
        raw_subjects = split_config["test_subjects"]
    else:
        raw_subjects = split_config["val_subjects"]

    # Map subject IDs to their image directory paths.
    # Test subjects (01-01-*) are in SS1/images/, val subjects (01-03-*)
    # are in SS3/images/, etc.
    SUBSET_MAP = {
        "01-01": "SS1", "01-02": "SS2", "01-03": "SS3",
        "01-04": "SS4", "01-05": "SS5",
    }

    # Build subject list with full image directory paths.
    # collect_samples joins img_dir + sub_id to find images.
    subjects = []
    for sid in raw_subjects:
        prefix = sid[:5]  # e.g. "01-01"
        subset = SUBSET_MAP.get(prefix)
        if subset:
            # Path under img_dir: SS1/images/01-01-0001
            subjects.append(os.path.join(subset, "images", sid))
        else:
            subjects.append(sid)

    print(f"Split: {args.split} ({len(subjects)} subjects)")

    # Load system prompt.
    with open(args.system_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Load ground-truth annotations (for IoU computation).
    annotation_data = {}
    if os.path.isdir(args.annotation_dir):
        annotation_data = _load_annotations_from_massex(args.annotation_dir)
        print(f"Loaded {len(annotation_data)} ground-truth annotations")

    # Collect samples.
    samples = collect_samples(subjects, args.img_dir, annotation_data)

    # Fix sub_id: strip the internal "SS1/images/" prefix so that
    # sub_id is a clean identifier like "01-01-0001".
    for s in samples:
        parts = s["sub_id"].split("/")
        s["sub_id"] = parts[-1]  # "01-01-0001"
        # Also fix custom_id to use clean sub_id
        s["custom_id"] = f"{s['sub_id']}#{s['custom_id'].split('#', 1)[1]}"
    print(f"Collected {len(samples)} samples")

    if not samples:
        print("No samples found. Check --img_dir and subject image directories.")
        return

    # Build API URLs.
    api_urls = [
        f"http://127.0.0.1:{args.base_port + i}/v1"
        for i in range(args.num_gpus)
    ]

    # Run inference.
    results = run_inference(
        samples=samples,
        api_urls=api_urls,
        system_prompt=system_prompt,
        model_name=args.model_name,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        seed=args.seed,
        num_threads=args.num_threads,
    )

    # Save results.
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl = os.path.join(args.output_dir, "results.jsonl")
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Results saved to {output_jsonl}")

    # Quick summary.
    valid = [r for r in results if r.get("pred", -1) in STAGE_MAP.values()]
    if valid:
        correct = sum(1 for r in valid if r["pred"] == r["label"])
        print(f"Valid predictions: {len(valid)}/{len(results)}")
        print(f"Accuracy: {correct / len(valid):.4f}")
    else:
        print("No valid predictions to summarize.")


if __name__ == "__main__":
    main()
