"""
Phase 2 SFT training data preparation script.

Reads annotations from MASS-EX CSV files (fine and coarse tracks) and generates
training samples with a 3-epoch sliding window for supervised fine-tuning.

Fine track (5 subjects):
  - Reads from MASS-EX/annotations/fine/{subject}.csv
  - CSV columns: custom_id, Subject, N, Stage, reasoning_text, applicable_rules
  - Output includes reasoning_text in the assistant response

Coarse track (remaining subjects):
  - Reads from MASS-EX/annotations/coarse/{subject}.csv
  - CSV columns: custom_id, Subject, N, Stage, applicable_rules
  - Output does NOT include reasoning_text

For each target epoch N (center), the preceding epoch N-1 and subsequent epoch N+1
are included as context images. First and last epochs of each recording are excluded
since they cannot serve as window centers.

Output: a single train.jsonl mixing both fine and coarse samples, plus subjects.json.

Usage:
    python scripts/prepare_sft_data.py
    python scripts/prepare_sft_data.py --data_dir data --massex_dir MASS-EX
"""

import os
import csv
import json
import random
import argparse
from collections import defaultdict


def load_split_config(split_json_path: str) -> dict:
    """Load the data split configuration file."""
    with open(split_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_system_prompt(prompt_path: str) -> str:
    """Load a system prompt from a markdown file."""
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_annotations_from_csv(csv_path: str, track: str) -> list:
    """
    Load annotation rows from a MASS-EX CSV file.

    Args:
        csv_path: Path to the CSV file.
        track: Either "fine" or "coarse". Fine CSVs have a reasoning_text column;
               coarse CSVs do not.

    Returns:
        A list of dicts, one per row, with keys:
          custom_id, Subject, N (int), Stage, applicable_rules (raw string),
          and reasoning_text (str or None).
    """
    rows = []
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entry = {
                "custom_id": row["custom_id"],
                "Subject": row["Subject"],
                "N": int(row["N"]),
                "Stage": row["Stage"],
                "applicable_rules": row.get("applicable_rules", ""),
            }
            if track == "fine":
                entry["reasoning_text"] = row.get("reasoning_text", "")
            else:
                entry["reasoning_text"] = None
            rows.append(entry)
    return rows


def parse_applicable_rules(raw: str) -> list:
    """
    Parse a comma-separated applicable_rules string into a JSON-compatible list.

    Example: "N2.1, N2.2" -> ["N2.1", "N2.2"]
    Returns an empty list if the input is empty or whitespace-only.
    """
    if not raw or not raw.strip():
        return []
    return [r.strip() for r in raw.split(",") if r.strip()]


def build_image_path(subject_id: str, epoch_idx: int, stage: str) -> str:
    """
    Build the relative image path for a given epoch.

    The path is relative to the data/ directory.
    Example: "MASS/SS3/images/01-03-0001/11_N2.png"
    """
    return f"MASS/SS3/images/{subject_id}/{epoch_idx}_{stage}.png"


def build_fine_sample(
    subject_id: str,
    preceding: dict,
    current: dict,
    subsequent: dict,
    system_prompt: str,
) -> dict:
    """
    Build a single fine-track training sample with reasoning_text.

    The assistant response contains reasoning_text, applicable_rules, and sleep_stage.
    """
    current_stage = current["Stage"]
    epoch_idx = current["N"]
    custom_id = f"{subject_id}#{epoch_idx}_{current_stage}"

    # Parse applicable_rules into a list
    rules_list = parse_applicable_rules(current["applicable_rules"])

    # Clean reasoning_text: escape double quotes and replace newlines with spaces
    reasoning_text = current["reasoning_text"] or ""
    reasoning_text = reasoning_text.replace('"', '\\"').replace("\n", " ")

    # Build the assistant JSON content string
    rules_json = json.dumps(rules_list, ensure_ascii=False)
    assistant_content = (
        f'```json\n{{\n'
        f'  "reasoning_text": "{reasoning_text}",\n'
        f'  "applicable_rules": {rules_json},\n'
        f'  "sleep_stage": "{current_stage}"\n'
        f'}}\n```'
    )

    # U+2011 (non-breaking hyphen) is used in "N-1" per spec
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Preceding epoch N\u20111: "},
                {
                    "type": "image",
                    "image": build_image_path(
                        subject_id, preceding["N"], preceding["Stage"]
                    ),
                },
                {"type": "text", "text": "**Target epoch N**: "},
                {
                    "type": "image",
                    "image": build_image_path(subject_id, epoch_idx, current_stage),
                },
                {"type": "text", "text": "Subsequent epoch N+1: "},
                {
                    "type": "image",
                    "image": build_image_path(
                        subject_id, subsequent["N"], subsequent["Stage"]
                    ),
                },
            ],
        },
        {"role": "assistant", "content": assistant_content},
    ]

    return {"id": custom_id, "messages": messages}


def build_coarse_sample(
    subject_id: str,
    preceding: dict,
    current: dict,
    subsequent: dict,
    system_prompt: str,
) -> dict:
    """
    Build a single coarse-track training sample without reasoning_text.

    The assistant response contains only applicable_rules and sleep_stage.
    """
    current_stage = current["Stage"]
    epoch_idx = current["N"]
    custom_id = f"{subject_id}#{epoch_idx}_{current_stage}"

    # Parse applicable_rules into a list
    rules_list = parse_applicable_rules(current["applicable_rules"])

    # Build the assistant JSON content string (no reasoning_text)
    rules_json = json.dumps(rules_list, ensure_ascii=False)
    assistant_content = (
        f'```json\n{{\n'
        f'  "applicable_rules": {rules_json},\n'
        f'  "sleep_stage": "{current_stage}"\n'
        f'}}\n```'
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Preceding epoch N\u20111: "},
                {
                    "type": "image",
                    "image": build_image_path(
                        subject_id, preceding["N"], preceding["Stage"]
                    ),
                },
                {"type": "text", "text": "**Target epoch N**: "},
                {
                    "type": "image",
                    "image": build_image_path(subject_id, epoch_idx, current_stage),
                },
                {"type": "text", "text": "Subsequent epoch N+1: "},
                {
                    "type": "image",
                    "image": build_image_path(
                        subject_id, subsequent["N"], subsequent["Stage"]
                    ),
                },
            ],
        },
        {"role": "assistant", "content": assistant_content},
    ]

    return {"id": custom_id, "messages": messages}


def process_subject(
    subject_id: str,
    track: str,
    massex_dir: str,
    system_prompt: str,
) -> list:
    """
    Process a single subject's CSV and produce training samples using a
    3-epoch sliding window.

    Args:
        subject_id: e.g. "01-03-0001"
        track: "fine" or "coarse"
        massex_dir: root directory for MASS-EX annotations
        system_prompt: the system prompt string for this track

    Returns:
        List of (sample_dict, stage_str) tuples.
    """
    csv_path = os.path.join(massex_dir, "annotations", track, f"{subject_id}.csv")
    if not os.path.isfile(csv_path):
        print(f"  Warning: CSV not found: {csv_path}")
        return []

    rows = load_annotations_from_csv(csv_path, track)
    if len(rows) < 3:
        print(f"  Warning: fewer than 3 rows for {subject_id}, skipping")
        return []

    # Sort by epoch index to guarantee temporal order
    rows.sort(key=lambda r: r["N"])

    samples = []
    # Sliding window: target is the center epoch (index 1 .. len-2)
    for pos in range(1, len(rows) - 1):
        preceding = rows[pos - 1]
        current = rows[pos]
        subsequent = rows[pos + 1]

        # Skip epochs where applicable_rules is empty or missing
        rules_raw = current.get("applicable_rules", "")
        if not rules_raw or not rules_raw.strip():
            continue

        # For fine track, also skip if reasoning_text is empty
        if track == "fine":
            reasoning = current.get("reasoning_text", "")
            if not reasoning or not reasoning.strip():
                continue
            sample = build_fine_sample(
                subject_id, preceding, current, subsequent, system_prompt
            )
        else:
            sample = build_coarse_sample(
                subject_id, preceding, current, subsequent, system_prompt
            )

        samples.append((sample, current["Stage"]))

    return samples


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 SFT training data preparation script"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory (default: data)",
    )
    parser.add_argument(
        "--massex_dir",
        type=str,
        default="MASS-EX",
        help="MASS-EX root directory (default: MASS-EX)",
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default="split.json",
        help="Path to split.json (default: data/split.json)",
    )
    parser.add_argument(
        "--fine_prompt",
        type=str,
        default="sleepvlm/prompts/phase2_sft_fine.md",
        help="Path to fine-track system prompt (default: sleepvlm/prompts/phase2_sft_fine.md)",
    )
    parser.add_argument(
        "--coarse_prompt",
        type=str,
        default="sleepvlm/prompts/phase2_sft_coarse.md",
        help="Path to coarse-track system prompt (default: sleepvlm/prompts/phase2_sft_coarse.md)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/phase2_sft",
        help="Output directory (default: data/phase2_sft)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    # ---------- Load configuration ----------
    print("Loading split configuration...")
    split_config = load_split_config(args.split_json)
    fine_subjects = split_config["fine_train_subjects"]
    coarse_subjects = split_config["coarse_train_subjects"]
    print(f"  Fine-track subjects ({len(fine_subjects)}): {fine_subjects}")
    print(f"  Coarse-track subjects ({len(coarse_subjects)}): {coarse_subjects}")

    # ---------- Load system prompts ----------
    print("Loading system prompts...")
    fine_system_prompt = load_system_prompt(args.fine_prompt)
    coarse_system_prompt = load_system_prompt(args.coarse_prompt)
    print(f"  Fine prompt length: {len(fine_system_prompt)} chars")
    print(f"  Coarse prompt length: {len(coarse_system_prompt)} chars")

    # ---------- Process fine-track subjects ----------
    print(f"\nProcessing fine-track subjects ({len(fine_subjects)})...")
    all_samples = []  # list of (sample_dict, stage_str)

    for subject_id in fine_subjects:
        subject_samples = process_subject(
            subject_id, "fine", args.massex_dir, fine_system_prompt
        )
        all_samples.extend(subject_samples)
        print(f"  {subject_id}: {len(subject_samples)} samples")

    fine_count = len(all_samples)
    print(f"  Fine-track total: {fine_count} samples")

    # ---------- Process coarse-track subjects ----------
    print(f"\nProcessing coarse-track subjects ({len(coarse_subjects)})...")

    for subject_id in coarse_subjects:
        subject_samples = process_subject(
            subject_id, "coarse", args.massex_dir, coarse_system_prompt
        )
        all_samples.extend(subject_samples)
        print(f"  {subject_id}: {len(subject_samples)} samples")

    coarse_count = len(all_samples) - fine_count
    print(f"  Coarse-track total: {coarse_count} samples")

    # ---------- Stage distribution ----------
    stage_counts = defaultdict(int)
    for _, stage in all_samples:
        stage_counts[stage] += 1
    print(f"\nStage distribution: {dict(sorted(stage_counts.items()))}")
    print(f"Total samples: {len(all_samples)}")

    # ---------- Shuffle ----------
    random.seed(args.seed)
    random.shuffle(all_samples)

    # ---------- Write output ----------
    os.makedirs(args.output_dir, exist_ok=True)

    train_jsonl_path = os.path.join(args.output_dir, "train.jsonl")
    with open(train_jsonl_path, "w", encoding="utf-8") as f:
        for sample, _ in all_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"\nTraining data saved: {train_jsonl_path}")

    # Save subjects split info
    subjects_json_path = os.path.join(args.output_dir, "subjects.json")
    subjects_info = {
        "fine_train_subjects": fine_subjects,
        "coarse_train_subjects": coarse_subjects,
        "fine_sample_count": fine_count,
        "coarse_sample_count": coarse_count,
        "total_sample_count": len(all_samples),
        "stage_distribution": dict(sorted(stage_counts.items())),
    }
    with open(subjects_json_path, "w", encoding="utf-8") as f:
        json.dump(subjects_info, f, ensure_ascii=False, indent=2)
    print(f"Subjects info saved: {subjects_json_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
