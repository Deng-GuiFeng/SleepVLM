"""Phase 1 WPT (Waveform-Perceptual Pre-training) data preparation script.

Assembles rendered waveform images with corresponding band power JSON files
into JSONL training data for Phase 1 Waveform-Perceptual Pre-training.

Input layout:
    data/MASS/SS{2,4,5}/images/{subject_id}/{epoch}.png
    data/MASS/SS{2,4,5}/band_power/{subject_id}.json

Output:
    data/phase1_wpt/train.jsonl
    data/phase1_wpt/val.jsonl
    data/phase1_wpt/subjects.json

Each JSONL record is a multi-turn conversation with system prompt, a user
message containing text + image, and an assistant response with compact band
power JSON wrapped in a code fence.
"""

import argparse
import json
import os
import random
import re
from typing import Optional

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Channel name mapping: band_power JSON key -> display label in output
# Note: hyphens below are U+2011 NON-BREAKING HYPHEN, not regular ASCII '-'.
# ---------------------------------------------------------------------------
CHANNEL_MAP = {
    "F4": "F4\u2011M1",
    "C4": "C4\u2011M1",
    "O2": "O2\u2011M1",
    "LOC": "LOC",
    "ROC": "ROC",
    "Chin": "Chin",
}

# Ordered list of output channel labels (controls JSON key ordering).
CHANNEL_ORDER = ["F4\u2011M1", "C4\u2011M1", "O2\u2011M1", "LOC", "ROC", "Chin"]

# Band names used for EEG/EOG channels.
BANDS = ["delta", "theta", "alpha", "beta", "mav"]

# Channels that only carry MAV (EMG).
EMG_CHANNELS = {"Chin"}

# MASS subsets used for Phase 1 pre-training.
SUBSETS = ["SS2", "SS4", "SS5"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_image_file(filename: str) -> bool:
    """Return True if the filename has a recognised image extension."""
    return os.path.splitext(filename)[1].lower() in {
        ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp",
    }


def parse_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract epoch index from an unlabelled image filename like '42.png'.

    Returns the integer epoch index, or None if the filename does not match
    the expected pattern.
    """
    m = re.match(r"^(\d+)\.[A-Za-z0-9]+$", filename)
    if m is None:
        return None
    return int(m.group(1))


def load_json(path: str) -> dict:
    """Load and return a JSON file as a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Band power conversion
# ---------------------------------------------------------------------------

def convert_epoch_to_compact(epoch_data: dict) -> dict:
    """Convert a single epoch's band power dict into compact array format.

    Input (per channel, keyed by 1-indexed second as string):
        {"F4": {"1": {"delta": 25.42, "theta": ..., "mav": ...}, ...}, ...}

    Output:
        {"F4-M1": [[25.4, 14.8, 14.2, 20.8, 5.2], ...],  # 30 elements
         "Chin":  [[2.4], [1.9], ...]}                      # 30 elements
    """
    compact = {}
    for src_key, dst_label in CHANNEL_MAP.items():
        if src_key not in epoch_data:
            continue
        ch_data = epoch_data[src_key]
        seconds = []
        for sec in range(1, 31):
            sec_data = ch_data.get(str(sec))
            if sec_data is not None:
                if src_key in EMG_CHANNELS:
                    seconds.append([round(sec_data.get("mav", 0.0), 1)])
                else:
                    seconds.append(
                        [round(sec_data.get(b, 0.0), 1) for b in BANDS]
                    )
            else:
                # Fill missing seconds with zeros.
                if src_key in EMG_CHANNELS:
                    seconds.append([0.0])
                else:
                    seconds.append([0.0] * 5)
        compact[dst_label] = seconds
    return compact


def format_compact_json(data: dict) -> str:
    """Format the compact band power dict as a readable JSON string.

    Each channel is on its own line with values rounded to 1 decimal place.
    Channel ordering follows CHANNEL_ORDER.
    """
    lines = ["{"]
    ch_lines = []
    for ch in CHANNEL_ORDER:
        if ch not in data:
            continue
        arr_strs = []
        for arr in data[ch]:
            arr_str = "[" + ",".join(f"{v:.1f}" for v in arr) + "]"
            arr_strs.append(arr_str)
        ch_lines.append(f'  "{ch}": [{", ".join(arr_strs)}]')
    lines.append(",\n".join(ch_lines))
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Subject collection and splitting
# ---------------------------------------------------------------------------

def collect_subjects(data_dir: str, subset: str):
    """Yield (unique_id, subject_id, img_dir, bp_dir) for each valid subject
    within a MASS subset directory.

    A subject is valid when both an image subdirectory and a matching band
    power JSON file exist.
    """
    img_root = os.path.join(data_dir, "MASS", subset, "images")
    bp_root = os.path.join(data_dir, "MASS", subset, "band_power")

    if not os.path.isdir(img_root):
        print(f"Warning: image directory does not exist: {img_root}")
        return
    if not os.path.isdir(bp_root):
        print(f"Warning: band_power directory does not exist: {bp_root}")
        return

    for sub_id in sorted(os.listdir(img_root)):
        sub_img_dir = os.path.join(img_root, sub_id)
        if not os.path.isdir(sub_img_dir):
            continue
        bp_file = os.path.join(bp_root, f"{sub_id}.json")
        if not os.path.isfile(bp_file):
            print(f"Warning: band_power file missing: {bp_file}")
            continue
        unique_id = f"{subset}_{sub_id}"
        yield unique_id, sub_id, sub_img_dir, bp_file


def stratified_split(data_dir: str, split_ratio: float, seed: int):
    """Collect subjects from all MASS subsets and perform a stratified
    train/val split within each subset.

    Returns (train_subjects, val_subjects, dataset_stats) where each subject
    entry is (unique_id, subject_id, img_dir, bp_file).
    """
    rng = random.Random(seed)
    train_subjects = []
    val_subjects = []
    dataset_stats = {}

    for subset in SUBSETS:
        subjects = list(collect_subjects(data_dir, subset))
        dataset_stats[subset] = len(subjects)
        print(f"Subset {subset}: {len(subjects)} subjects")

        if not subjects:
            continue

        rng.shuffle(subjects)
        split_idx = int(len(subjects) * split_ratio)
        ds_train = subjects[:split_idx]
        ds_val = subjects[split_idx:]

        train_subjects.extend(ds_train)
        val_subjects.extend(ds_val)
        print(f"  -> train: {len(ds_train)}, val: {len(ds_val)}")

    return train_subjects, val_subjects, dataset_stats


# ---------------------------------------------------------------------------
# JSONL record construction
# ---------------------------------------------------------------------------

def build_record(
    unique_id: str,
    epoch_idx: int,
    image_rel_path: str,
    compact_data: dict,
    system_prompt: str,
) -> dict:
    """Build a single JSONL conversation record."""
    assistant_body = format_compact_json(compact_data)
    return {
        "id": f"{unique_id}#{epoch_idx}",
        "messages": [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this 30-second PSG waveform image and estimate the band power:",
                    },
                    {"type": "image", "image": image_rel_path},
                ],
            },
            {
                "role": "assistant",
                "content": f"```json\n{assistant_body}\n```",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Prepare Phase 1 WPT training data from MASS waveform images and band power labels."
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root data directory (default: data)",
    )
    parser.add_argument(
        "--prompt_file", type=str, default="sleepvlm/prompts/phase1_wpt.md",
        help="Path to the system prompt markdown file (default: sleepvlm/prompts/phase1_wpt.md)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/phase1_wpt",
        help="Output directory for JSONL files (default: data/phase1_wpt)",
    )
    parser.add_argument(
        "--split_ratio", type=float, default=0.8,
        help="Fraction of subjects for training (default: 0.8)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible splits (default: 42)",
    )
    args = parser.parse_args()

    # Load system prompt.
    with open(args.prompt_file, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    # Collect subjects and split within each subset.
    train_subjects, val_subjects, dataset_stats = stratified_split(
        args.data_dir, args.split_ratio, args.seed,
    )

    total = len(train_subjects) + len(val_subjects)
    print(f"\nTotal subjects: {total}")
    if total == 0:
        print("Error: no valid subjects found.")
        return
    print(f"Split: train={len(train_subjects)}, val={len(val_subjects)}")

    # Prepare output directory and file paths.
    os.makedirs(args.output_dir, exist_ok=True)
    train_path = os.path.join(args.output_dir, "train.jsonl")
    val_path = os.path.join(args.output_dir, "val.jsonl")
    subjects_path = os.path.join(args.output_dir, "subjects.json")

    # Save subject split metadata.
    subjects_info = {
        "total_subjects_num": total,
        "train_subjects_num": len(train_subjects),
        "val_subjects_num": len(val_subjects),
        "split_ratio": args.split_ratio,
        "seed": args.seed,
        "datasets": dataset_stats,
        "train_subjects": [s[0] for s in train_subjects],
        "val_subjects": [s[0] for s in val_subjects],
    }
    with open(subjects_path, "w", encoding="utf-8") as f:
        json.dump(subjects_info, f, ensure_ascii=False, indent=2)

    # Build lookup sets for routing records to train or val.
    train_ids = {s[0] for s in train_subjects}
    val_ids = {s[0] for s in val_subjects}

    all_subjects = train_subjects + val_subjects

    train_count = 0
    val_count = 0
    skip_count = 0

    with open(train_path, "w", encoding="utf-8") as f_train, \
         open(val_path, "w", encoding="utf-8") as f_val:

        for unique_id, sub_id, sub_img_dir, bp_file in tqdm(
            all_subjects, desc="Processing subjects"
        ):
            # Load band power data for the entire subject.
            bp_data = load_json(bp_file)

            # Determine the MASS subset and build the relative image prefix.
            # sub_img_dir is like: data/MASS/SS2/images/01-02-0001
            # We want the relative path from data_dir: MASS/SS2/images/01-02-0001
            subset = unique_id.split("_", 1)[0]  # e.g. "SS2"
            rel_prefix = os.path.join("MASS", subset, "images", sub_id)

            # Iterate over image files for this subject.
            img_files = [
                fname for fname in os.listdir(sub_img_dir)
                if is_image_file(fname)
            ]

            for img_file in img_files:
                epoch_idx = parse_epoch_from_filename(img_file)
                if epoch_idx is None:
                    continue

                # Look up band power data for this epoch.
                epoch_key = str(epoch_idx)
                if epoch_key not in bp_data:
                    skip_count += 1
                    continue

                # Convert to compact format.
                compact = convert_epoch_to_compact(bp_data[epoch_key])

                # Skip epochs with fewer than 4 channels of valid data.
                if len(compact) < 4:
                    skip_count += 1
                    continue

                # Image path relative to the data/ directory.
                image_rel = os.path.join(rel_prefix, img_file)

                record = build_record(
                    unique_id, epoch_idx, image_rel, compact, system_prompt,
                )
                line = json.dumps(record, ensure_ascii=False) + "\n"

                if unique_id in train_ids:
                    f_train.write(line)
                    train_count += 1
                elif unique_id in val_ids:
                    f_val.write(line)
                    val_count += 1

    print(f"\nDone.")
    print(f"Train samples: {train_count}")
    print(f"Val samples:   {val_count}")
    print(f"Skipped:       {skip_count}")
    print(f"\nOutput files:")
    print(f"  {subjects_path}")
    print(f"  {train_path}")
    print(f"  {val_path}")


if __name__ == "__main__":
    main()
