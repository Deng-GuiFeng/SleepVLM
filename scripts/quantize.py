#!/usr/bin/env python3
"""
W4A16 quantization for a fine-tuned Qwen2.5-VL model using Intel AutoRound.

Workflow:
  1. Load the full-precision (bf16/fp16) model.
  2. Build a calibration dataset from a training JSONL with stratified
     sampling by sleep stage, resolving image paths and applying the
     Qwen2.5-VL chat template.
  3. Run AutoRound quantization (W4A16) on all nn.Linear layers inside the
     transformer blocks, skipping the vision encoder and lm_head.
  4. Save the quantized model and patch config.json for vLLM compatibility.

Usage:

  python scripts/quantize.py \
      --model_path runs/sft/merged/checkpoint-9000 \
      --output_dir runs/sft/merged/checkpoint-9000-W4 \
      --calibration_jsonl data/train.jsonl \
      --image_base_dir data/images \
      --num_samples 5000
"""

import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="W4A16 quantization for fine-tuned Qwen2.5-VL models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Required
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to the full-precision model directory.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory for the quantized model output.")
    parser.add_argument("--calibration_jsonl", type=str, required=True,
                        help="JSONL file with calibration samples.")
    parser.add_argument("--image_base_dir", type=str, required=True,
                        help="Root directory for images referenced in the JSONL.")

    # Optional quantization hyper-parameters
    parser.add_argument("--bits", type=int, default=4,
                        help="Weight quantization bit-width (default: 4).")
    parser.add_argument("--group_size", type=int, default=128,
                        help="Quantization group size (default: 128).")
    parser.add_argument("--num_samples", type=int, default=5000,
                        help="Number of calibration samples (default: 5000).")
    parser.add_argument("--iters", type=int, default=200,
                        help="AutoRound optimization iterations (default: 200).")
    parser.add_argument("--seqlen", type=int, default=2048,
                        help="Maximum sequence length for calibration (default: 2048).")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42).")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Patch config.json for vLLM
# ---------------------------------------------------------------------------

def fix_config_for_vllm(
    source_model_path: str,
    quantized_model_path: str,
) -> None:
    """Copy vision-token ID fields from the source model's config.json into
    the quantized model's config.json so that vLLM can locate them.

    AutoRound may drop these top-level fields during quantization, causing
    vLLM to raise an ``AttributeError`` on ``vision_start_token_id`` etc.
    """
    source_cfg_path = os.path.join(source_model_path, "config.json")
    quant_cfg_path = os.path.join(quantized_model_path, "config.json")

    if not os.path.exists(source_cfg_path):
        logger.warning("Source config.json not found: %s", source_cfg_path)
        return
    if not os.path.exists(quant_cfg_path):
        logger.warning("Quantized config.json not found: %s", quant_cfg_path)
        return

    with open(source_cfg_path, "r", encoding="utf-8") as f:
        source_cfg = json.load(f)
    with open(quant_cfg_path, "r", encoding="utf-8") as f:
        quant_cfg = json.load(f)

    required_fields = [
        "vision_start_token_id",
        "vision_end_token_id",
        "vision_token_id",
        "image_token_id",
        "video_token_id",
    ]

    added: List[str] = []
    for field in required_fields:
        if field in source_cfg and field not in quant_cfg:
            quant_cfg[field] = source_cfg[field]
            added.append(field)

        # Also ensure text_config has these fields.
        if "text_config" in quant_cfg:
            if field in source_cfg and field not in quant_cfg["text_config"]:
                quant_cfg["text_config"][field] = source_cfg[field]

    if added:
        with open(quant_cfg_path, "w", encoding="utf-8") as f:
            json.dump(quant_cfg, f, indent=2, ensure_ascii=False)
        logger.info("  Patched config.json -- added: %s", ", ".join(added))
    else:
        logger.info("  config.json already has all required fields.")


# ---------------------------------------------------------------------------
# Calibration data: load and stratified-sample from training JSONL
# ---------------------------------------------------------------------------

def load_calibration_samples(
    calibration_jsonl: str,
    image_base_dir: str,
    num_samples: int = 5000,
    seed: int = 42,
) -> List[dict]:
    """Load calibration samples with stratified sampling by sleep stage.

    Each line in the JSONL is expected to have:
      - ``id``: formatted as ``subject#epoch_stage`` (e.g. ``01-03-0001#100_N3``).
      - ``messages``: list of chat messages; user messages may contain
        ``{"type": "image", "image": "<relative_path>"}`` items.

    Returns a list of sample dicts, each augmented with an ``_image_paths``
    key that holds the resolved absolute paths.
    """
    logger.info("Loading calibration data from %s", calibration_jsonl)
    logger.info("Image base directory: %s", image_base_dir)

    stage_buckets: Dict[str, List[dict]] = defaultdict(list)

    with open(calibration_jsonl, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                sample = json.loads(line)
            except json.JSONDecodeError:
                if line_idx < 10:
                    logger.warning("Skipping malformed JSON at line %d", line_idx)
                continue

            sample_id = sample.get("id", "")

            # Extract the sleep-stage label from the ID.
            if "#" in sample_id:
                stage = sample_id.split("#")[1].split("_")[-1]
            else:
                stage = "unknown"

            # Resolve and validate all image paths.
            messages = sample.get("messages", [])
            image_paths: List[str] = []
            all_exist = True

            for msg in messages:
                if msg.get("role") != "user" or not isinstance(msg.get("content"), list):
                    continue
                for item in msg["content"]:
                    if item.get("type") == "image":
                        rel = item.get("image", "")
                        full = os.path.join(image_base_dir, rel)
                        if os.path.exists(full):
                            image_paths.append(full)
                        else:
                            all_exist = False
                            break
                if not all_exist:
                    break

            if all_exist and image_paths:
                sample["_image_paths"] = image_paths
                stage_buckets[stage].append(sample)

    # Report per-stage counts.
    logger.info("Stage distribution in calibration pool:")
    for stg in sorted(stage_buckets):
        logger.info("  %s: %d samples", stg, len(stage_buckets[stg]))

    # Stratified sampling: allocate quota proportional to stage frequency.
    random.seed(seed)
    total_valid = sum(len(v) for v in stage_buckets.values())
    selected: List[dict] = []

    for stg, bucket in stage_buckets.items():
        quota = max(1, int(num_samples * len(bucket) / total_valid))
        quota = min(quota, len(bucket))
        chosen = random.sample(bucket, quota)
        selected.extend(chosen)
        logger.info("  Sampled %d from stage %s", len(chosen), stg)

    # Top up if we are still short.
    shortfall = num_samples - len(selected)
    if shortfall > 0:
        pool = [s for bucket in stage_buckets.values() for s in bucket]
        available = [s for s in pool if s not in selected]
        if available:
            extra = random.sample(available, min(shortfall, len(available)))
            selected.extend(extra)

    selected = selected[:num_samples]
    logger.info("Total calibration samples selected: %d", len(selected))
    return selected


# ---------------------------------------------------------------------------
# Calibration data: convert samples to formatted text via chat template
# ---------------------------------------------------------------------------

def prepare_calibration_texts(
    samples: List[dict],
    processor,
) -> List[str]:
    """Apply the Qwen2.5-VL chat template to each sample and return a list
    of formatted strings suitable for AutoRound calibration."""
    texts: List[str] = []

    for sample in samples:
        messages = sample.get("messages", [])
        image_paths = sample.get("_image_paths", [])

        # Rebuild messages with absolute image paths.
        rebuilt: List[dict] = []
        img_idx = 0

        for msg in messages:
            role = msg.get("role")
            content = msg.get("content")

            if role == "user" and isinstance(content, list):
                new_content = []
                for item in content:
                    if item.get("type") == "image" and img_idx < len(image_paths):
                        new_content.append({
                            "type": "image",
                            "image": image_paths[img_idx],
                        })
                        img_idx += 1
                    elif item.get("type") == "text":
                        new_content.append({
                            "type": "text",
                            "text": item.get("text", ""),
                        })
                rebuilt.append({"role": role, "content": new_content})
            else:
                rebuilt.append({"role": role, "content": content})

        try:
            text = processor.apply_chat_template(
                rebuilt, tokenize=False, add_generation_prompt=True,
            )
            texts.append(text)
        except Exception as exc:
            logger.debug("Skipping sample during template application: %s", exc)
            continue

    return texts


# ---------------------------------------------------------------------------
# Core quantization routine
# ---------------------------------------------------------------------------

def quantize_model(
    model_path: str,
    output_dir: str,
    calibration_jsonl: str,
    image_base_dir: str,
    bits: int = 4,
    group_size: int = 128,
    num_samples: int = 5000,
    iters: int = 200,
    seqlen: int = 2048,
    seed: int = 42,
) -> bool:
    """Run the full quantization pipeline and return True on success."""

    # Deferred import -- only available when the auto_round environment is
    # active (e.g. ``conda activate awq_autoround``).
    from auto_round.compressors.mllm.compressor import MLLMCompressor, get_template

    logger.info("=" * 60)
    logger.info("W%dA16 quantization (AutoRound)", bits)
    logger.info("=" * 60)
    logger.info("  Model path         : %s", model_path)
    logger.info("  Output directory   : %s", output_dir)
    logger.info("  Calibration JSONL  : %s", calibration_jsonl)
    logger.info("  Image base dir     : %s", image_base_dir)
    logger.info("  Quantization config: W%dA16, group_size=%d, iters=%d",
                bits, group_size, iters)
    logger.info("  Calibration samples: %d", num_samples)

    # Step 1: Load model and processor.
    logger.info("[Step 1/5] Loading model ...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    logger.info("  Model class: %s", type(model).__name__)

    # Step 2: Load calibration data with stratified sampling.
    logger.info("[Step 2/5] Loading calibration data ...")
    samples = load_calibration_samples(
        calibration_jsonl=calibration_jsonl,
        image_base_dir=image_base_dir,
        num_samples=num_samples,
        seed=seed,
    )
    if not samples:
        logger.error("No valid calibration samples found. Check data paths.")
        return False

    # Step 3: Build calibration texts.
    logger.info("[Step 3/5] Preparing calibration texts ...")
    calibration_texts = prepare_calibration_texts(samples, processor)
    logger.info("  Generated %d calibration texts.", len(calibration_texts))
    if not calibration_texts:
        logger.error("No calibration texts could be generated.")
        return False

    # Step 4: Create the quantizer.
    logger.info("[Step 4/5] Setting up AutoRound quantizer ...")
    template = get_template(
        "qwen2_5_vl",
        model=model,
        tokenizer=tokenizer,
        processor=processor,
    )
    logger.info("  Template model type: %s", template.model_type)

    scheme = f"W{bits}A16"
    compressor = MLLMCompressor(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        scheme=scheme,
        dataset=calibration_texts,
        nsamples=len(calibration_texts),
        iters=iters,
        seqlen=seqlen,
        batch_size=1,
        seed=seed,
    )
    compressor.template = template

    # Step 5: Quantize, save, and patch config.
    logger.info("[Step 5/5] Quantizing and saving (this may take 20-40 min) ...")
    os.makedirs(output_dir, exist_ok=True)

    compressor.quantize_and_save(
        output_dir,
        format="auto_round",
        inplace=False,
    )
    processor.save_pretrained(output_dir)

    logger.info("Patching config.json for vLLM compatibility ...")
    fix_config_for_vllm(model_path, output_dir)

    # Print compression summary.
    original_bytes = sum(
        f.stat().st_size for f in Path(model_path).glob("**/*") if f.is_file()
    )
    quantized_bytes = sum(
        f.stat().st_size for f in Path(output_dir).glob("**/*") if f.is_file()
    )
    orig_gb = original_bytes / (1024 ** 3)
    quant_gb = quantized_bytes / (1024 ** 3)

    logger.info("=" * 60)
    logger.info("Quantization complete.")
    logger.info("  Original model : %.2f GB", orig_gb)
    logger.info("  Quantized model: %.2f GB", quant_gb)
    if quant_gb > 0:
        logger.info("  Compression    : %.2fx", orig_gb / quant_gb)
    logger.info("  Output dir     : %s", output_dir)
    logger.info("=" * 60)

    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if not os.path.exists(args.model_path):
        logger.error("Model path does not exist: %s", args.model_path)
        sys.exit(1)
    if not os.path.exists(args.calibration_jsonl):
        logger.error("Calibration JSONL not found: %s", args.calibration_jsonl)
        sys.exit(1)
    if not os.path.exists(args.image_base_dir):
        logger.error("Image base directory not found: %s", args.image_base_dir)
        sys.exit(1)

    success = quantize_model(
        model_path=args.model_path,
        output_dir=args.output_dir,
        calibration_jsonl=args.calibration_jsonl,
        image_base_dir=args.image_base_dir,
        bits=args.bits,
        group_size=args.group_size,
        num_samples=args.num_samples,
        iters=args.iters,
        seqlen=args.seqlen,
        seed=args.seed,
    )

    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
