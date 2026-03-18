#!/usr/bin/env python3
"""
Merge LoRA adapters into a Qwen2.5-VL base model.

Supports two modes:
  - Single merge: merge one adapter checkpoint into the base model.
  - Batch merge:  merge every checkpoint-* directory under a given folder.

After merging, the script patches config.json so that vLLM can find the
vision token IDs at the top level (they sometimes exist only inside
text_config after a PEFT merge).

Usage examples:

  # Single merge (auto-detect base model from adapter_config.json):
  python scripts/merge_lora.py \
      --adapter_path runs/sft/checkpoint-5000 \
      --output_path runs/sft/merged/checkpoint-5000

  # Single merge with explicit base model:
  python scripts/merge_lora.py \
      --base_model_path models/Qwen2.5-VL-3B-Instruct \
      --adapter_path runs/sft/checkpoint-5000 \
      --output_path runs/sft/merged/checkpoint-5000

  # Batch merge (all checkpoint-* dirs):
  python scripts/merge_lora.py --batch \
      --adapter_dir runs/sft/ \
      --output_dir runs/sft/merged/
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters into Qwen2.5-VL base model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # -- single merge mode --
    parser.add_argument(
        "--base_model_path", type=str, default=None,
        help="Path to the base model. Auto-detected from adapter_config.json "
             "if omitted.",
    )
    parser.add_argument(
        "--adapter_path", type=str, default=None,
        help="Path to a single LoRA adapter checkpoint.",
    )
    parser.add_argument(
        "--output_path", type=str, default=None,
        help="Directory where the merged model will be saved.",
    )

    # -- batch merge mode --
    parser.add_argument(
        "--batch", action="store_true",
        help="Enable batch mode: merge every matching checkpoint directory.",
    )
    parser.add_argument(
        "--adapter_dir", type=str, default=None,
        help="Parent directory that contains multiple LoRA checkpoint dirs.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Parent directory for all merged outputs (batch mode).",
    )
    parser.add_argument(
        "--clean", action="store_true",
        help="Remove the output directory before merging (batch mode only).",
    )
    parser.add_argument(
        "--pattern", type=str, default="checkpoint-*",
        help="Glob pattern used to discover checkpoint directories "
             "(default: checkpoint-*).",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helper: resolve the base-model path
# ---------------------------------------------------------------------------

def get_base_model_path(
    adapter_path: str,
    provided_path: Optional[str] = None,
) -> str:
    """Return the base-model path.

    Resolution order:
      1. Explicit ``--base_model_path`` from the CLI.
      2. ``base_model_name_or_path`` inside ``adapter_config.json``.
    Raises if neither source yields a valid path.
    """
    if provided_path:
        print(f"[info] Using base model from CLI: {provided_path}")
        return provided_path

    config_file = Path(adapter_path) / "adapter_config.json"
    if config_file.exists():
        with open(config_file, "r") as f:
            cfg = json.load(f)
        base_path = cfg.get("base_model_name_or_path")
        if base_path:
            print(f"[info] Base model read from adapter_config.json: {base_path}")
            if not Path(base_path).exists():
                raise FileNotFoundError(
                    f"Path listed in adapter_config.json does not exist: "
                    f"{base_path}. Use --base_model_path to override."
                )
            return base_path

    raise ValueError(
        "Cannot determine the base model path. "
        "Provide --base_model_path explicitly or make sure "
        "adapter_config.json contains base_model_name_or_path."
    )


# ---------------------------------------------------------------------------
# Helper: check whether a directory is a valid LoRA checkpoint
# ---------------------------------------------------------------------------

def is_lora_checkpoint(path: Path) -> bool:
    """A directory is a valid LoRA checkpoint when it contains both
    ``adapter_config.json`` and adapter weights (safetensors or bin)."""
    if not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = (
        (path / "adapter_model.safetensors").exists()
        or (path / "adapter_model.bin").exists()
    )
    return has_config and has_weights


# ---------------------------------------------------------------------------
# Helper: discover checkpoints in a directory
# ---------------------------------------------------------------------------

def find_lora_checkpoints(
    adapter_dir: str,
    pattern: str = "checkpoint-*",
) -> List[Tuple[Path, int]]:
    """Return a sorted list of ``(checkpoint_path, step_number)`` tuples."""
    root = Path(adapter_dir)
    if not root.exists():
        raise FileNotFoundError(f"Directory does not exist: {root}")

    checkpoints: List[Tuple[Path, int]] = []
    for path in root.glob(pattern):
        if is_lora_checkpoint(path):
            try:
                step = int(path.name.split("-")[-1])
            except ValueError:
                step = 0
            checkpoints.append((path, step))

    checkpoints.sort(key=lambda x: x[1])
    return checkpoints


# ---------------------------------------------------------------------------
# Helper: patch config.json for vLLM compatibility
# ---------------------------------------------------------------------------

def fix_config_for_vllm(output_path: str) -> None:
    """Copy vision-token IDs from ``text_config`` to the top level of
    ``config.json`` so that vLLM can locate them."""
    config_path = Path(output_path) / "config.json"
    if not config_path.exists():
        print(f"[warn] config.json not found at {config_path}")
        return

    with open(config_path, "r") as f:
        config = json.load(f)

    fields = [
        "vision_start_token_id",
        "vision_end_token_id",
        "image_token_id",
        "video_token_id",
    ]

    missing = [fld for fld in fields if fld not in config]
    if not missing:
        print("[info] config.json already contains all vision-token fields.")
        return

    text_config = config.get("text_config", {})
    patched: List[str] = []
    for fld in missing:
        if fld in text_config:
            config[fld] = text_config[fld]
            patched.append(f"{fld}={text_config[fld]}")

    if patched:
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"[info] Patched config.json for vLLM: {', '.join(patched)}")
    else:
        print(
            "[warn] Could not patch config.json -- vision-token fields are "
            "missing from text_config as well."
        )


# ---------------------------------------------------------------------------
# Core: merge a single LoRA adapter
# ---------------------------------------------------------------------------

def merge_single(
    adapter_path: str,
    output_path: str,
    base_model_path: Optional[str] = None,
    base_model=None,
    processor=None,
) -> Tuple:
    """Merge one LoRA adapter into the base model and save the result.

    Parameters
    ----------
    adapter_path : str
        Path to the LoRA adapter directory.
    output_path : str
        Where to save the merged model.
    base_model_path : str, optional
        Explicit base-model path. Auto-detected when *None*.
    base_model : PreTrainedModel, optional
        A pre-loaded base model (reused across batch calls).
    processor : AutoProcessor, optional
        A pre-loaded processor (reused across batch calls).

    Returns
    -------
    tuple of (base_model, processor, resolved_base_path)
        The caller can feed these back in to avoid redundant loads.
    """
    resolved_base_path = get_base_model_path(adapter_path, base_model_path)

    # Load the base model if not already provided.
    if base_model is None:
        print(f"Loading base model from {resolved_base_path} ...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            resolved_base_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

    print(f"Loading LoRA adapter from {adapter_path} ...")
    model = PeftModel.from_pretrained(base_model, adapter_path)

    print("Merging LoRA weights into base model ...")
    merged_model = model.merge_and_unload()

    os.makedirs(output_path, exist_ok=True)
    print(f"Saving merged model to {output_path} ...")
    merged_model.save_pretrained(output_path, safe_serialization=True)

    # Save processor alongside the model.
    if processor is None:
        processor = AutoProcessor.from_pretrained(resolved_base_path)
    processor.save_pretrained(output_path)

    # Patch config.json for vLLM.
    fix_config_for_vllm(output_path)

    # Free the merged model and reload a fresh base for the next round
    # (merge_and_unload modifies the model in place).
    del merged_model
    del model
    torch.cuda.empty_cache()

    base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        resolved_base_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    return base_model, processor, resolved_base_path


# ---------------------------------------------------------------------------
# Core: batch merge
# ---------------------------------------------------------------------------

def batch_merge(
    adapter_dir: str,
    output_dir: str,
    base_model_path: Optional[str] = None,
    clean: bool = False,
    pattern: str = "checkpoint-*",
) -> None:
    """Merge every matching LoRA checkpoint under *adapter_dir*."""
    print("=" * 60)
    print("Batch LoRA merge")
    print("=" * 60)
    print(f"  Source dir : {adapter_dir}")
    print(f"  Output dir : {output_dir}")
    print(f"  Pattern    : {pattern}")
    print(f"  Clean first: {clean}")
    print("=" * 60)

    checkpoints = find_lora_checkpoints(adapter_dir, pattern)
    if not checkpoints:
        print(f"[error] No valid LoRA checkpoints found (pattern: {pattern}).")
        return

    print(f"\nFound {len(checkpoints)} checkpoint(s):")
    for ckpt_path, step in checkpoints:
        print(f"  - {ckpt_path.name}  (step {step})")
    print()

    out = Path(output_dir)
    if clean and out.exists():
        print(f"Removing existing output directory: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    base_model = None
    processor = None
    resolved_base_path = None

    for i, (ckpt_path, step) in enumerate(checkpoints):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(checkpoints)}] Merging: {ckpt_path.name}")
        print("=" * 60)

        ckpt_output = out / ckpt_path.name
        try:
            base_model, processor, resolved_base_path = merge_single(
                adapter_path=str(ckpt_path),
                output_path=str(ckpt_output),
                base_model_path=base_model_path or resolved_base_path,
                base_model=base_model,
                processor=processor,
            )
            print(f"[ok] Saved to {ckpt_output}")
        except Exception as e:
            print(f"[error] Failed on {ckpt_path.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print(f"Batch merge complete. Processed {len(checkpoints)} checkpoint(s).")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    if args.batch:
        # -- batch mode --
        if not args.adapter_dir or not args.output_dir:
            raise ValueError(
                "Batch mode requires both --adapter_dir and --output_dir."
            )
        batch_merge(
            adapter_dir=args.adapter_dir,
            output_dir=args.output_dir,
            base_model_path=args.base_model_path,
            clean=args.clean,
            pattern=args.pattern,
        )
    else:
        # -- single mode --
        if not args.adapter_path or not args.output_path:
            raise ValueError(
                "Single mode requires both --adapter_path and --output_path."
            )

        base_model_path = get_base_model_path(
            args.adapter_path, args.base_model_path
        )

        print("=" * 60)
        print("LoRA merge configuration:")
        print(f"  Base model : {base_model_path}")
        print(f"  Adapter    : {args.adapter_path}")
        print(f"  Output     : {args.output_path}")
        print("=" * 60)

        print(f"\nLoading base model from {base_model_path} ...")
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        print(f"Loading LoRA adapter from {args.adapter_path} ...")
        model = PeftModel.from_pretrained(base_model, args.adapter_path)

        print("Merging weights ...")
        model = model.merge_and_unload()

        os.makedirs(args.output_path, exist_ok=True)
        print(f"Saving merged model to {args.output_path} ...")
        model.save_pretrained(args.output_path, safe_serialization=True)

        print(f"Saving processor from {base_model_path} ...")
        processor = AutoProcessor.from_pretrained(base_model_path)
        processor.save_pretrained(args.output_path)

        # Patch config.json for vLLM.
        fix_config_for_vllm(args.output_path)

        print("=" * 60)
        print(f"Merge complete. Output: {args.output_path}")
        print("=" * 60)


if __name__ == "__main__":
    main()
