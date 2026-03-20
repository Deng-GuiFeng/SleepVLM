#!/usr/bin/env python3
"""Unified LoRA training script for Qwen2.5-VL (Phase 1 WPT / Phase 2 SFT).

Supports two training phases via --freeze_vision_encoder:
  Phase 1 (WPT): Vision encoder unfrozen. LoRA applied to all nn.Linear except lm_head.
  Phase 2 (SFT): Vision encoder frozen. LoRA applied to language model nn.Linear only.

Training only -- no validation. Periodic and epoch-end checkpoint saves.

Usage:
    accelerate launch --multi_gpu \
        sleepvlm/training/train.py \
        --model_path models/Qwen2.5-VL-3B-Instruct \
        --train_file data/phase2_sft/train.jsonl \
        --image_root data \
        --output_dir outputs/phase2_sft
"""

import os
import json
import argparse
import logging
import math
import platform
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoProcessor,
    Qwen2_5_VLForConditionalGeneration,
    get_scheduler,
    set_seed as hf_set_seed,
)
import transformers as _transformers
from accelerate import Accelerator, DistributedDataParallelKwargs
from tqdm.auto import tqdm
from peft import LoraConfig, TaskType, get_peft_model

# Disable TorchDynamo to avoid FlashAttention + FakeTensor conflicts.
try:
    import torch._dynamo as _dynamo

    _dynamo.config.suppress_errors = True
    _dynamo.disable()
except Exception:
    pass

try:
    from qwen_vl_utils import process_vision_info
except Exception as exc:
    raise RuntimeError(
        "Missing dependency qwen-vl-utils. Install: pip install qwen-vl-utils"
    ) from exc


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _is_url_or_data_uri(path: str) -> bool:
    """Return True if *path* is a URL, data URI, or file URI."""
    return (
        path.startswith("http://")
        or path.startswith("https://")
        or path.startswith("data:")
        or path.startswith("file://")
    )


def _resolve_image_path(image: str, image_root: Optional[str]) -> str:
    """Resolve a relative image path against *image_root*.

    Absolute paths, URLs, and data URIs are returned unchanged.
    """
    if _is_url_or_data_uri(image) or image_root is None:
        return image
    p = Path(image)
    if p.is_absolute():
        return str(p)
    return str((Path(image_root) / image).resolve())


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class JSONLSFTDataset(Dataset):
    """Reads a JSONL file of multi-turn conversations with optional images.

    Each line must contain a ``messages`` key whose value is a list of
    role/content dicts (system, user, assistant).  Image paths inside user
    messages are resolved against *image_root*.

    For label masking the collator needs two views:
      * ``messages_full``   -- the complete conversation (for the full target)
      * ``messages_prompt`` -- everything except the *last* assistant turn
        (so the collator can measure the prompt length).
    """

    def __init__(self, jsonl_path: str, image_root: Optional[str] = None):
        self.path = Path(jsonl_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Training file not found: {self.path}")
        self.image_root = image_root
        self.samples: List[Dict[str, Any]] = []

        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "messages" not in item:
                    continue

                messages = item["messages"]
                # Resolve relative image paths in user messages.
                for msg in messages:
                    if msg.get("role") != "user":
                        continue
                    content = msg.get("content", [])
                    if isinstance(content, list):
                        for part in content:
                            if isinstance(part, dict) and part.get("type") == "image":
                                img = part.get("image")
                                if isinstance(img, str):
                                    part["image"] = _resolve_image_path(
                                        img, self.image_root
                                    )
                self.samples.append({"messages": messages})

        if not self.samples:
            raise ValueError(f"No valid samples found in {self.path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        messages: List[Dict[str, Any]] = item["messages"]

        # Build the prompt view: drop the last assistant message so the
        # collator can identify where supervised content begins.
        messages_prompt: List[Dict[str, Any]] = []
        last_assistant_found = False
        for msg in reversed(messages):
            if msg.get("role") == "assistant" and not last_assistant_found:
                last_assistant_found = True
                continue
            messages_prompt.insert(0, msg)

        return {"messages_full": messages, "messages_prompt": messages_prompt}


# ---------------------------------------------------------------------------
# Collator -- builds input tensors and label masks
# ---------------------------------------------------------------------------

class Collator:
    """Tokenize conversations and create labels that supervise only the last
    assistant turn.

    Everything outside the last assistant's content span receives label=-100,
    so the loss is computed only on the answer tokens.
    """

    def __init__(self, processor: AutoProcessor):
        self.processor = processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts_full: List[str] = []
        prompt_lens: List[int] = []
        all_image_inputs: List[Any] = []

        for sample in batch:
            messages_full = sample["messages_full"]
            messages_prompt = sample["messages_prompt"]

            text_full = self.processor.apply_chat_template(
                messages_full, tokenize=False, add_generation_prompt=False
            )
            text_prompt = self.processor.apply_chat_template(
                messages_prompt, tokenize=False, add_generation_prompt=True
            )
            texts_full.append(text_full)

            prompt_ids = self.processor.tokenizer(
                text_prompt, add_special_tokens=False
            ).input_ids
            prompt_lens.append(len(prompt_ids))

            image_inputs, _ = process_vision_info(messages_full)
            all_image_inputs.append(image_inputs)

        proc_kwargs: Dict[str, Any] = dict(
            text=texts_full, padding=True, return_tensors="pt"
        )
        if any(x is not None for x in all_image_inputs):
            proc_kwargs["images"] = all_image_inputs
        inputs = self.processor(**proc_kwargs)

        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        labels = torch.full_like(input_ids, -100)

        tokenizer = self.processor.tokenizer
        im_start_token = "<|im_start|>"
        im_end_token = "<|im_end|>"

        for i in range(input_ids.size(0)):
            ids_i = input_ids[i].tolist()
            effective_len = (
                int(attention_mask[i].sum().item())
                if attention_mask is not None
                else len(ids_i)
            )
            ids_trim = ids_i[:effective_len]
            tokens = tokenizer.convert_ids_to_tokens(ids_trim)

            # Walk the token sequence and find the *last* assistant span.
            answer_start: Optional[int] = None
            answer_end: Optional[int] = None
            j = 0
            while j < len(tokens):
                if tokens[j] == im_start_token:
                    window = tokens[j + 1 : j + 5]
                    if any(tok == "assistant" for tok in window):
                        try:
                            rel = window.index("assistant")
                            assistant_tok_idx = j + 1 + rel
                        except ValueError:
                            assistant_tok_idx = None
                        k = (
                            assistant_tok_idx + 1
                            if assistant_tok_idx is not None
                            else j + 1
                        )
                        # Skip newline tokens between role tag and content.
                        while k < len(tokens) and tokens[k] in ["\n"]:
                            k += 1
                        m = k
                        while m < len(tokens) and tokens[m] != im_end_token:
                            m += 1
                        answer_start = k
                        answer_end = m
                        j = m
                        continue
                j += 1

            if (
                answer_start is not None
                and answer_end is not None
                and answer_end > answer_start
            ):
                # Strip leading whitespace/newline tokens from the span.
                trim_start = answer_start
                while trim_start < answer_end:
                    try:
                        piece = tokenizer.decode(
                            [ids_trim[trim_start]], skip_special_tokens=False
                        )
                    except Exception:
                        piece = ""
                    if piece.strip() == "" or piece in ["\n", "\r", "\t"]:
                        trim_start += 1
                        continue
                    break
                if trim_start < answer_end:
                    labels[i, trim_start:answer_end] = input_ids[
                        i, trim_start:answer_end
                    ]
            else:
                # Fallback: supervise everything after the prompt.
                pl = prompt_lens[i] if i < len(prompt_lens) else 0
                labels[i, pl:effective_len] = input_ids[i, pl:effective_len]

        inputs["labels"] = labels
        inputs["prompt_lens"] = torch.tensor(prompt_lens, dtype=torch.long)
        return inputs


# ---------------------------------------------------------------------------
# LoRA target module discovery
# ---------------------------------------------------------------------------

def find_all_linear_names(
    model: torch.nn.Module, freeze_vision_encoder: bool = False
) -> List[str]:
    """Return the leaf module names of all nn.Linear layers suitable for LoRA.

    * Always excludes ``lm_head``.
    * When *freeze_vision_encoder* is True, also excludes modules inside the
      vision encoder (names containing 'visual' or 'vision').
    """
    lora_module_names: set[str] = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        parts = name.split(".")
        if "lm_head" in parts:
            continue
        if freeze_vision_encoder and ("visual" in parts or "vision" in parts):
            continue
        leaf = parts[-1]
        if leaf.isdigit():
            continue
        lora_module_names.add(leaf)

    lora_module_names.discard("lm_head")
    return sorted(lora_module_names)


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

def _build_optimizer(
    model: torch.nn.Module, lr: float, weight_decay: float
) -> torch.optim.Optimizer:
    """Build an AdamW optimizer with separate decay / no-decay groups.

    Bias terms and normalization weights are excluded from weight decay.
    """
    no_decay_keywords = {
        "bias",
        "LayerNorm.weight",
        "layernorm.weight",
        "ln_f.weight",
        "norm.weight",
        "rmsnorm.weight",
        "norm.bias",
        "LayerNorm.bias",
        "layernorm.bias",
        "rmsnorm.bias",
    }
    decay_params: List[torch.nn.Parameter] = []
    no_decay_params: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(nd in name for nd in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        betas=(0.9, 0.95),
        eps=1e-8,
    )


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _setup_logging(
    output_dir: Path, is_main_process: bool, process_index: int
) -> logging.Logger:
    """Configure a logger that writes to file (all ranks) and console (rank 0)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"

    logger = logging.getLogger("sleepvlm.train")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter(
            fmt=f"%(asctime)s | %(levelname)s | rank={process_index} | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if is_main_process:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    return logger


def _log_env_and_args(
    logger: logging.Logger, args: argparse.Namespace, accelerator: Accelerator
) -> None:
    """Log environment info and hyperparameters for reproducibility."""
    try:
        import qwen_vl_utils as _qvu

        qvu_ver = getattr(_qvu, "__version__", "unknown")
    except Exception:
        qvu_ver = "unavailable"

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    cuda_ok = torch.cuda.is_available()
    gpu_info: List[Dict[str, Any]] = []
    if cuda_ok:
        try:
            for i in range(torch.cuda.device_count()):
                gpu_info.append(
                    {
                        "index": i,
                        "name": torch.cuda.get_device_name(i),
                        "capability": torch.cuda.get_device_capability(i),
                        "total_mem_GB": round(
                            torch.cuda.get_device_properties(i).total_memory
                            / 1024**3,
                            2,
                        ),
                    }
                )
        except Exception:
            pass

    logger.info("=========== Environment ===========")
    logger.info(f"Host: {platform.node()} | OS: {platform.platform()}")
    logger.info(
        f"Python: {platform.python_version()} | "
        f"Torch: {torch.__version__} | "
        f"Transformers: {_transformers.__version__}"
    )
    logger.info(
        f"Accelerate processes: {accelerator.num_processes} | "
        f"local_process_index: {accelerator.process_index}"
    )
    logger.info(f"qwen-vl-utils: {qvu_ver}")
    logger.info(f"CUDA available: {cuda_ok} | CUDA_VISIBLE_DEVICES: {cuda_visible}")
    for info in gpu_info:
        logger.info(
            f"  GPU[{info['index']}]: {info['name']} | "
            f"capability={info['capability']} | "
            f"total_mem={info['total_mem_GB']} GB"
        )
    logger.info("=========== Hyperparameters =======")
    for key, val in sorted(vars(args).items()):
        logger.info(f"  {key}: {val}")
    logger.info("===================================")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified LoRA training for Qwen2.5-VL (Phase 1 WPT / Phase 2 SFT)"
    )

    # Paths
    parser.add_argument(
        "--model_path", type=str, required=True,
        help="Path or HuggingFace repo ID for the base model",
    )
    parser.add_argument("--train_file", type=str, required=True, help="JSONL training data")
    parser.add_argument(
        "--image_root", type=str, default=None,
        help="Root directory to resolve relative image paths",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Directory for checkpoints and logs")

    # Training
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every N optimizer update steps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker count")
    parser.add_argument("--log_every", type=int, default=1, help="Log metrics every N micro-steps")

    # Image processing
    parser.add_argument("--min_pixels", type=int, default=4 * 28 * 28, help="Min pixels for image resizing (default 3136)")
    parser.add_argument("--max_pixels", type=int, default=1280 * 28 * 28, help="Max pixels for image resizing (default 1003520)")

    # TensorBoard
    parser.add_argument("--tensorboard", action="store_true", help="Enable TensorBoard logging")
    parser.add_argument("--tb_log_dir", type=str, default=None, help="TensorBoard log dir (default: output_dir/tensorboard)")

    # Phase control
    parser.add_argument(
        "--freeze_vision_encoder", action="store_true",
        help="Freeze vision encoder (Phase 2 SFT). Omit for Phase 1 WPT.",
    )

    # LoRA
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout rate")
    parser.add_argument(
        "--lora_target_modules", type=str, nargs="+", default=None,
        help="Explicit list of LoRA target module names. Auto-detected if omitted.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # ---- Accelerator (auto-detects available GPUs) ----
    accelerator = Accelerator(
        mixed_precision="bf16",
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[
            DistributedDataParallelKwargs(find_unused_parameters=False)
        ],
    )
    hf_set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    output_dir = Path(args.output_dir)
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Logging ----
    logger = _setup_logging(
        output_dir, accelerator.is_main_process, accelerator.process_index
    )
    if accelerator.is_main_process:
        phase = "Phase 2 (SFT)" if args.freeze_vision_encoder else "Phase 1 (WPT)"
        logger.info(f"Training phase: {phase}")
        logger.info(f"Log file: {(output_dir / 'training.log').as_posix()}")
    _log_env_and_args(logger, args, accelerator)

    # ---- TensorBoard (optional, main process only) ----
    writer = None
    if accelerator.is_main_process and args.tensorboard:
        tb_dir = (
            Path(args.tb_log_dir) if args.tb_log_dir else (output_dir / "tensorboard")
        )
        tb_dir.mkdir(parents=True, exist_ok=True)
        try:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard enabled. Log dir: {tb_dir}")
            try:
                writer.add_text("env/host", platform.node(), 0)
                writer.add_text("env/torch", torch.__version__, 0)
                writer.add_text(
                    "hparams",
                    json.dumps(vars(args), ensure_ascii=False, indent=2)[:8000],
                    0,
                )
            except Exception:
                pass
        except Exception as exc:
            writer = None
            logger.warning(
                f"Failed to initialize TensorBoard: {exc}. "
                "Install with: pip install tensorboard"
            )

    # ---- Processor and Model ----
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        use_fast=True,
    )
    logger.info(
        f"Loaded processor from: {args.model_path} "
        f"(min_pixels={args.min_pixels}, max_pixels={args.max_pixels})"
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16
    )
    model.config.use_cache = False

    # Enable gradient checkpointing for memory efficiency.
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        model.enable_input_require_grads()
        logger.info("Gradient checkpointing enabled (use_reentrant=False).")

    # ---- LoRA ----
    if args.lora_target_modules is None:
        target_modules = find_all_linear_names(
            model, freeze_vision_encoder=args.freeze_vision_encoder
        )
    else:
        target_modules = args.lora_target_modules
    logger.info(f"LoRA target modules: {target_modules}")

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Model parameters -- total: {total_params:,} | trainable: {trainable_params:,}"
    )

    # ---- Dataset and DataLoader ----
    train_dataset = JSONLSFTDataset(args.train_file, image_root=args.image_root)
    logger.info(f"Training samples: {len(train_dataset)}")

    collator = Collator(processor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.per_device_train_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collator,
        pin_memory=True,
    )

    # ---- Optimizer and LR Scheduler ----
    optimizer = _build_optimizer(
        model, lr=args.learning_rate, weight_decay=args.weight_decay
    )

    num_update_steps_per_epoch = math.ceil(
        len(train_loader) / args.gradient_accumulation_steps
    )
    total_training_steps = num_update_steps_per_epoch * args.epochs
    num_warmup_steps = int(total_training_steps * args.warmup_ratio)

    if accelerator.is_main_process:
        logger.info(
            f"Schedule: batches/epoch={len(train_loader)} | "
            f"grad_accum={args.gradient_accumulation_steps} | "
            f"updates/epoch={num_update_steps_per_epoch} | "
            f"total_updates={total_training_steps} | "
            f"warmup_steps={num_warmup_steps}"
        )

    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=total_training_steps,
    )

    # ---- Prepare with Accelerator ----
    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )

    # ---- Training Loop ----
    global_step = 0
    update_step = 0
    grad_clip_warned = False
    tokens_processed = 0
    train_start_time = time.perf_counter()

    total_micro_steps = len(train_loader) * args.epochs
    show_progress = accelerator.is_main_process
    pbar = tqdm(total=total_micro_steps, disable=not show_progress, dynamic_ncols=True)

    for epoch in range(args.epochs):
        model.train()

        # Set epoch on distributed sampler for proper shuffling.
        try:
            sampler = getattr(train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
        except Exception:
            pass

        for step, batch in enumerate(train_loader):
            iter_start = time.perf_counter()

            with accelerator.accumulate(model):
                prompt_lens = batch.pop("prompt_lens", None)
                batch = {
                    k: v.to(accelerator.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                batch_tokens = (
                    int(batch["attention_mask"].sum().item())
                    if isinstance(batch.get("attention_mask"), torch.Tensor)
                    else 0
                )

                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                grad_norm_val = None
                if accelerator.sync_gradients:
                    try:
                        grad_norm_val = accelerator.clip_grad_norm_(
                            model.parameters(), args.max_grad_norm
                        )
                    except Exception as exc:
                        if accelerator.is_main_process and not grad_clip_warned:
                            logger.warning(f"clip_grad_norm_ failed: {exc!r}")
                            grad_clip_warned = True

                optimizer.step()
                if accelerator.sync_gradients:
                    lr_scheduler.step()
                    update_step += 1
                optimizer.zero_grad()

            global_step += 1
            if show_progress:
                pbar.update(1)
                if global_step % max(1, args.log_every) == 0:
                    pbar.set_postfix(
                        {"epoch": epoch, "step": global_step, "loss": f"{loss.item():.4f}"}
                    )

            # -- Debug: decode supervised span on the very first step --
            if (
                accelerator.is_main_process
                and global_step == 1
                and prompt_lens is not None
            ):
                try:
                    idx0 = 0
                    pl0 = int(prompt_lens[idx0])
                    ids0 = batch["input_ids"][idx0].detach().cpu()
                    lbl0 = batch["labels"][idx0].detach().cpu()
                    sup_pos = (lbl0 != -100).nonzero(as_tuple=True)[0]
                    sup_count = int(sup_pos.numel())
                    sup_tokens = ids0[sup_pos]
                    decoded = processor.tokenizer.decode(
                        sup_tokens.tolist(), skip_special_tokens=False
                    )
                    logger.info(
                        f"[DebugSupervisedSpan] sample=0 prompt_len={pl0} "
                        f"supervised_tokens={sup_count} snippet={decoded!r}"
                    )
                except Exception as exc:
                    logger.warning(f"[DebugSupervisedSpan] failed: {exc}")

            # -- Periodic logging --
            if accelerator.is_main_process and (
                global_step % max(1, args.log_every) == 0
            ):
                step_time = time.perf_counter() - iter_start
                tokens_processed += batch_tokens

                # Current learning rate
                try:
                    lr_list = getattr(
                        lr_scheduler,
                        "get_last_lr",
                        lambda: [optimizer.param_groups[0]["lr"]],
                    )()
                    lr_val = lr_list[0] if isinstance(lr_list, (list, tuple)) else float(lr_list)
                except Exception:
                    lr_val = optimizer.param_groups[0]["lr"]

                # GPU memory
                mem_alloc = mem_reserved = peak_mem = None
                if torch.cuda.is_available():
                    try:
                        mem_alloc = round(torch.cuda.memory_allocated() / (1024**2), 1)
                        mem_reserved = round(torch.cuda.memory_reserved() / (1024**2), 1)
                        peak_mem = round(torch.cuda.max_memory_allocated() / (1024**2), 1)
                    except Exception:
                        pass

                # Supervised token counts
                try:
                    if isinstance(batch.get("labels"), torch.Tensor):
                        sup_counts = (batch["labels"] != -100).sum(dim=1).tolist()
                        if len(sup_counts) > 8:
                            sc_str = (
                                f"min={min(sup_counts)} max={max(sup_counts)} "
                                f"avg={sum(sup_counts)/len(sup_counts):.1f}"
                            )
                        else:
                            sc_str = str(sup_counts)
                    else:
                        sc_str = "n/a"
                except Exception:
                    sc_str = "err"

                sched_phase = (
                    "warmup"
                    if update_step < num_warmup_steps
                    else ("decay" if update_step <= total_training_steps else "post")
                )
                mem_str = (
                    f"mem(MB) alloc={mem_alloc} reserved={mem_reserved} peak={peak_mem}"
                    if mem_alloc is not None
                    else "mem(MB)=n/a"
                )
                tps = batch_tokens / step_time if step_time > 0 else 0.0

                logger.info(
                    " | ".join(
                        [
                            f"epoch={epoch}",
                            f"micro_step={global_step}",
                            f"update_step={update_step}",
                            f"loss={loss.item():.6f}",
                            f"lr={lr_val:.6e}",
                            f"phase={sched_phase}",
                            f"grad_norm={grad_norm_val if grad_norm_val is None else float(grad_norm_val)}",
                            f"supervised_tok={sc_str}",
                            f"batch_tokens={batch_tokens}",
                            f"tok/s={tps:.1f}",
                            mem_str,
                        ]
                    )
                )

                # TensorBoard scalars
                if writer is not None:
                    try:
                        writer.add_scalar("train/loss", float(loss.item()), global_step)
                        writer.add_scalar("train/lr", float(lr_val), global_step)
                        writer.add_scalar("train/update_step", float(update_step), global_step)
                        if grad_norm_val is not None:
                            try:
                                writer.add_scalar(
                                    "train/grad_norm", float(grad_norm_val), global_step
                                )
                            except Exception:
                                pass
                        if isinstance(batch.get("labels"), torch.Tensor):
                            avg_sup = float(
                                sum(sup_counts) / len(sup_counts)
                            )
                            writer.add_scalar(
                                "train/avg_supervised_tokens", avg_sup, global_step
                            )
                        writer.add_scalar("train/batch_tokens", batch_tokens, global_step)
                        if step_time > 0:
                            writer.add_scalar("train/tokens_per_sec", tps, global_step)
                            writer.add_scalar(
                                "train/step_time_sec", step_time, global_step
                            )
                        if mem_alloc is not None:
                            writer.add_scalar(
                                "cuda/mem_alloc_MB", float(mem_alloc), global_step
                            )
                            writer.add_scalar(
                                "cuda/mem_reserved_MB", float(mem_reserved), global_step
                            )
                            writer.add_scalar(
                                "cuda/peak_mem_alloc_MB", float(peak_mem), global_step
                            )
                        writer.add_scalar(
                            "train/epoch_progress",
                            float(epoch + (step + 1) / max(1, len(train_loader))),
                            global_step,
                        )
                    except Exception:
                        pass

            # -- Periodic checkpoint save --
            if (
                update_step > 0
                and update_step % args.save_steps == 0
                and accelerator.is_main_process
                and accelerator.sync_gradients
            ):
                ckpt_dir = output_dir / f"checkpoint-{update_step}"
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                accelerator.unwrap_model(model).save_pretrained(
                    ckpt_dir, safe_serialization=True
                )
                processor.save_pretrained(ckpt_dir)
                logger.info(
                    f"Saved checkpoint: {ckpt_dir} "
                    f"(update_step={update_step}, micro_step={global_step})"
                )

        # -- End-of-epoch checkpoint --
        if accelerator.is_main_process:
            epoch_ckpt = output_dir / f"epoch-{epoch + 1}"
            epoch_ckpt.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(
                epoch_ckpt, safe_serialization=True
            )
            processor.save_pretrained(epoch_ckpt)
            logger.info(f"Saved epoch checkpoint: {epoch_ckpt} (epoch={epoch + 1})")

    # ---- Cleanup ----
    pbar.close()
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Save a final checkpoint if the last update_step was not on a
        # save_steps boundary (and was not already saved as an epoch ckpt).
        if update_step > 0 and update_step % args.save_steps != 0:
            ckpt_dir = output_dir / f"checkpoint-{update_step}"
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            accelerator.unwrap_model(model).save_pretrained(
                ckpt_dir, safe_serialization=True
            )
            processor.save_pretrained(ckpt_dir)
            logger.info(
                f"Saved final checkpoint: {ckpt_dir} "
                f"(update_step={update_step}, micro_step={global_step})"
            )

        total_time = time.perf_counter() - train_start_time
        logger.info(
            f"Training complete. "
            f"total_time={total_time / 60:.2f} min | "
            f"total_updates={update_step} | "
            f"tokens_processed={tokens_processed}"
        )
        if writer is not None:
            try:
                writer.add_scalar(
                    "metrics/total_updates", float(update_step), global_step
                )
                writer.close()
            except Exception:
                pass

    try:
        accelerator.end_training()
    except Exception:
        pass


if __name__ == "__main__":
    main()
