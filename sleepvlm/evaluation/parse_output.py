"""
Parse model-generated JSON output for sleep staging predictions.

The fine-tuned model is expected to produce a JSON object (optionally wrapped
in a fenced code block) containing three keys:

    {
      "sleep_stage": "N2",
      "reasoning_text": "...",
      "applicable_rules": "N2.1, N2.3"
    }

This module provides a tolerant parser that handles common formatting
variations produced by language models: fenced code blocks, stray backticks,
fullwidth quotation marks, trailing commas, and bare JSON objects.
"""

from __future__ import annotations

import json
import re
from typing import Optional, Tuple

# Canonical mapping from AASM sleep-stage labels to integer indices.
STAGE_MAP = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "R": 4}


def parse_model_output(
    out_text: str,
) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    """Parse a model-generated response that should contain a JSON object.

    The function applies several heuristics in order of priority:

    1. **Fenced code block** -- extract the first ````` ```json ... ``` ````` or
       ````` ``` ... ``` ````` region and treat its body as JSON.
    2. **Bare JSON** -- locate the first ``{...}`` block that contains the key
       ``"sleep_stage"`` and attempt to parse it.
    3. **Tolerant cleanup** -- replace fullwidth quotation marks, strip trailing
       commas before ``}`` or ``]``, and retry parsing.

    Parameters
    ----------
    out_text : str
        Raw text output from the language model.

    Returns
    -------
    tuple of (sleep_stage, reasoning_text, applicable_rules, error_msg)
        On success *error_msg* is ``None`` and the other fields are populated.
        On failure the first three fields are ``None`` and *error_msg* carries a
        human-readable description of the problem.
    """
    text = out_text.strip()
    raw_json_str: Optional[str] = None

    try:
        # -- Step 1: Try to extract from a fenced code block ----------------
        fence_match = re.search(
            r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE
        )
        if fence_match:
            raw_json_str = fence_match.group(1).strip()
        else:
            # -- Step 2: Fall back to bare JSON containing "sleep_stage" ----
            brace_match = re.search(r"\{[\s\S]*?\}", text)
            if brace_match and "sleep_stage" in brace_match.group(0):
                raw_json_str = brace_match.group(0)

        if raw_json_str is None:
            raise ValueError("No JSON structure found in model output")

        # -- Step 3: Clean up common artifacts ------------------------------
        # Strip residual backticks that may surround the JSON body.
        raw_json_str = raw_json_str.strip().lstrip("`").rstrip("`").strip()

        # Replace fullwidth (smart) quotation marks with standard ASCII ones.
        raw_json_str = (
            raw_json_str.replace("\u201c", '"')
            .replace("\u201d", '"')
            .replace("\u2018", "'")
        )

        # -- Step 4: Parse JSON (with one retry after cleanup) --------------
        try:
            data = json.loads(raw_json_str)
        except json.JSONDecodeError:
            # Remove trailing commas before closing braces / brackets.
            cleaned = re.sub(r",\s*}\s*$", "}\n", raw_json_str)
            cleaned = re.sub(r",\s*]", "]", cleaned)
            data = json.loads(cleaned)

        # -- Step 5: Extract and validate fields ----------------------------
        sleep_stage = data.get("sleep_stage")
        reasoning_text = data.get("reasoning_text")
        applicable_rules = data.get("applicable_rules")

        if isinstance(sleep_stage, str):
            sleep_stage = sleep_stage.strip().upper()
        if sleep_stage not in STAGE_MAP:
            raise ValueError(
                f"Invalid sleep_stage value: {sleep_stage!r} "
                f"(expected one of {list(STAGE_MAP.keys())})"
            )

        # Normalize optional text fields.
        if isinstance(reasoning_text, str):
            reasoning_text = reasoning_text.strip()
        if isinstance(applicable_rules, str):
            applicable_rules = applicable_rules.strip()

        return sleep_stage, reasoning_text, applicable_rules, None

    except Exception as exc:  # noqa: BLE001
        return None, None, None, str(exc)
