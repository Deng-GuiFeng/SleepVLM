# -*- coding: utf-8 -*-
"""
wpt_targets.py -- Phase 1 WPT training targets: spectral band power and
amplitude features.

For every 30-second sleep epoch the module computes per-second (1-second
window) features from preprocessed PSG signals:

  * EEG / EOG channels (F4, C4, O2, LOC, ROC):
      - Welch PSD integrated over four canonical bands, reported in dB
        (10 * log10):
            delta  0.3 --  4 Hz
            theta  4   --  8 Hz
            alpha  8   -- 13 Hz
            beta  13   -- 30 Hz
      - Mean Absolute Value (MAV) in microvolts
      --> 5 values per second per channel

  * EMG channel (Chin):
      - MAV only
      --> 1 value per second

Signal preprocessing mirrors the renderer pipeline:
  * EEG / EOG: 0.3--35 Hz bandpass (4th-order Butterworth, zero-phase
    filtfilt) followed by a 50 Hz notch filter (Q = 20).
  * EMG: 10--100 Hz bandpass followed by 50 Hz notch.
  * All channels are resampled to 100 Hz via linear interpolation.

Usage
-----
    from sleepvlm.data.wpt_targets import export_band_power_json

    path = export_band_power_json(sig_dict, stages, output_dir, subject_id)

where *sig_dict* maps channel names to dicts of the form
``{"sample_rate": int, "data": np.ndarray}``.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional, Union

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Target sampling rate after resampling (Hz).
TARGET_SAMPLE_RATE: int = 100

# Duration of one sleep epoch (seconds).
EPOCH_DURATION: int = 30

# Duration of the analysis window inside each epoch (seconds).
WINDOW_DURATION: int = 1

# Butterworth filter order used for all bandpass filters.
BUTTER_ORDER: int = 4

# Power-line notch filter parameters.
NOTCH_FREQ: float = 50.0
NOTCH_Q: float = 20.0

# Bandpass limits for EEG / EOG channels (Hz).
EEG_BP_LOW: float = 0.3
EEG_BP_HIGH: float = 35.0

# Bandpass limits for EMG channels (Hz).
EMG_BP_LOW: float = 10.0
EMG_BP_HIGH: float = 100.0

# Canonical EEG frequency bands (Hz).  The lower bound is inclusive, the
# upper bound is exclusive when we build the frequency mask.
BAND_RANGES: Dict[str, tuple[float, float]] = {
    "delta": (0.3, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
}

# Sleep-stage integer -> label mapping.
STAGE_MAP: Dict[int, str] = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "R"}

# Number of decimal places kept in the output JSON.
OUTPUT_PRECISION: int = 2

# Small constant to prevent log10(0).
_EPS: float = 1e-12


# ---------------------------------------------------------------------------
# Channel-type helper
# ---------------------------------------------------------------------------

def is_emg_channel(ch_name: str) -> bool:
    """Return True if *ch_name* refers to an EMG channel.

    The check is case-insensitive and matches any name containing the
    substring ``'chin'`` or ``'emg'``.
    """
    lower = ch_name.lower()
    return "chin" in lower or "emg" in lower


# ---------------------------------------------------------------------------
# Signal preprocessing
# ---------------------------------------------------------------------------

def _handle_nan(x: np.ndarray) -> np.ndarray:
    """Replace NaN values with linearly interpolated neighbours.

    If the signal is entirely NaN, a zero array of the same shape is
    returned.  The input array is never mutated.
    """
    if not np.isnan(x).any():
        return x
    nans = np.isnan(x)
    valid = ~nans
    if valid.any():
        x = x.copy()
        x[nans] = np.interp(
            np.flatnonzero(nans), np.flatnonzero(valid), x[valid]
        )
    else:
        x = np.zeros_like(x)
    return x


def _design_bandpass(
    sample_rate: float, low: float, high: float
) -> tuple[np.ndarray, np.ndarray]:
    """Design a 4th-order Butterworth bandpass filter.

    The high cutoff is clamped to 0.99 * Nyquist when it would otherwise
    exceed the Nyquist frequency, and the low cutoff is adjusted
    accordingly if needed.
    """
    nyquist = 0.5 * sample_rate
    if high >= nyquist:
        high = nyquist * 0.99
    if low >= high:
        low = high * 0.1
    wn = [low / nyquist, high / nyquist]
    b, a = butter(BUTTER_ORDER, wn, btype="bandpass")
    return b, a


def _design_notch(
    sample_rate: float,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Design a 50 Hz notch filter (quality factor Q = 20).

    Returns ``(None, None)`` when the sampling rate is too low for the
    notch frequency to be representable.
    """
    if NOTCH_FREQ >= 0.5 * sample_rate:
        return None, None
    b, a = iirnotch(NOTCH_FREQ, NOTCH_Q, sample_rate)
    return b, a


def _apply_bandpass(
    x: np.ndarray, sample_rate: float, low: float, high: float
) -> np.ndarray:
    """Zero-phase bandpass filter the signal *x*."""
    x = _handle_nan(x)
    b, a = _design_bandpass(sample_rate, low, high)
    return filtfilt(b, a, x, axis=0, method="gust").astype(np.float32)


def _apply_notch(x: np.ndarray, sample_rate: float) -> np.ndarray:
    """Zero-phase 50 Hz notch filter.  No-op if sample rate is too low."""
    b, a = _design_notch(sample_rate)
    if b is None:
        return x
    return filtfilt(b, a, x, axis=0, method="gust").astype(np.float32)


def _linear_resample(
    x: np.ndarray, src_rate: float, dst_rate: float = TARGET_SAMPLE_RATE
) -> np.ndarray:
    """Resample *x* from *src_rate* to *dst_rate* via linear interpolation."""
    if abs(src_rate - dst_rate) < 1e-12:
        return x.astype(np.float32)
    n_src = x.shape[0]
    n_dst = int(round(n_src * dst_rate / src_rate))
    old_t = np.linspace(0, n_src - 1, n_src, dtype=np.float64)
    new_t = np.linspace(0, n_src - 1, n_dst, dtype=np.float64)
    return np.interp(new_t, old_t, x).astype(np.float32)


def _preprocess_eeg(sig: np.ndarray, sample_rate: float) -> np.ndarray:
    """Preprocess an EEG/EOG channel.

    Pipeline: bandpass 0.3--35 Hz -> 50 Hz notch -> resample to 100 Hz.
    """
    sig = _apply_bandpass(sig, sample_rate, EEG_BP_LOW, EEG_BP_HIGH)
    sig = _apply_notch(sig, sample_rate)
    sig = _linear_resample(sig, sample_rate)
    return sig


def _preprocess_emg(sig: np.ndarray, sample_rate: float) -> np.ndarray:
    """Preprocess an EMG channel.

    Pipeline: bandpass 10--100 Hz -> 50 Hz notch -> resample to 100 Hz.
    The upper bandpass cutoff is clamped inside ``_design_bandpass`` if
    the original sampling rate is too low.
    """
    sig = _apply_bandpass(sig, sample_rate, EMG_BP_LOW, EMG_BP_HIGH)
    sig = _apply_notch(sig, sample_rate)
    sig = _linear_resample(sig, sample_rate)
    return sig


def _preprocess_all(
    sig_dict: Dict[str, dict],
) -> Dict[str, np.ndarray]:
    """Preprocess every channel and reshape into (n_epochs, epoch_len).

    Parameters
    ----------
    sig_dict : dict
        ``{channel_name: {"sample_rate": int, "data": 1-D ndarray}}``.

    Returns
    -------
    dict
        ``{channel_name: 2-D ndarray of shape (n_epochs, epoch_len)}``.
        Channels with insufficient data are silently dropped.
    """
    epoch_len = EPOCH_DURATION * TARGET_SAMPLE_RATE  # 3000 samples

    processed: Dict[str, np.ndarray] = {}
    for ch_name, ch_info in sig_dict.items():
        raw = ch_info["data"].copy()
        sr = ch_info["sample_rate"]

        if is_emg_channel(ch_name):
            sig = _preprocess_emg(raw, sr)
        else:
            sig = _preprocess_eeg(raw, sr)

        n_epochs = len(sig) // epoch_len
        if n_epochs == 0:
            continue
        processed[ch_name] = sig[: n_epochs * epoch_len].reshape(
            n_epochs, epoch_len
        )

    return processed


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _welch_psd(
    window_data: np.ndarray, sample_rate: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute PSD via Welch's method for a 1-second window.

    ``nperseg`` equals the full window length; ``noverlap`` is 50 %.
    """
    nperseg = len(window_data)
    noverlap = nperseg // 2
    freqs, psd = welch(
        window_data, fs=sample_rate, nperseg=nperseg, noverlap=noverlap,
        scaling="density",
    )
    return freqs, psd


def _integrate_bands(
    freqs: np.ndarray, psd: np.ndarray
) -> Dict[str, float]:
    """Integrate PSD over the four canonical bands and return dB values.

    Power is computed via trapezoidal integration, then converted:
    ``power_dB = 10 * log10(power)``.
    """
    result: Dict[str, float] = {}
    for band_name, (fmin, fmax) in BAND_RANGES.items():
        mask = (freqs >= fmin) & (freqs < fmax)
        if mask.any():
            power = float(np.trapz(psd[mask], freqs[mask]))
        else:
            power = _EPS
        result[band_name] = 10.0 * np.log10(power + _EPS)
    return result


def _eeg_epoch_features(
    epoch_data: np.ndarray, sample_rate: int
) -> Dict[str, Dict[str, float]]:
    """Per-second band-power + MAV features for one EEG/EOG epoch.

    Returns
    -------
    dict
        ``{"1": {"delta": ..., "theta": ..., "alpha": ..., "beta": ...,
        "mav": ...}, "2": {...}, ...}`` with 1-based string keys for the
        30 one-second windows.
    """
    win_len = WINDOW_DURATION * sample_rate
    n_windows = len(epoch_data) // win_len
    prec = OUTPUT_PRECISION

    features: Dict[str, Dict[str, float]] = {}
    for i in range(n_windows):
        seg = epoch_data[i * win_len : (i + 1) * win_len]

        # Spectral band powers (dB).
        freqs, psd = _welch_psd(seg, sample_rate)
        bp = _integrate_bands(freqs, psd)

        # Mean Absolute Value (microvolts, assuming input is in uV).
        mav = float(np.mean(np.abs(seg)))

        entry = {k: round(float(v), prec) for k, v in bp.items()}
        entry["mav"] = round(mav, prec)
        features[str(i + 1)] = entry

    return features


def _emg_epoch_features(
    epoch_data: np.ndarray, sample_rate: int
) -> Dict[str, Dict[str, float]]:
    """Per-second MAV feature for one EMG epoch.

    Returns
    -------
    dict
        ``{"1": {"mav": ...}, "2": {"mav": ...}, ...}``
    """
    win_len = WINDOW_DURATION * sample_rate
    n_windows = len(epoch_data) // win_len
    prec = OUTPUT_PRECISION

    features: Dict[str, Dict[str, float]] = {}
    for i in range(n_windows):
        seg = epoch_data[i * win_len : (i + 1) * win_len]
        mav = float(np.mean(np.abs(seg)))
        features[str(i + 1)] = {"mav": round(mav, prec)}

    return features


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def export_band_power_json(
    sig_dict: Dict[str, dict],
    stages: Optional[Union[np.ndarray, list]],
    output_dir: str,
    subject_id: str,
) -> Optional[str]:
    """Compute per-epoch band-power / MAV features and write a JSON file.

    Parameters
    ----------
    sig_dict : dict
        Raw signal dictionary.  Keys are channel names (e.g. ``"F4"``,
        ``"C4"``, ``"O2"``, ``"LOC"``, ``"ROC"``, ``"Chin"``).  Values
        are dicts with ``"sample_rate"`` (int, Hz) and ``"data"``
        (1-D ``np.ndarray``).
    stages : array-like or None
        Integer sleep-stage labels, one per 30-second epoch.  Use
        ``None`` for unlabeled recordings -- epoch keys will then be
        plain indices (``"0"``, ``"1"``, ...) instead of
        ``"0_W"``, ``"1_N1"``, etc.
    output_dir : str
        Directory where the JSON file will be written (created if it
        does not exist).
    subject_id : str
        Used as the JSON filename stem (``<subject_id>.json``).

    Returns
    -------
    str or None
        Absolute path to the written JSON file, or ``None`` if no
        channels survived preprocessing.
    """
    # ---- preprocess all channels ----
    processed = _preprocess_all(sig_dict)
    if not processed:
        return None

    channel_names = list(processed.keys())
    n_epochs = processed[channel_names[0]].shape[0]

    # Clamp to the shorter of signal / annotation length.
    if stages is not None:
        n_epochs = min(n_epochs, len(stages))

    # ---- build the nested dict ----
    result: Dict[str, dict] = {}
    for epoch_idx in range(n_epochs):
        # Epoch key: "idx_LABEL" when labelled, "idx" when unlabelled.
        if stages is not None:
            stage_int = int(stages[epoch_idx])
            label = STAGE_MAP.get(stage_int, "?")
            epoch_key = f"{epoch_idx}_{label}"
        else:
            epoch_key = str(epoch_idx)

        epoch_features: Dict[str, dict] = {}
        for ch in channel_names:
            epoch_data = processed[ch][epoch_idx]
            if is_emg_channel(ch):
                epoch_features[ch] = _emg_epoch_features(
                    epoch_data, TARGET_SAMPLE_RATE
                )
            else:
                epoch_features[ch] = _eeg_epoch_features(
                    epoch_data, TARGET_SAMPLE_RATE
                )
        result[epoch_key] = epoch_features

    # ---- write JSON ----
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{subject_id}.json")
    with open(output_path, "w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2, ensure_ascii=False)

    return output_path
