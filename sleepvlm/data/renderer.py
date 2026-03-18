#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
renderer.py - PSG Waveform Image Rendering Module for SleepVLM

Renders multi-channel polysomnography (PSG) signals into standardized
448x224 images suitable for training vision-language models on sleep staging.

Each 30-second epoch is rendered as one image. Two output naming modes are
supported:
  - Labeled mode:   {epoch}_{stage}.png   (for MASS-SS1, SS3)
  - Unlabeled mode: {epoch}.png           (for MASS-SS2, SS4, SS5)

Rendering specifications
------------------------
- Image size       : 448 x 224 pixels, 100 DPI
- Background       : black (#000000)
- Channel layout   : F4 (yellow), C4 (green), O2 (red),
                     LOC (cyan), ROC (magenta), Chin (blue)
- Amplitude range  : +/-50 uV for EEG/EOG, +/-40 uV for EMG (not clipped)
- Time grid        : 1 s minor lines, 5 s major lines
- Horizontal lines : separator between each channel band

Signal preprocessing
--------------------
- EEG / EOG : 0.3-35 Hz bandpass (4th-order Butterworth, zero-phase) + 50 Hz notch (Q=20)
- EMG       : 10-100 Hz bandpass (4th-order Butterworth, zero-phase) + 50 Hz notch (Q=20)
- Resample to 100 Hz
- Segment into 30-second epochs

Authors
-------
    Guifeng Deng, Pan Wang, Haiteng Jiang
    Zhejiang University / Wenzhou Medical University

License
-------
    CC BY-NC 4.0
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; safe for headless servers
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal
from scipy.interpolate import interp1d


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Canonical channel display order (top to bottom in the rendered image).
CHANNEL_ORDER = ["F4", "C4", "O2", "LOC", "ROC", "Chin"]

# Per-channel rendering colours (hex).
CHANNEL_COLORS = {
    "F4":   "#FFFF00",   # yellow  - frontal EEG
    "C4":   "#00FF00",   # green   - central EEG
    "O2":   "#FF0000",   # red     - occipital EEG
    "LOC":  "#00FFFF",   # cyan    - left EOG
    "ROC":  "#FF00FF",   # magenta - right EOG
    "Chin": "#0000FF",   # blue    - chin EMG
}

# Fixed amplitude display ranges in micro-volts.
# Signals are NOT clipped; values outside the range simply extend beyond
# the channel band boundaries, matching clinical PSG viewer behaviour.
AMPLITUDE_RANGES = {
    "F4":   (-50.0, 50.0),
    "C4":   (-50.0, 50.0),
    "O2":   (-50.0, 50.0),
    "LOC":  (-50.0, 50.0),
    "ROC":  (-50.0, 50.0),
    "Chin": (-40.0, 40.0),
}

# Image geometry.
IMAGE_WIDTH  = 448    # pixels
IMAGE_HEIGHT = 224    # pixels
DPI          = 100

# Epoch / sampling parameters.
EPOCH_DURATION_S   = 30     # seconds per epoch
DEFAULT_TARGET_RATE = 100   # Hz after resampling

# Grid style.
GRID_COLOR       = "#404040"
GRID_ALPHA_MINOR = 0.6
GRID_ALPHA_MAJOR = 0.8
GRID_LW_MINOR    = 0.5
GRID_LW_MAJOR    = 0.8

# Signal trace style.
TRACE_LINEWIDTH = 0.6

# Filter parameters -- EEG and EOG share the same band.
BANDPASS_EEG_EOG = {"low": 0.3,  "high": 35.0,  "order": 4}
BANDPASS_EMG     = {"low": 10.0, "high": 100.0, "order": 4}
NOTCH_FREQ       = 50.0
NOTCH_Q          = 20

# Sleep-stage integer code -> short string label.
STAGE_LABELS = {
    0: "W",
    1: "N1",
    2: "N2",
    3: "N3",
    4: "R",
}


# ---------------------------------------------------------------------------
# Signal processing helpers
# ---------------------------------------------------------------------------

def _apply_bandpass(data, fs, low, high, order=4):
    """Zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : np.ndarray
        1-D signal array.
    fs : float
        Sampling rate in Hz.
    low, high : float
        Cutoff frequencies in Hz.
    order : int
        Filter order.

    Returns
    -------
    np.ndarray
        Filtered signal (same length as *data*).
    """
    nyq = fs / 2.0
    lo = max(low / nyq, 1e-3)
    hi = min(high / nyq, 0.999)
    if lo >= hi:
        return data
    try:
        b, a = scipy_signal.butter(order, [lo, hi], btype="bandpass")
        return scipy_signal.filtfilt(b, a, data)
    except Exception as exc:
        print(f"Warning: bandpass filter failed ({exc}); returning raw data")
        return data


def _apply_notch(data, fs, freq=NOTCH_FREQ, Q=NOTCH_Q):
    """Zero-phase IIR notch filter for power-line interference.

    Parameters
    ----------
    data : np.ndarray
        1-D signal array.
    fs : float
        Sampling rate in Hz.
    freq : float
        Centre frequency of the notch (default 50 Hz).
    Q : float
        Quality factor (default 20).

    Returns
    -------
    np.ndarray
        Filtered signal.
    """
    try:
        b, a = scipy_signal.iirnotch(freq, Q, fs)
        return scipy_signal.filtfilt(b, a, data)
    except Exception as exc:
        print(f"Warning: notch filter failed ({exc}); returning raw data")
        return data


def _resample(data, fs_in, fs_out):
    """Resample a 1-D signal via linear interpolation.

    Parameters
    ----------
    data : np.ndarray
        Input signal.
    fs_in : float
        Original sampling rate.
    fs_out : float
        Target sampling rate.

    Returns
    -------
    np.ndarray
        Resampled signal.
    """
    if abs(fs_in - fs_out) < 0.1:
        return data
    n_in = len(data)
    n_out = int(round(fs_out / fs_in * n_in))
    x_in = np.linspace(0, n_in - 1, n_in)
    x_out = np.linspace(0, n_in - 1, n_out)
    try:
        return interp1d(x_in, data, kind="linear")(x_out)
    except Exception as exc:
        print(f"Warning: resampling failed ({exc}); returning raw data")
        return data


def _preprocess_channel(data, fs, channel_name, target_rate):
    """Filter, resample, and epoch-segment a single channel.

    Parameters
    ----------
    data : np.ndarray
        Raw 1-D signal in micro-volts.
    fs : float
        Sampling rate (Hz).
    channel_name : str
        One of the keys in CHANNEL_ORDER.
    target_rate : float
        Desired output sampling rate (Hz).

    Returns
    -------
    np.ndarray or None
        2-D array of shape ``(n_epochs, samples_per_epoch)`` or *None*
        when the recording is too short for even a single epoch.
    """
    sig = data.copy()

    # Choose bandpass parameters by channel type.
    if channel_name in ("F4", "C4", "O2", "LOC", "ROC"):
        bp = BANDPASS_EEG_EOG
    elif channel_name == "Chin":
        bp = BANDPASS_EMG
    else:
        raise ValueError(f"Unknown channel name: {channel_name}")

    sig = _apply_bandpass(sig, fs, bp["low"], bp["high"], bp["order"])
    sig = _apply_notch(sig, fs)

    if fs != target_rate:
        sig = _resample(sig, fs, target_rate)

    samples_per_epoch = int(EPOCH_DURATION_S * target_rate)
    n_epochs = len(sig) // samples_per_epoch
    if n_epochs == 0:
        print(f"Warning: channel {channel_name} too short for any epoch")
        return None

    sig = sig[: n_epochs * samples_per_epoch]
    return sig.reshape(n_epochs, samples_per_epoch)


# ---------------------------------------------------------------------------
# Public preprocessing API
# ---------------------------------------------------------------------------

def preprocess_signals(sig_dict, target_rate=DEFAULT_TARGET_RATE):
    """Preprocess all channels: filter, resample, epoch-segment.

    Parameters
    ----------
    sig_dict : dict
        Mapping of channel name to channel info dict::

            {
                "F4":  {"sample_rate": 256, "data": np.ndarray},
                "C4":  {"sample_rate": 256, "data": np.ndarray},
                ...
            }

    target_rate : float, optional
        Target sampling rate after resampling (default 100 Hz).

    Returns
    -------
    dict
        ``{channel_name: np.ndarray}`` where each array has shape
        ``(n_epochs, samples_per_epoch)``.  Channels that fail
        preprocessing are silently omitted.
    """
    processed = {}
    for ch_name, ch_info in sig_dict.items():
        result = _preprocess_channel(
            ch_info["data"],
            ch_info["sample_rate"],
            ch_name,
            target_rate,
        )
        if result is not None:
            processed[ch_name] = result
    return processed


# ---------------------------------------------------------------------------
# Single-epoch rendering
# ---------------------------------------------------------------------------

def _render_single_epoch(epoch_matrix, channel_names, output_path):
    """Render one 30-second epoch to a PNG file.

    Parameters
    ----------
    epoch_matrix : np.ndarray
        Shape ``(samples_per_epoch, n_channels)`` -- columns follow the
        same order as *channel_names*.
    channel_names : list[str]
        Channel names corresponding to columns of *epoch_matrix*.
    output_path : str
        Full filesystem path for the output PNG.

    Returns
    -------
    str
        *output_path* (echoed back for convenience).
    """
    n_samples, n_channels = epoch_matrix.shape

    fig_w = IMAGE_WIDTH / DPI
    fig_h = IMAGE_HEIGHT / DPI
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=DPI)
    fig.patch.set_facecolor("black")

    ax = fig.add_subplot(111)
    ax.set_facecolor("black")

    time = np.linspace(0, EPOCH_DURATION_S, n_samples)
    band_height = 1.0 / n_channels

    # -- draw signal traces --------------------------------------------------
    for ch_idx, ch_name in enumerate(channel_names):
        color = CHANNEL_COLORS.get(ch_name, "#FFFFFF")
        amp_lo, amp_hi = AMPLITUDE_RANGES.get(ch_name, (-50.0, 50.0))
        amp_range = amp_hi - amp_lo
        if amp_range == 0:
            amp_range = 1.0

        trace = epoch_matrix[:, ch_idx]

        # Normalise so that the full amplitude range maps to one band height.
        # Signals are NOT clipped -- values beyond the range simply extend
        # into neighbouring bands, replicating clinical PSG viewer behaviour.
        normalised = (trace - (amp_lo + amp_hi) / 2.0) / amp_range

        # Vertical centre of this channel's band (top-to-bottom ordering).
        centre_y = (n_channels - ch_idx - 0.5) * band_height
        y = centre_y + normalised * band_height

        ax.plot(time, y, color=color, linewidth=TRACE_LINEWIDTH,
                antialiased=True)

    # -- vertical time grid ---------------------------------------------------
    for t in range(1, EPOCH_DURATION_S):
        is_major = (t % 5 == 0)
        ax.axvline(
            x=t,
            color=GRID_COLOR,
            alpha=GRID_ALPHA_MAJOR if is_major else GRID_ALPHA_MINOR,
            linewidth=GRID_LW_MAJOR if is_major else GRID_LW_MINOR,
            linestyle="-",
        )

    # -- horizontal channel separators ----------------------------------------
    for i in range(1, n_channels):
        ax.axhline(
            y=i * band_height,
            color=GRID_COLOR,
            alpha=GRID_ALPHA_MAJOR,
            linewidth=GRID_LW_MAJOR,
            linestyle="-",
        )

    # -- axis housekeeping ----------------------------------------------------
    ax.set_xlim(0, EPOCH_DURATION_S)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(
        output_path,
        facecolor="black",
        bbox_inches="tight",
        pad_inches=0,
        dpi=DPI,
    )
    plt.close(fig)

    return output_path


# ---------------------------------------------------------------------------
# Main public rendering API
# ---------------------------------------------------------------------------

def render_psg_from_dict(sig_dict, stages, output_dir, subject_id):
    """Preprocess and render every epoch of a PSG recording to PNG images.

    Supports two naming modes selected automatically by the value of
    *stages*:

    * **Labeled** (``stages`` is array-like): files are named
      ``{epoch}_{stage}.png`` (e.g. ``42_N2.png``).  Suitable for
      MASS-SS1, SS3 which provide per-30s sleep-stage annotations.
    * **Unlabeled** (``stages is None``): files are named
      ``{epoch}.png`` (e.g. ``42.png``).  Suitable for MASS-SS2, SS4,
      SS5 which lack 30-second stage labels.

    Parameters
    ----------
    sig_dict : dict
        Channel data, same format accepted by :func:`preprocess_signals`::

            {"F4": {"sample_rate": 256, "data": np.ndarray}, ...}

    stages : array-like or None
        Integer sleep-stage codes per epoch (0=W, 1=N1, 2=N2, 3=N3, 4=R).
        Pass ``None`` for recordings without per-epoch annotations.
    output_dir : str
        Root output directory.  A subdirectory named *subject_id* is
        created automatically.
    subject_id : str
        Subject / recording identifier used as the subdirectory name.

    Returns
    -------
    list[str]
        Absolute paths of all successfully rendered PNG files.

    Examples
    --------
    Labeled mode (MASS-SS1 / SS3)::

        paths = render_psg_from_dict(sig_dict, stages, "/out", "01-03-0001")

    Unlabeled mode (MASS-SS2 / SS4 / SS5)::

        paths = render_psg_from_dict(sig_dict, None, "/out", "01-02-0001")
    """
    labeled = stages is not None

    # ---- preprocess --------------------------------------------------------
    processed = preprocess_signals(sig_dict, DEFAULT_TARGET_RATE)
    if not processed:
        print(f"[Skip] {subject_id}: no channels survived preprocessing")
        return []

    # Determine available channels in canonical order.
    channel_names = [ch for ch in CHANNEL_ORDER if ch in processed]
    n_channels = len(channel_names)
    if n_channels == 0:
        print(f"[Skip] {subject_id}: none of the expected channels found")
        return []

    # Number of epochs (must agree across channels).
    n_epochs = processed[channel_names[0]].shape[0]
    samples_per_epoch = processed[channel_names[0]].shape[1]

    # When labels are provided, constrain to the shorter of (signal, labels).
    if labeled:
        if len(stages) < n_epochs:
            print(
                f"Warning: {subject_id}: fewer annotations ({len(stages)}) "
                f"than signal epochs ({n_epochs}); truncating to annotations"
            )
            n_epochs = len(stages)

    # Stack channels into a single (n_epochs, samples, channels) tensor.
    block = np.zeros((n_epochs, samples_per_epoch, n_channels))
    for ch_idx, ch_name in enumerate(channel_names):
        block[:, :, ch_idx] = processed[ch_name][:n_epochs]

    # ---- render each epoch -------------------------------------------------
    subject_dir = os.path.join(output_dir, subject_id)
    os.makedirs(subject_dir, exist_ok=True)

    rendered_paths = []
    for epoch_idx in range(n_epochs):
        # Build output filename.
        if labeled:
            stage_code = int(stages[epoch_idx])
            stage_str = STAGE_LABELS.get(stage_code, "?")
            filename = f"{epoch_idx}_{stage_str}.png"
        else:
            filename = f"{epoch_idx}.png"

        out_path = os.path.join(subject_dir, filename)

        try:
            _render_single_epoch(block[epoch_idx], channel_names, out_path)
            rendered_paths.append(out_path)
        except Exception as exc:
            print(f"Error rendering epoch {epoch_idx} of {subject_id}: {exc}")
            continue

    return rendered_paths


# ---------------------------------------------------------------------------
# CLI convenience
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("SleepVLM PSG Renderer")
    print("=" * 50)
    print()
    print("Import this module and call render_psg_from_dict().")
    print()
    print("Labeled mode (MASS-SS1, SS3):")
    print("  from sleepvlm.data.renderer import render_psg_from_dict")
    print('  paths = render_psg_from_dict(sig_dict, stages, "output", "subj01")')
    print()
    print("Unlabeled mode (MASS-SS2, SS4, SS5):")
    print("  from sleepvlm.data.renderer import render_psg_from_dict")
    print('  paths = render_psg_from_dict(sig_dict, None, "output", "subj01")')
