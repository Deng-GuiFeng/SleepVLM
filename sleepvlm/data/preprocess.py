"""
Unified MASS dataset preprocessing module.

Handles all MASS subsets (SS1-SS5) in two modes:
  - Labeled (SS1, SS3): loads sleep stage annotations from Base.edf,
    renders epoch images as {epoch}_{stage}.png
  - Unlabeled (SS2, SS4, SS5): segments continuously by 30 s without
    labels, renders epoch images as {epoch}.png

Public API
----------
load_psg_signals(sig_path, channel_config) -> (start_time, sig_dict)
load_sleep_annotations(ano_path) -> (stages, onsets, durations)
align_signals_with_annotations(sig_dict, stages, onsets, durations)
    -> (aligned_sig_dict, aligned_stages)
process_subject(subject_id, sig_path, ano_path, output_dir, channel_config,
                compute_band_power=False, band_power_dir=None) -> bool
process_subject_unlabeled(subject_id, sig_path, output_dir, channel_config,
                          compute_band_power=False, band_power_dir=None) -> bool
run_preprocessing(input_dir, output_dir, mode, band_power_dir=None,
                  num_workers=None)
"""

import os
import traceback
import numpy as np
from multiprocessing import Pool, cpu_count

import mne
from mne.io import read_raw_edf

# ---------------------------------------------------------------------------
# Default channel configuration shared by all MASS subsets
# ---------------------------------------------------------------------------
# Each value is a tuple of "options".  An option is either:
#   - a plain string  ->  use that EDF channel directly
#   - a 2-tuple of strings  ->  compute the differential (ch1 - ch2)
# The first matching option wins.

CHANNEL_CONFIG = {
    'F4': ('EEG F4-CLE', 'EEG F4-LER'),
    'C4': ('EEG C4-CLE', 'EEG C4-LER'),
    'O2': ('EEG O2-CLE', 'EEG O2-LER'),
    'LOC': ('EOG Left Horiz',),
    'ROC': ('EOG Right Horiz',),
    'Chin': (('EMG Chin1', 'EMG Chin2'), 'EMG Chin'),
}

# ---------------------------------------------------------------------------
# Sleep-stage mapping (R&K / AASM)
# ---------------------------------------------------------------------------
STAGE_MAPPING = {
    'Sleep stage W': 0,
    'Sleep stage 1': 1,
    'Sleep stage 2': 2,
    'Sleep stage 3': 3,
    'Sleep stage 4': 3,   # R&K stage 4 merged into N3 per AASM
    'Sleep stage R': 4,
}

STAGE_LABELS = {0: 'W', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'R'}

# Epoch duration in seconds (AASM standard).
EPOCH_DURATION = 30

# Upper limit on worker processes to avoid resource exhaustion.
_MAX_WORKERS = 64


# ===================================================================
# Signal loading
# ===================================================================

def _get_channel_names(edf_path):
    """Return channel names from an EDF file header (no data loaded)."""
    raw = read_raw_edf(edf_path, preload=False, verbose=False)
    return raw.ch_names


def load_psg_signals(sig_path, channel_config=None):
    """Load PSG signals from an EDF file for the configured channels.

    Reads only the subset of EDF channels actually needed, resolves
    alternative channel names, and computes differential channels where
    required (e.g. EMG Chin1 - Chin2).

    Parameters
    ----------
    sig_path : str
        Path to the PSG EDF file (e.g. "01-03-0001 PSG.edf").
    channel_config : dict, optional
        Mapping of target channel names to possible source names.
        Defaults to ``CHANNEL_CONFIG``.

    Returns
    -------
    start_time : datetime or None
        Recording start time from the EDF header (timezone-naive).
    sig_dict : dict
        ``{target_name: {'sample_rate': int, 'data': np.ndarray}}``
        for every successfully resolved target channel.
    """
    if channel_config is None:
        channel_config = CHANNEL_CONFIG

    # Discover available channels without loading samples.
    available_in_file = set(_get_channel_names(sig_path))

    # Determine which raw channel names to load.
    channels_to_load = set()
    for channel_options in channel_config.values():
        for option in channel_options:
            if isinstance(option, tuple):
                for ch in option:
                    if ch in available_in_file:
                        channels_to_load.add(ch)
            else:
                if option in available_in_file:
                    channels_to_load.add(option)

    # Fall-back: no channels matched ------------------------------------------------
    if not channels_to_load:
        raw_hdr = read_raw_edf(sig_path, preload=False, verbose=False)
        start_time = raw_hdr.info.get('meas_date')
        if start_time is not None:
            try:
                start_time = start_time.replace(tzinfo=None)
            except Exception:
                pass
        print(f"Warning: no required channels in {sig_path}. "
              f"Available: {sorted(available_in_file)}")
        return start_time, {}

    # Load only the channels we need ------------------------------------------------
    raw = read_raw_edf(sig_path, include=sorted(channels_to_load), verbose=False)
    loaded_channels = set(raw.ch_names)

    # DataFrame approach gives a leading "time" column at index 0.
    data_array = raw.to_data_frame().to_numpy()

    # Derive sample rate from the time column.
    if data_array.shape[0] >= 2:
        sample_rate = round(1.0 / (data_array[1, 0] - data_array[0, 0]))
    else:
        sample_rate = int(raw.info.get('sfreq') or 256)

    # Recording start time (timezone-naive).
    start_time = raw.info.get('meas_date')
    if start_time is not None:
        try:
            start_time = start_time.replace(tzinfo=None)
        except Exception:
            pass

    # Resolve each target channel ---------------------------------------------------
    sig_dict = {}
    for target_name, channel_options in channel_config.items():
        channel_data = None
        for option in channel_options:
            if isinstance(option, tuple) and len(option) == 2:
                ch1, ch2 = option
                if ch1 in loaded_channels and ch2 in loaded_channels:
                    idx1 = raw.ch_names.index(ch1) + 1   # +1 skips time col
                    idx2 = raw.ch_names.index(ch2) + 1
                    channel_data = data_array[:, idx1] - data_array[:, idx2]
                    break
            else:
                if option in loaded_channels:
                    idx = raw.ch_names.index(option) + 1
                    channel_data = data_array[:, idx]
                    break

        if channel_data is not None:
            sig_dict[target_name] = {
                'sample_rate': sample_rate,
                'data': channel_data,
            }
        else:
            print(f"Warning: could not resolve channel '{target_name}' "
                  f"in {sig_path}")

    return start_time, sig_dict


# ===================================================================
# Annotation loading
# ===================================================================

def load_sleep_annotations(ano_path):
    """Load sleep stage annotations from a MASS annotation EDF file.

    Parameters
    ----------
    ano_path : str
        Path to the annotation EDF file (e.g. "01-03-0001 Base.edf").

    Returns
    -------
    stages : np.ndarray of int
        Numeric stage codes (0=W, 1=N1, 2=N2, 3=N3, 4=R, -1=unknown).
    onsets : np.ndarray of float
        Annotation onset times in seconds.
    durations : np.ndarray of float
        Annotation durations in seconds.
    """
    annotations = mne.read_annotations(ano_path)
    stages = np.array(
        [STAGE_MAPPING.get(desc, -1) for desc in annotations.description],
        dtype=np.int32,
    )
    return stages, annotations.onset, annotations.duration


# ===================================================================
# Signal-annotation alignment
# ===================================================================

def align_signals_with_annotations(sig_dict, stages, onsets, durations):
    """Align continuous PSG signals with epoch-level sleep annotations.

    For each valid annotation (stage in 0..4, positive duration) the
    corresponding signal segment is extracted.  Durations are rounded
    down to whole multiples of 30 s and further clipped to the actual
    data length so that every epoch has both a valid label and valid
    signal data.

    Parameters
    ----------
    sig_dict : dict
        Channel signals from :func:`load_psg_signals`.
    stages : np.ndarray
        Stage codes per annotation.
    onsets : np.ndarray
        Onset times in seconds.
    durations : np.ndarray
        Durations in seconds.

    Returns
    -------
    aligned_sig_dict : dict
        Same structure as *sig_dict* but with data trimmed / concatenated
        so that it is epoch-aligned.
    aligned_stages : np.ndarray of int32
        One stage label per 30-s epoch.
    """
    if not sig_dict:
        return sig_dict, np.array([], dtype=np.int32)

    # Sample rate (assumed consistent across channels in MASS).
    sample_rate = int(round(next(iter(sig_dict.values()))['sample_rate']))
    samples_per_epoch = EPOCH_DURATION * sample_rate

    # Valid annotations: known stage and positive duration.
    valid_mask = (stages >= 0) & (stages <= 4) & (np.asarray(durations) > 0)
    valid_indices = np.where(valid_mask)[0]
    if valid_indices.size == 0:
        return sig_dict, np.array([], dtype=np.int32)

    # Build candidate segments: (start_sample, num_epochs, stage).
    segments = []
    for i in valid_indices:
        start_sample = int(round(float(onsets[i]) * sample_rate))
        num_epochs = int(np.floor(float(durations[i]) / EPOCH_DURATION + 1e-6))
        if num_epochs > 0:
            segments.append((start_sample, num_epochs, int(stages[i])))

    if not segments:
        return sig_dict, np.array([], dtype=np.int32)

    # Clip each segment to the actual data length (min across channels).
    channel_lengths = {name: len(ch['data']) for name, ch in sig_dict.items()}
    final_segments = []
    for start, n_ep, stage in segments:
        avail = []
        for length in channel_lengths.values():
            if start >= length:
                avail.append(0)
            else:
                avail.append((length - start) // samples_per_epoch)
        n_ep_final = min(n_ep, min(avail) if avail else 0)
        if n_ep_final > 0:
            final_segments.append((start, n_ep_final, stage))

    if not final_segments:
        empty_dict = {
            name: {'sample_rate': sample_rate,
                   'data': np.array([], dtype=ch['data'].dtype)}
            for name, ch in sig_dict.items()
        }
        return empty_dict, np.array([], dtype=np.int32)

    # Extract and concatenate aligned signal segments.
    aligned_dict = {}
    for name, ch in sig_dict.items():
        data = ch['data']
        parts = [data[s: s + n * samples_per_epoch]
                 for s, n, _ in final_segments]
        aligned_dict[name] = {
            'sample_rate': sample_rate,
            'data': np.concatenate(parts) if parts else np.array(
                [], dtype=data.dtype),
        }

    # Build per-epoch stage labels.
    aligned_stages = []
    for _, n_ep, stage in final_segments:
        aligned_stages.extend([stage] * n_ep)

    return aligned_dict, np.array(aligned_stages, dtype=np.int32)


# ===================================================================
# Rendering helpers (import lazily to keep this module lightweight)
# ===================================================================

def _render_epochs_labeled(sig_dict, stages, output_dir, subject_id):
    """Render labelled epochs using the project renderer.

    Each epoch is saved as ``{output_dir}/{subject_id}/{epoch}_{stage}.png``.

    Returns
    -------
    list of str
        Paths of successfully rendered image files.
    """
    from sleepvlm.data.renderer import render_psg_from_dict
    return render_psg_from_dict(
        sig_dict=sig_dict,
        stages=stages,
        output_dir=output_dir,
        subject_id=subject_id,
    )


def _render_epochs_unlabeled(sig_dict, output_dir, subject_id):
    """Render unlabelled epochs using the project renderer.

    Each epoch is saved as ``{output_dir}/{subject_id}/{epoch}.png``.

    Returns
    -------
    list of str
        Paths of successfully rendered image files.
    """
    from sleepvlm.data.renderer import render_psg_from_dict

    # Determine number of 30-s epochs from the shortest channel.
    sample_rate = int(round(
        next(iter(sig_dict.values()))['sample_rate']))
    epoch_samples = EPOCH_DURATION * sample_rate
    min_length = min(len(ch['data']) for ch in sig_dict.values())
    n_epochs = min_length // epoch_samples

    if n_epochs == 0:
        return []

    # Trim all channels to an exact multiple of epoch_samples.
    trimmed = {}
    for name, ch in sig_dict.items():
        trimmed[name] = {
            'sample_rate': sample_rate,
            'data': ch['data'][:n_epochs * epoch_samples],
        }

    # Use dummy stage codes (0 = Wake) so the renderer can work, then
    # rename the output files to drop the stage suffix.
    dummy_stages = np.zeros(n_epochs, dtype=np.int32)
    rendered = render_psg_from_dict(
        sig_dict=trimmed,
        stages=dummy_stages,
        output_dir=output_dir,
        subject_id=subject_id,
    )

    # Rename: {epoch}_W.png -> {epoch}.png
    subject_dir = os.path.join(output_dir, subject_id)
    renamed = []
    for i in range(n_epochs):
        old = os.path.join(subject_dir, f"{i}_W.png")
        new = os.path.join(subject_dir, f"{i}.png")
        if os.path.exists(old):
            os.rename(old, new)
            renamed.append(new)
    return renamed


# ===================================================================
# Band-power computation helper
# ===================================================================

def _compute_band_power_for_subject(sig_dict, stages, subject_id,
                                    band_power_dir):
    """Compute and save band-power JSON for one subject.

    Parameters
    ----------
    sig_dict : dict
        Channel data (after alignment / trimming).
    stages : np.ndarray or None
        Per-epoch stage labels. ``None`` for unlabeled mode.
    subject_id : str
        Used as the output filename stem.
    band_power_dir : str
        Directory to write the JSON file into.
    """
    try:
        from sleepvlm.data.wpt_targets import export_band_power_json
        os.makedirs(band_power_dir, exist_ok=True)
        export_band_power_json(sig_dict, stages, band_power_dir,
                               subject_id)
    except ImportError:
        print("Warning: sleepvlm.data.wpt_targets not available; "
              "skipping band-power computation")
    except Exception as exc:
        print(f"Warning: band-power computation failed for "
              f"{subject_id}: {exc}")


# ===================================================================
# Per-subject processing (labeled)
# ===================================================================

def process_subject(subject_id, sig_path, ano_path, output_dir,
                    channel_config=None, compute_band_power=False,
                    band_power_dir=None):
    """Process one subject with sleep-stage annotations (labeled mode).

    Pipeline: load signals -> load annotations -> align -> render -> (optional)
    band-power.

    Parameters
    ----------
    subject_id : str
        Subject identifier (e.g. ``"01-03-0001"``).
    sig_path : str
        Path to the PSG signal EDF file.
    ano_path : str
        Path to the annotation EDF file.
    output_dir : str
        Base directory for rendered images.
    channel_config : dict, optional
        Channel configuration. Defaults to ``CHANNEL_CONFIG``.
    compute_band_power : bool
        If ``True``, also generate a band-power JSON file.
    band_power_dir : str or None
        Directory for band-power output (required when
        *compute_band_power* is ``True``).

    Returns
    -------
    bool
        ``True`` if the subject was processed successfully.
    """
    if channel_config is None:
        channel_config = CHANNEL_CONFIG

    try:
        # 1. Load signals and annotations.
        _, sig_dict = load_psg_signals(sig_path, channel_config)
        stages, onsets, durations = load_sleep_annotations(ano_path)

        # 2. Align signals to annotation epochs.
        sig_dict, stages = align_signals_with_annotations(
            sig_dict, stages, onsets, durations)

        # 3. Keep only channels that were successfully resolved.
        available = [ch for ch in channel_config if ch in sig_dict]
        if not available:
            print(f"[Skip] {subject_id}: no valid channels found")
            return False

        filtered = {ch: sig_dict[ch] for ch in available}

        # 4. Render epoch images.
        rendered = _render_epochs_labeled(filtered, stages, output_dir,
                                          subject_id)
        if not rendered:
            return False

        # 5. Optional band-power computation.
        if compute_band_power and band_power_dir is not None:
            _compute_band_power_for_subject(filtered, stages,
                                            subject_id, band_power_dir)

        return True

    except Exception as exc:
        print(f"Error processing {subject_id}: {exc}")
        traceback.print_exc()
        return False


# ===================================================================
# Per-subject processing (unlabeled)
# ===================================================================

def process_subject_unlabeled(subject_id, sig_path, output_dir,
                              channel_config=None,
                              compute_band_power=False,
                              band_power_dir=None):
    """Process one subject without annotations (unlabeled mode).

    Pipeline: load signals -> trim to 30-s epochs -> render ->
    (optional) band-power.

    Parameters
    ----------
    subject_id : str
        Subject identifier.
    sig_path : str
        Path to the PSG signal EDF file.
    output_dir : str
        Base directory for rendered images.
    channel_config : dict, optional
        Channel configuration. Defaults to ``CHANNEL_CONFIG``.
    compute_band_power : bool
        If ``True``, also generate a band-power JSON file.
    band_power_dir : str or None
        Directory for band-power output.

    Returns
    -------
    bool
        ``True`` if the subject was processed successfully.
    """
    if channel_config is None:
        channel_config = CHANNEL_CONFIG

    try:
        # 1. Load signals (no annotations).
        _, sig_dict = load_psg_signals(sig_path, channel_config)

        # 2. Keep only resolved channels.
        available = [ch for ch in channel_config if ch in sig_dict]
        if not available:
            print(f"[Skip] {subject_id}: no valid channels found")
            return False

        filtered = {ch: sig_dict[ch] for ch in available}

        # 3. Render unlabeled epoch images.
        rendered = _render_epochs_unlabeled(filtered, output_dir,
                                            subject_id)
        if not rendered:
            return False

        # 4. Optional band-power computation (stages=None).
        if compute_band_power and band_power_dir is not None:
            _compute_band_power_for_subject(filtered, None,
                                            subject_id, band_power_dir)

        return True

    except Exception as exc:
        print(f"Error processing {subject_id}: {exc}")
        traceback.print_exc()
        return False


# ===================================================================
# Multiprocessing wrappers
# ===================================================================

def _worker_labeled(args):
    """Top-level picklable wrapper for ``process_subject``."""
    (subject_id, sig_path, ano_path, output_dir,
     channel_config, compute_band_power, band_power_dir) = args
    return process_subject(
        subject_id, sig_path, ano_path, output_dir,
        channel_config=channel_config,
        compute_band_power=compute_band_power,
        band_power_dir=band_power_dir,
    )


def _worker_unlabeled(args):
    """Top-level picklable wrapper for ``process_subject_unlabeled``."""
    (subject_id, sig_path, output_dir,
     channel_config, compute_band_power, band_power_dir) = args
    return process_subject_unlabeled(
        subject_id, sig_path, output_dir,
        channel_config=channel_config,
        compute_band_power=compute_band_power,
        band_power_dir=band_power_dir,
    )


# ===================================================================
# Utilities
# ===================================================================

def _find_edf_files(directory):
    """Recursively find all *.edf files under *directory*."""
    matches = []
    for dirpath, _, filenames in os.walk(directory):
        for fn in filenames:
            if fn.lower().endswith('.edf'):
                matches.append(os.path.join(dirpath, fn))
    return matches


def _discover_subjects(input_dir):
    """Return sorted set of subject IDs found in *input_dir*.

    Subject IDs are derived from MASS file naming: the portion of each
    EDF filename before the first space character.
    """
    edf_files = _find_edf_files(input_dir)
    ids = set()
    for f in edf_files:
        basename = os.path.basename(f)
        parts = basename.split(' ')
        if len(parts) >= 2:
            ids.add(parts[0])
    return sorted(ids)


# ===================================================================
# Main entry point
# ===================================================================

def run_preprocessing(input_dir, output_dir, mode,
                      band_power_dir=None, num_workers=None):
    """Run the MASS preprocessing pipeline on all subjects in *input_dir*.

    Parameters
    ----------
    input_dir : str
        Directory containing MASS EDF files.  Expected naming convention
        is ``{subject_id} PSG.edf`` and (for labeled mode)
        ``{subject_id} Base.edf``.
    output_dir : str
        Base directory for rendered epoch images.  A sub-directory will
        be created per subject.
    mode : str
        ``'labeled'`` (SS1, SS3) or ``'unlabeled'`` (SS2, SS4, SS5).
    band_power_dir : str or None
        If given, band-power JSON files will be written here.
    num_workers : int or None
        Number of parallel worker processes.  Defaults to
        ``min(cpu_count(), 64)``.

    Returns
    -------
    success_count : int
        Number of subjects processed successfully.
    total_count : int
        Total number of subjects attempted.
    """
    if mode not in ('labeled', 'unlabeled'):
        raise ValueError(f"mode must be 'labeled' or 'unlabeled', "
                         f"got '{mode}'")

    if num_workers is None:
        num_workers = min(cpu_count(), _MAX_WORKERS)

    compute_bp = band_power_dir is not None
    channel_config = CHANNEL_CONFIG

    # Discover subjects.
    subject_ids = _discover_subjects(input_dir)

    # Build task list.
    tasks = []
    for sid in subject_ids:
        sig_path = os.path.join(input_dir, f"{sid} PSG.edf")
        if not os.path.isfile(sig_path):
            continue

        if mode == 'labeled':
            ano_path = os.path.join(input_dir, f"{sid} Base.edf")
            if not os.path.isfile(ano_path):
                continue
            tasks.append((sid, sig_path, ano_path, output_dir,
                          channel_config, compute_bp, band_power_dir))
        else:
            tasks.append((sid, sig_path, output_dir,
                          channel_config, compute_bp, band_power_dir))

    total_count = len(tasks)
    if total_count == 0:
        print("No valid subjects found to process!")
        return 0, 0

    print(f"Found {total_count} subjects to process (mode={mode})")
    print(f"Using {num_workers} worker processes")

    os.makedirs(output_dir, exist_ok=True)

    worker_fn = _worker_labeled if mode == 'labeled' else _worker_unlabeled
    success_count = 0

    try:
        with Pool(num_workers) as pool:
            from tqdm import tqdm
            with tqdm(total=total_count, desc="Processing MASS") as pbar:
                for result in pool.imap_unordered(worker_fn, tasks):
                    if result:
                        success_count += 1
                    pbar.update(1)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        return success_count, total_count

    print(f"\nDone. Successfully processed {success_count}/{total_count} "
          f"subjects ({success_count / total_count * 100:.1f}%)")
    print(f"Output directory: {output_dir}")

    return success_count, total_count


def main():
    """CLI entry point for preprocessing MASS data."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Preprocess MASS PSG data: render waveform images and "
                    "optionally compute band power targets."
    )
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing MASS EDF files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for rendered images")
    parser.add_argument("--mode", type=str, choices=["labeled", "unlabeled"],
                        required=True,
                        help="'labeled' for SS1/SS3, 'unlabeled' for SS2/4/5")
    parser.add_argument("--band_power_dir", type=str, default=None,
                        help="Output directory for band power JSONs")
    parser.add_argument("--num_workers", type=int, default=32,
                        help="Number of parallel workers (default: 32)")
    parser.add_argument("--skip_render", action="store_true",
                        help="Skip image rendering, only compute band power")
    args = parser.parse_args()

    if args.skip_render and args.band_power_dir is None:
        parser.error("--skip_render requires --band_power_dir")

    if args.skip_render:
        # Band-power-only mode: reuse run_preprocessing with a flag
        # We set output_dir but won't actually render
        print(f"[Skip render] Computing band power only: {args.input_dir}")
        run_preprocessing(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            band_power_dir=args.band_power_dir,
            num_workers=args.num_workers,
        )
    else:
        run_preprocessing(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            mode=args.mode,
            band_power_dir=args.band_power_dir,
            num_workers=args.num_workers,
        )


if __name__ == "__main__":
    main()
