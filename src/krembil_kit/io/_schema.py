"""
HDF5 Schema Writer — Version 1.0 Baseline
==========================================

Writes the standardized HDF5 structure defined in the Version 1.0
schema specification. All ingestion readers funnel their extracted
data through this single writer to guarantee uniform output.

Schema Layout (Version 1.0):
    /signals        — Voltage arrays (2D continuous or per-segment)
    /channels       — Channel names, units, sampling rates
    /events         — Annotations / markers
    /metadata       — Global recording information
    root attrs      — schema_version = "1.0"

Memory-safe: For continuous recordings, data is streamed from the
MNE Raw object in chunks to avoid loading the full file into RAM.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


SCHEMA_VERSION = "1.0"

# Default chunk: 60 seconds worth of data per write cycle.
# At 500 Hz × 30 channels × 4 bytes = ~3.4 MB per chunk — trivial.
_CHUNK_DURATION_SEC = 60.0


def write_hdf5(
    output_path: str,
    signals: Any,
    channel_names: List[str],
    channel_units: List[str],
    sampling_rates: np.ndarray,
    events: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    discontinuous: bool = False,
    segment_start_times: Optional[List[float]] = None,
) -> Path:
    """
    Write extracted recording data into a Version 1.0 HDF5 file.

    Supports two modes:
      - **Streaming** (memory-safe): If `signals` is an MNE Raw object,
        data is read and written in chunks without full preload.
      - **Array**: If `signals` is a numpy array or list of arrays,
        it is written directly (legacy path, for small files or
        discontinuous segments already in memory).

    Parameters
    ----------
    output_path : str
        Destination path for the .h5 file.
    signals : mne.io.Raw, np.ndarray, or list of np.ndarray
        - mne.io.Raw: streamed in chunks (memory-safe).
        - np.ndarray: written directly (n_channels, n_samples).
        - list of np.ndarray: one per segment (discontinuous).
    channel_names : list of str
        Ordered channel labels.
    channel_units : list of str
        Physical unit for each channel (e.g., 'uV', 'mV').
    sampling_rates : np.ndarray
        Sampling rate per channel (1D array, length n_channels).
    events : dict, optional
        Dictionary with keys 'onsets', 'durations', 'descriptions'.
    metadata : dict, optional
        Global recording info.
    discontinuous : bool
        If True, signals is a list of segments (EDF+D handling).
    segment_start_times : list of float, optional
        Absolute start time (seconds) for each segment.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as f:
        # ── Root attribute: schema version ──────────────────────────
        f.attrs["schema_version"] = SCHEMA_VERSION

        # ── /signals ────────────────────────────────────────────────
        sig_grp = f.create_group("signals")

        if discontinuous:
            _write_discontinuous_signals(
                sig_grp, signals, segment_start_times
            )
        else:
            _write_continuous_signals(sig_grp, signals, sampling_rates)

        # ── /channels ───────────────────────────────────────────────
        ch_grp = f.create_group("channels")
        ch_grp.create_dataset(
            "names",
            data=np.array(channel_names, dtype=h5py.string_dtype()),
        )
        ch_grp.create_dataset(
            "units",
            data=np.array(channel_units, dtype=h5py.string_dtype()),
        )
        ch_grp.create_dataset(
            "sampling_rates",
            data=np.asarray(sampling_rates, dtype=np.float64),
        )

        # ── /events ────────────────────────────────────────────────
        evt_grp = f.create_group("events")
        if events is not None:
            onsets = events.get("onsets", [])
            durations = events.get("durations", [])
            descriptions = events.get("descriptions", [])

            evt_grp.create_dataset(
                "onsets",
                data=np.asarray(onsets, dtype=np.float64),
            )
            evt_grp.create_dataset(
                "durations",
                data=np.asarray(durations, dtype=np.float64),
            )
            evt_grp.create_dataset(
                "descriptions",
                data=np.array(
                    descriptions, dtype=h5py.string_dtype()
                ),
            )
        else:
            evt_grp.create_dataset("onsets", data=np.array([]))
            evt_grp.create_dataset("durations", data=np.array([]))
            evt_grp.create_dataset(
                "descriptions",
                data=np.array([], dtype=h5py.string_dtype()),
            )

        # ── /metadata ──────────────────────────────────────────────
        meta_grp = f.create_group("metadata")
        if metadata is not None:
            for key, value in metadata.items():
                if value is not None:
                    meta_grp.attrs[key] = str(value)

    return output_path


# ── Internal writers ────────────────────────────────────────────────


def _write_continuous_signals(sig_grp, signals, sampling_rates):
    """
    Write continuous signal data, streaming from MNE Raw if possible.
    """
    # Check if signals is an MNE Raw object (duck-type check)
    if hasattr(signals, "get_data") and hasattr(signals, "n_times"):
        _stream_from_raw(sig_grp, signals, sampling_rates)
    else:
        # Direct array write (small files or pre-loaded data)
        sig_grp.create_dataset(
            "data",
            data=np.asarray(signals, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
    sig_grp.attrs["discontinuous"] = False


def _stream_from_raw(sig_grp, raw, sampling_rates):
    """
    Stream data from an MNE Raw object into HDF5 in fixed-size
    chunks. Peak memory usage = one chunk (~60s of data).
    """
    sfreq = float(sampling_rates[0])
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    chunk_samples = int(_CHUNK_DURATION_SEC * sfreq)

    # Create the dataset with final shape, fill in chunks
    ds = sig_grp.create_dataset(
        "data",
        shape=(n_channels, n_samples),
        dtype=np.float32,
        compression="gzip",
        compression_opts=4,
        chunks=(n_channels, min(chunk_samples, n_samples)),
    )

    # Write in chunks
    start = 0
    while start < n_samples:
        stop = min(start + chunk_samples, n_samples)
        chunk = raw.get_data(start=start, stop=stop)
        ds[:, start:stop] = chunk.astype(np.float32)
        start = stop


def _write_discontinuous_signals(sig_grp, signals, segment_start_times):
    """
    Write discontinuous segments. Each segment is typically small
    enough to write directly (gaps imply shorter arrays).
    """
    if segment_start_times is None:
        raise ValueError(
            "segment_start_times is required for "
            "discontinuous recordings."
        )

    for idx, (seg, t0) in enumerate(
        zip(signals, segment_start_times)
    ):
        # If segment is an MNE Raw object, extract its data
        if hasattr(seg, "get_data"):
            seg = seg.get_data()

        ds = sig_grp.create_dataset(
            f"segment_{idx}",
            data=np.asarray(seg, dtype=np.float32),
            compression="gzip",
            compression_opts=4,
        )
        ds.attrs["start_time_seconds"] = float(t0)

    sig_grp.attrs["discontinuous"] = True
    sig_grp.attrs["n_segments"] = len(signals)
