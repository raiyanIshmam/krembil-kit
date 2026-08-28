"""
HDF5 Schema Writer — Version 1.0
=================================

Writes extracted recording data into the standardized HDF5 layout.
All readers funnel their output through write_hdf5() so every file
has the same structure regardless of source format.

Schema layout:
    /signals    voltage arrays (continuous: one dataset; discontinuous:
                one dataset per segment)
    /channels   names, units, sampling rates
    /events     annotation onsets, durations, descriptions
    /metadata   recording info (start time, source format, subject id)
    root attr   schema_version

Continuous signals are streamed from the MNE Raw object in fixed-size
chunks so the full recording is never held in memory at once.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any


SCHEMA_VERSION = "1.0"

# Signals are written this many seconds at a time. At 500 Hz with
# 30 channels this is ~3.4 MB per chunk.
_CHUNK_DURATION_SEC = 60.0

_COMPRESSION = "gzip"
_COMPRESSION_LEVEL = 4


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
    Write recording data into a Version 1.0 HDF5 file.

    Parameters
    ----------
    output_path : str
        Destination path for the .h5 file.
    signals : mne.io.Raw or list of np.ndarray
        For continuous recordings, an MNE Raw object (streamed to disk
        in chunks). For discontinuous recordings, a list of 2D arrays,
        one per segment.
    channel_names : list of str
        Ordered channel labels.
    channel_units : list of str
        Physical unit for each channel (e.g., 'uV').
    sampling_rates : np.ndarray
        Sampling rate per channel (1D, length n_channels).
    events : dict, optional
        Keys 'onsets', 'durations', 'descriptions'. None if no events.
    metadata : dict, optional
        Recording info written as attributes on /metadata.
    discontinuous : bool
        True if signals is a list of segments (EDF+D).
    segment_start_times : list of float, optional
        Absolute start time (seconds) of each segment. Required when
        discontinuous is True.

    Returns
    -------
    Path
        Path to the written HDF5 file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5file:
        h5file.attrs["schema_version"] = SCHEMA_VERSION

        signals_group = h5file.create_group("signals")
        if discontinuous:
            _write_discontinuous_signals(signals_group, signals, segment_start_times)
        else:
            _write_continuous_signals(signals_group, signals, float(sampling_rates[0]))

        _write_channels(h5file, channel_names, channel_units, sampling_rates)
        _write_events(h5file, events)
        _write_metadata(h5file, metadata)

    return output_path


# ── Signal writers ──────────────────────────────────────────────────

def _write_continuous_signals(group, raw, sfreq):
    """
    Stream a continuous recording from an MNE Raw object into a single
    dataset, reading and writing one chunk at a time.
    """
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    chunk_samples = int(_CHUNK_DURATION_SEC * sfreq)

    dataset = group.create_dataset(
        "data",
        shape=(n_channels, n_samples),
        dtype=np.float32,
        compression=_COMPRESSION,
        compression_opts=_COMPRESSION_LEVEL,
        chunks=(n_channels, min(chunk_samples, n_samples)),
    )

    start = 0
    while start < n_samples:
        stop = min(start + chunk_samples, n_samples)
        dataset[:, start:stop] = raw.get_data(start=start, stop=stop).astype(np.float32)
        start = stop

    group.attrs["discontinuous"] = False


def _write_discontinuous_signals(group, segments, segment_start_times):
    """
    Write each segment of a discontinuous recording as its own dataset,
    tagged with its absolute start time.
    """
    if segment_start_times is None:
        raise ValueError(
            "segment_start_times is required for discontinuous recordings."
        )

    for idx, (segment, start_time) in enumerate(zip(segments, segment_start_times)):
        dataset = group.create_dataset(
            f"segment_{idx}",
            data=np.asarray(segment, dtype=np.float32),
            compression=_COMPRESSION,
            compression_opts=_COMPRESSION_LEVEL,
        )
        dataset.attrs["start_time_seconds"] = float(start_time)

    group.attrs["discontinuous"] = True
    group.attrs["n_segments"] = len(segments)


# ── Metadata writers ────────────────────────────────────────────────

def _write_channels(h5file, channel_names, channel_units, sampling_rates):
    group = h5file.create_group("channels")
    group.create_dataset(
        "names", data=np.array(channel_names, dtype=h5py.string_dtype())
    )
    group.create_dataset(
        "units", data=np.array(channel_units, dtype=h5py.string_dtype())
    )
    group.create_dataset(
        "sampling_rates", data=np.asarray(sampling_rates, dtype=np.float64)
    )


def _write_events(h5file, events):
    group = h5file.create_group("events")
    if events is None:
        events = {"onsets": [], "durations": [], "descriptions": []}

    group.create_dataset(
        "onsets", data=np.asarray(events["onsets"], dtype=np.float64)
    )
    group.create_dataset(
        "durations", data=np.asarray(events["durations"], dtype=np.float64)
    )
    group.create_dataset(
        "descriptions",
        data=np.array(events["descriptions"], dtype=h5py.string_dtype()),
    )


def _write_metadata(h5file, metadata):
    group = h5file.create_group("metadata")
    if metadata is None:
        return
    for key, value in metadata.items():
        if value is not None:
            group.attrs[key] = str(value)
