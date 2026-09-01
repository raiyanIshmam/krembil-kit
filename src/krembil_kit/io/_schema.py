"""
HDF5 Schema Writer
==================

Writes extracted recording data into the standardized HDF5 layout.
All readers funnel their output through write_hdf5() so every file
has the same structure regardless of source format.

Schema layout:
    /signals    sample arrays (continuous: one dataset; discontinuous:
                one dataset per segment)
    /channels   names, units, sampling rates
    /events     annotation onsets, durations, descriptions
    /metadata   recording info (start time, source format, subject id)
    root attr   schema_version

Continuous signals are streamed from the MNE Raw object in fixed-size
chunks so the full recording is never held in memory at once.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any

import h5py
import numpy as np


# Written to the root attrs of every file, and checked when one is
# opened. Only the major part is compared, so a file stays readable
# across minor versions.
#
# Increment the MINOR part when adding a group, dataset or attribute.
# Older readers ignore what they do not recognise, and newer ones must
# cope with the field being absent from older files.
#
# Increment the MAJOR part when removing a field, renaming one, or
# changing the meaning or shape of something that already exists. That
# is the only change which makes previously converted files unreadable,
# so treat it as a migration rather than a version bump.
SCHEMA_VERSION = "1.0"

# Signals are written this many seconds at a time, which bounds how much
# is held in memory while writing. At 500 Hz with 30 channels that is
# about 3.6 MB per batch.
#
# The same figure is used as the HDF5 chunk shape, so it also sets read
# granularity: reading any part of a chunk decompresses all of it. The
# two concerns want different sizes — larger is fine for write batching,
# smaller is better for random reads — so this is worth measuring against
# real access patterns rather than assuming one number suits both.
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
    Write recording data into a standardized HDF5 file.

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
        Unit of the stored samples for each channel. Currently 'V' for
        everything the readers produce, since MNE hands back volts.
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
    n_channels = len(channel_names)
    if len(channel_units) != n_channels or len(sampling_rates) != n_channels:
        raise ValueError(
            f"Channel information does not line up: {n_channels} names, "
            f"{len(channel_units)} units, {len(sampling_rates)} sampling "
            f"rates. All three describe the same channels by position, so "
            f"they must be the same length."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(output_path, "w") as h5file:
        h5file.attrs["schema_version"] = SCHEMA_VERSION

        signals_group = h5file.create_group("signals")
        if discontinuous:
            _write_discontinuous_signals(
                signals_group, signals, segment_start_times
            )
        else:
            # One rate describes every channel, because the readers refuse
            # recordings whose channels disagree.
            _write_continuous_signals(
                signals_group, signals, float(sampling_rates[0])
            )

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
        chunk = raw.get_data(start=start, stop=stop)
        dataset[:, start:stop] = chunk.astype(np.float32)
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

    paired = zip(segments, segment_start_times)
    for idx, (segment, start_time) in enumerate(paired):
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
    """
    Write the three parallel channel datasets.

    They are parallel by contract: entry i of each describes the same
    channel as row i of the signal data. A channel is identified by its
    position, not by its label.
    """
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
    """
    Write the event datasets, using empty ones when there are no events.

    The group and all three datasets are always created, so nothing
    downstream has to check whether /events exists before reading it.
    """
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
    """
    Write recording information as attributes on /metadata.

    Values are stored as strings, and a key is written only when the
    source recording supplied a value — so which attributes are present
    varies between files and readers should not assume any given one.
    """
    group = h5file.create_group("metadata")
    if metadata is None:
        return
    for key, value in metadata.items():
        if value is not None:
            group.attrs[key] = str(value)
