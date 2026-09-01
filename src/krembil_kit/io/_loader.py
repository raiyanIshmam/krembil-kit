"""
HDF5 Loader — Version-Aware Data Access
=========================================

Provides lazy, memory-safe access to HDF5 files produced by ingest().
Signal data is read from disk only when explicitly requested.

Usage:
    >>> from krembil_kit.io import load
    >>> with load("recording.h5") as recording:
    ...     recording.channel_names
    ...     chunk = recording.get_signals(start=0, stop=5000)
    ['Fp1', 'Fz', 'F3', ...]

The `with` form closes the file when the block exits. Without it the file
is closed once the Recording is garbage collected, which works but is not
deterministic.
"""

import warnings
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from ._schema import SCHEMA_VERSION

# Reads larger than this warn about memory. 50 million float32 samples is
# 200 MB, roughly 17 minutes of 100 channels at 500 Hz.
_LARGE_FILE_SAMPLES = 50_000_000


def _major_version(version) -> int:
    """
    The major part of a schema version like "1.0".

    Only the major part is compared when opening a file. Minor versions
    are additive by rule, so a newer file may carry fields an older loader
    does not know about and can ignore. Anything that removes or reshapes
    what already exists increments the major instead, and that is the only
    case where an old file becomes unreadable.
    """
    text = str(version)
    major = text.split(".")[0]
    if not major.isdigit():
        raise ValueError(
            f"Schema version {text!r} is not a version number like '1.0'."
        )
    return int(major)


def _as_str(values) -> List[str]:
    """
    Decode a dataset of strings.

    h5py returns byte strings or str depending on how they were written
    and on the h5py version, so both are handled.
    """
    return [v.decode() if isinstance(v, bytes) else v for v in values]


class Recording:
    """
    Lazy handle to an ingested HDF5 file.

    Metadata, channel info, and events are available immediately.
    Signal data is loaded only via get_signals() or get_segment().
    """

    def __init__(self, path: str):
        self._path = Path(path).resolve()
        if not self._path.exists():
            raise FileNotFoundError(f"File not found: {self._path}")

        self._file = h5py.File(self._path, "r")
        self._validate_version()
        self._load_metadata()

    def _validate_version(self):
        version = self._file.attrs.get("schema_version")
        if version is None:
            raise ValueError(
                f"{self._path.name} is not a krembil-kit file: it has no "
                f"schema_version attribute."
            )

        if _major_version(version) != _major_version(SCHEMA_VERSION):
            raise ValueError(
                f"{self._path.name} uses schema version {version}, and this "
                f"loader reads version {SCHEMA_VERSION}. The major versions "
                f"differ, so the layout is not compatible."
            )

    def _load_metadata(self):
        h5file = self._file

        self._channel_names = _as_str(h5file["channels"]["names"][:])
        self._channel_units = _as_str(h5file["channels"]["units"][:])
        self._sampling_rates = h5file["channels"]["sampling_rates"][:]

        signals = h5file["signals"]
        self._discontinuous = bool(signals.attrs.get("discontinuous", False))

        if self._discontinuous:
            self._n_segments = int(signals.attrs["n_segments"])
            segments = [
                signals[f"segment_{i}"] for i in range(self._n_segments)
            ]
            self._n_samples = sum(seg.shape[1] for seg in segments)
            self._segment_start_times = [
                float(seg.attrs["start_time_seconds"]) for seg in segments
            ]
        else:
            self._n_segments = None
            self._n_samples = signals["data"].shape[1]
            self._segment_start_times = None

        onsets = h5file["events"]["onsets"][:]
        if len(onsets) > 0:
            self._events = {
                "onsets": list(onsets),
                "durations": list(h5file["events"]["durations"][:]),
                "descriptions": _as_str(h5file["events"]["descriptions"][:]),
            }
        else:
            self._events = None

        self._metadata = dict(h5file["metadata"].attrs)

    # ────────────────────────────────────────────────────────────────
    # Properties (instant, no disk I/O)
    # ────────────────────────────────────────────────────────────────

    @property
    def channel_names(self) -> List[str]:
        return self._channel_names

    @property
    def n_channels(self) -> int:
        return len(self._channel_names)

    @property
    def sampling_rate(self) -> float:
        """
        The sampling rate shared by every channel.

        Ingest refuses recordings whose channels disagree, so a single
        value describes all of them.
        """
        return float(self._sampling_rates[0])

    @property
    def channel_units(self) -> List[str]:
        return self._channel_units

    @property
    def discontinuous(self) -> bool:
        return self._discontinuous

    @property
    def n_samples(self) -> int:
        """
        Total samples in the recording.

        For a discontinuous recording this is the sum across segments, so
        it counts samples that were recorded and does not include gaps.
        """
        return self._n_samples

    @property
    def n_segments(self) -> Optional[int]:
        """Number of segments for discontinuous files. None if continuous."""
        return self._n_segments

    @property
    def duration_seconds(self) -> float:
        """
        Total recorded time in seconds.

        Gaps are not counted, so for a discontinuous recording this is
        shorter than the span from the first segment to the last. Use
        segment_start_times if you need that span.
        """
        return self._n_samples / self.sampling_rate

    @property
    def events(self) -> Optional[Dict[str, Any]]:
        return self._events

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def segment_start_times(self) -> Optional[List[float]]:
        """Absolute start time of each segment. None if continuous."""
        return self._segment_start_times

    # ────────────────────────────────────────────────────────────────
    # Signal access (reads from disk)
    # ────────────────────────────────────────────────────────────────

    def _channel_indices(self, channels: List[str]) -> List[int]:
        """
        Positions of the named channels, in the order they were asked for.
        """
        missing = [ch for ch in channels if ch not in self._channel_names]
        if missing:
            raise ValueError(
                f"{self._path.name} does not have these channels: {missing}. "
                f"Use .channel_names to see the {self.n_channels} it has."
            )
        return [self._channel_names.index(ch) for ch in channels]

    def get_signals(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        channels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Read signal data from a continuous recording.

        Parameters
        ----------
        start : int, optional
            First sample index. Defaults to 0.
        stop : int, optional
            Last sample index (exclusive). Defaults to end of file.
        channels : list of str, optional
            Channel names to load. Defaults to all channels.

        Returns
        -------
        np.ndarray
            Shape (n_channels, n_samples).
        """
        if self._discontinuous:
            raise ValueError(
                "This is a discontinuous recording. "
                "Use get_segment() instead."
            )

        ds = self._file["signals"]["data"]

        if start is None:
            start = 0
        if stop is None:
            stop = ds.shape[1]

        n_channels_read = ds.shape[0] if channels is None else len(channels)
        n_requested = (stop - start) * n_channels_read
        if n_requested > _LARGE_FILE_SAMPLES:
            megabytes = n_requested * 4 / 1e6  # signals are stored float32
            warnings.warn(
                f"This read is about {megabytes:.0f} MB. Pass start and stop "
                f"to read a smaller window.",
                stacklevel=2,
            )

        if channels is not None:
            return ds[self._channel_indices(channels), start:stop]

        return ds[:, start:stop]

    def get_segment(
        self,
        index: int,
        channels: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Read signal data from one segment of a discontinuous recording.

        Parameters
        ----------
        index : int
            Segment index (0-based).
        channels : list of str, optional
            Channel names to load. Defaults to all channels.

        Returns
        -------
        np.ndarray
            Shape (n_channels, n_samples_in_segment).
        """
        if not self._discontinuous:
            raise ValueError(
                "This is a continuous recording. "
                "Use get_signals() instead."
            )

        if index < 0 or index >= self._n_segments:
            raise IndexError(
                f"Segment index {index} out of range "
                f"(file has {self._n_segments} segments)."
            )

        ds = self._file["signals"][f"segment_{index}"]

        if channels is not None:
            return ds[self._channel_indices(channels), :]

        return ds[:]

    # ────────────────────────────────────────────────────────────────
    # Lifecycle
    # ────────────────────────────────────────────────────────────────

    def close(self):
        """Close the underlying HDF5 file."""
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __repr__(self):
        if self._discontinuous:
            info = f"{self._n_segments} segments"
        else:
            info = f"{self._n_samples} samples"
        return (
            f"Recording({self._path.name}: "
            f"{self.n_channels}ch @ {self.sampling_rate}Hz, "
            f"{info}, {self.duration_seconds:.1f}s)"
        )


def load(file_path: str) -> Recording:
    """
    Open an ingested HDF5 file for lazy data access.

    Parameters
    ----------
    file_path : str
        Path to the .h5 file produced by ingest().

    Returns
    -------
    Recording
        Lazy handle with immediate access to metadata and
        on-demand signal loading.
    """
    return Recording(file_path)
