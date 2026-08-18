"""
HDF5 Loader — Version-Aware Data Access
=========================================

Provides lazy, memory-safe access to HDF5 files produced by ingest().
Signal data is read from disk only when explicitly requested.

Usage:
    >>> from krembil_kit.io import load
    >>> data = load("recording.h5")
    >>> data.channel_names
    ['Fp1', 'Fz', 'F3', ...]
    >>> data.sampling_rate
    500.0
    >>> chunk = data.get_signals(start=0, stop=5000)
    >>> chunk.shape
    (43, 5000)
"""

import warnings
import h5py
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

from ._schema import SCHEMA_VERSION

_LARGE_FILE_SAMPLES = 50_000_000  # ~100s at 500Hz × 100ch


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
        version = self._file.attrs.get("schema_version", None)
        if version is None:
            raise ValueError(
                f"Not a valid krembil-kit HDF5 file: "
                f"missing schema_version attribute."
            )
        if version != SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported schema version '{version}'. "
                f"This loader supports version '{SCHEMA_VERSION}'."
            )

    def _load_metadata(self):
        f = self._file

        # Channel info
        self._channel_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["channels"]["names"][:]
        ]
        self._sampling_rates = f["channels"]["sampling_rates"][:]
        self._channel_units = [
            u.decode() if isinstance(u, bytes) else u
            for u in f["channels"]["units"][:]
        ]

        # Discontinuous flag
        self._discontinuous = bool(
            f["signals"].attrs.get("discontinuous", False)
        )

        # Signal shape
        if self._discontinuous:
            self._n_segments = int(f["signals"].attrs["n_segments"])
            self._n_samples = None
        else:
            self._n_segments = None
            self._n_samples = f["signals"]["data"].shape[1]

        # Events
        onsets = f["events"]["onsets"][:]
        if len(onsets) > 0:
            descriptions = [
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            ]
            durations = f["events"]["durations"][:]
            self._events = {
                "onsets": list(onsets),
                "durations": list(durations),
                "descriptions": descriptions,
            }
        else:
            self._events = None

        # Metadata attributes
        meta_grp = f["metadata"]
        self._metadata = dict(meta_grp.attrs)

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
        return float(self._sampling_rates[0])

    @property
    def channel_units(self) -> List[str]:
        return self._channel_units

    @property
    def discontinuous(self) -> bool:
        return self._discontinuous

    @property
    def n_samples(self) -> Optional[int]:
        """Total samples for continuous files. None if discontinuous."""
        return self._n_samples

    @property
    def n_segments(self) -> Optional[int]:
        """Number of segments for discontinuous files. None if continuous."""
        return self._n_segments

    @property
    def duration_seconds(self) -> Optional[float]:
        """Recording duration in seconds. None if discontinuous."""
        if self._n_samples is not None:
            return self._n_samples / self.sampling_rate
        return None

    @property
    def events(self) -> Optional[Dict[str, Any]]:
        return self._events

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata

    @property
    def segment_start_times(self) -> Optional[List[float]]:
        """Absolute start time of each segment. None if continuous."""
        if not self._discontinuous:
            return None
        times = []
        for i in range(self._n_segments):
            t = self._file["signals"][f"segment_{i}"].attrs["start_time_seconds"]
            times.append(float(t))
        return times

    # ────────────────────────────────────────────────────────────────
    # Signal access (reads from disk)
    # ────────────────────────────────────────────────────────────────

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

        # Warn on large full-file loads
        n_requested = (stop - start) * ds.shape[0]
        if start == 0 and stop == ds.shape[1] and n_requested > _LARGE_FILE_SAMPLES:
            mb = n_requested * 4 / 1e6
            warnings.warn(
                f"Loading full file into memory ({mb:.0f} MB). "
                f"Consider using start/stop to load a subset.",
                stacklevel=2,
            )

        if channels is not None:
            indices = [self._channel_names.index(ch) for ch in channels]
            return ds[indices, start:stop]

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
            indices = [self._channel_names.index(ch) for ch in channels]
            return ds[indices, :]

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
            info = f"{self._n_samples} samples ({self.duration_seconds:.1f}s)"
        return (
            f"Recording({self._path.name}: "
            f"{self.n_channels}ch @ {self.sampling_rate}Hz, {info})"
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
