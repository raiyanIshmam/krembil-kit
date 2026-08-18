"""
BrainVision (.vhdr / .eeg / .vmrk) Isolated Reader
====================================================

Reads BrainVision format recordings (three associated files):
    - .vhdr  — Header file (channel info, sampling rate, format)
    - .eeg   — Raw binary data
    - .vmrk  — Marker file (event annotations)

MNE-Python handles the tri-file relationship automatically when
given the .vhdr path. This reader also accepts a .eeg path and
resolves the .vhdr sibling.

Zero preprocessing is applied. Data is extracted as-is.

Memory-safe: The Raw object is passed with preload=False so the
schema writer can stream data in chunks.
"""

import numpy as np
import mne
from pathlib import Path
from typing import Dict, Any


def read_brainvision(file_path: str) -> Dict[str, Any]:
    """
    Read a BrainVision recording and return standardized dictionary
    for the schema writer.

    The Raw object is passed with preload=False for memory-safe
    streaming during HDF5 write.

    Parameters
    ----------
    file_path : str
        Path to the .vhdr header file, or the .eeg data file.
        If .eeg is provided, the reader resolves the matching .vhdr.

    Returns
    -------
    dict
        Standardized data dictionary ready for write_hdf5().

    Raises
    ------
    FileNotFoundError
        If the .vhdr file cannot be located.
    """
    file_path = Path(file_path).resolve()

    # Resolve to .vhdr if user passed .eeg or .vmrk
    vhdr_path = _resolve_vhdr(file_path)

    raw = mne.io.read_raw_brainvision(
        str(vhdr_path), preload=False, verbose=False
    )

    channel_names = raw.ch_names
    sfreq = raw.info["sfreq"]
    n_channels = len(channel_names)

    # Channel units — BrainVision typically records in µV
    channel_units = _extract_channel_units(raw)
    sampling_rates = np.full(n_channels, sfreq, dtype=np.float64)

    # Extract markers from .vmrk as events
    events = _extract_annotations(raw)

    # Metadata
    meas_date = raw.info.get("meas_date")
    start_datetime = (
        meas_date.isoformat() if meas_date is not None else None
    )
    subject_info = raw.info.get("subject_info") or {}

    device_info = raw.info.get("device_info") or {}
    metadata = {
        "start_datetime": start_datetime,
        "subject_id": subject_info.get("his_id", None),
        "equipment": device_info.get("type", None),
        "source_format": "BrainVision",
        "source_file": vhdr_path.name,
    }

    return {
        "signals": raw,  # Raw object, streamed during write
        "channel_names": channel_names,
        "channel_units": channel_units,
        "sampling_rates": sampling_rates,
        "events": events,
        "metadata": metadata,
        "discontinuous": False,
        "segment_start_times": None,
    }


# ── Private helpers ─────────────────────────────────────────────────


def _resolve_vhdr(file_path: Path) -> Path:
    """
    Given any of the three BrainVision files (.vhdr, .eeg, .vmrk),
    resolve and return the path to the .vhdr header file.
    """
    if file_path.suffix.lower() == ".vhdr":
        vhdr_path = file_path
    elif file_path.suffix.lower() in (".eeg", ".vmrk"):
        vhdr_path = file_path.with_suffix(".vhdr")
    else:
        raise ValueError(
            f"Unsupported BrainVision extension: {file_path.suffix}. "
            f"Expected .vhdr, .eeg, or .vmrk"
        )

    if not vhdr_path.exists():
        raise FileNotFoundError(
            f"BrainVision header file not found: {vhdr_path}"
        )

    return vhdr_path


def _extract_channel_units(raw) -> list:
    """Extract physical units from MNE Raw channel info."""
    channel_units = []
    for ch_info in raw.info["chs"]:
        unit_code = ch_info.get("unit", 0)
        if unit_code == 107:
            channel_units.append("V")
        elif unit_code == 201:
            channel_units.append("T")
        else:
            channel_units.append("uV")
    return channel_units


def _extract_annotations(raw) -> Dict[str, Any]:
    """
    Convert MNE annotations (parsed from .vmrk) to the standardized
    events dictionary. Returns None if no annotations exist.
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return None

    onsets = list(annotations.onset)
    durations = list(annotations.duration)
    descriptions = list(annotations.description)

    return {
        "onsets": onsets,
        "durations": durations,
        "descriptions": descriptions,
    }
