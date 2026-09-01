"""
BrainVision (.vhdr / .eeg / .vmrk) Reader
=========================================

A BrainVision recording is three files:

    .vhdr    header — channel names, sampling rate, data format
    .eeg     the raw binary samples
    .vmrk    markers, which become our events

MNE handles the relationship between them when given the .vhdr, and this
reader accepts any of the three extensions and resolves to it.

No preprocessing is applied. The Raw object is passed on with
preload=False so the schema writer can stream it to disk in chunks.

Unlike the EDF reader there is no mixed sampling rate guard here, and
that is deliberate. The .vhdr header declares a single SamplingInterval
for the whole recording, so every channel shares a rate by construction.
The format has no way to express anything else.
"""

from pathlib import Path
from typing import Dict, Any, List, Optional

import mne
import numpy as np


def read_brainvision(file_path: str) -> Dict[str, Any]:
    """
    Read a BrainVision recording and return the standardized dictionary
    for the schema writer.

    Accepts the path to any of the three files — .vhdr, .eeg or .vmrk —
    and resolves to the .vhdr, which is the one MNE needs.

    Raises
    ------
    ValueError
        If the path carries none of the three extensions.
    FileNotFoundError
        If the .vhdr sibling does not exist.
    """
    vhdr_path = _resolve_vhdr(Path(file_path).resolve())

    raw = mne.io.read_raw_brainvision(
        str(vhdr_path), preload=False, verbose=False
    )

    channel_names = raw.ch_names
    channel_units = _extract_channel_units(raw)
    sampling_rates = np.full(
        len(channel_names), raw.info["sfreq"], dtype=np.float64
    )

    # MNE turns the .vmrk markers into annotations.
    events = _extract_annotations(raw)

    meas_date = raw.info.get("meas_date")
    subject_info = raw.info.get("subject_info") or {}
    device_info = raw.info.get("device_info") or {}

    metadata = {
        "start_datetime": meas_date.isoformat() if meas_date else None,
        "subject_id": subject_info.get("his_id"),
        "equipment": device_info.get("type"),
        "source_format": "BrainVision",
        "source_file": vhdr_path.name,
    }

    return {
        "signals": raw,
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


def _extract_channel_units(raw) -> List[str]:
    """
    Map MNE's per-channel unit code to a string.

    107 and 201 are MNE's own FIFF codes for volts and tesla. In practice
    MNE reports volts for every BrainVision channel, so this returns "V"
    throughout and the other two branches never run. The .vhdr header
    states each channel's unit and we ignore it, which is wrong for any
    channel that is not a voltage.
    """
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


def _extract_annotations(raw) -> Optional[Dict[str, Any]]:
    """
    Convert the markers MNE parsed from the .vmrk file into the events
    dictionary the schema writer expects, or None when there are none.
    """
    annotations = raw.annotations
    if annotations is None or len(annotations) == 0:
        return None

    return {
        "onsets": list(annotations.onset),
        "durations": list(annotations.duration),
        "descriptions": list(annotations.description),
    }
