"""
EDF / EDF+C / EDF+D Readers
============================

Isolated readers for the three EDF format variants. Each returns a
standardized dictionary consumed by the schema writer.

Format detection reads the 44-byte reserved field in the EDF header:
    "EDF+C" → continuous
    "EDF+D" → discontinuous
    otherwise → plain EDF

No preprocessing is applied. Data is extracted as-is.

For EDF and EDF+C, the MNE Raw object is passed with preload=False
so the schema writer can stream data to disk in chunks.

For EDF+D, segment boundaries are determined by parsing the TAL
(Time-stamped Annotation List) onset times directly from the file,
per the EDF+ specification (Kemp & Olivan, 2003). Each segment is
read individually from disk.
"""

import re
import struct
import numpy as np
import mne
from pathlib import Path
from typing import Dict, Any, List, Tuple


# ────────────────────────────────────────────────────────────────────
# Public: format detection
# ────────────────────────────────────────────────────────────────────

def detect_edf_subtype(file_path: str) -> str:
    """
    Read the 44-byte reserved field (bytes 192–235) of the EDF
    header to classify the file as 'edf', 'edf+c', or 'edf+d'.
    """
    with open(file_path, "rb") as fh:
        fh.seek(192)
        reserved = fh.read(44).decode("ascii", errors="ignore").strip()

    if "EDF+D" in reserved:
        return "edf+d"
    elif "EDF+C" in reserved:
        return "edf+c"
    return "edf"


# ────────────────────────────────────────────────────────────────────
# Public: readers
# ────────────────────────────────────────────────────────────────────

def read_edf(file_path: str) -> Dict[str, Any]:
    """Read a plain EDF file (no annotations)."""
    file_path = str(Path(file_path).resolve())
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    return {
        "signals": raw,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _uniform_rates(raw),
        "events": None,
        "metadata": _build_metadata(raw, "EDF", file_path),
        "discontinuous": False,
        "segment_start_times": None,
    }


def read_edf_plus_c(file_path: str) -> Dict[str, Any]:
    """Read an EDF+C file (continuous, with annotations)."""
    file_path = str(Path(file_path).resolve())
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    return {
        "signals": raw,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _uniform_rates(raw),
        "events": _annotations_to_events(raw),
        "metadata": _build_metadata(raw, "EDF+C", file_path),
        "discontinuous": False,
        "segment_start_times": None,
    }


def read_edf_plus_d(file_path: str) -> Dict[str, Any]:
    """
    Read an EDF+D file (discontinuous).

    Segment boundaries are identified by parsing the TAL onset times
    from each data record and detecting gaps. Each segment is then
    read from disk individually — memory usage is bounded by the
    largest single segment.
    """
    file_path = str(Path(file_path).resolve())
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    sfreq = raw.info["sfreq"]
    record_duration, n_records = _read_record_params(file_path)
    tal_onsets = _parse_tal_onsets(file_path)
    boundaries = _segment_boundaries(tal_onsets, record_duration, sfreq, n_records)

    segments = []
    segment_start_times = []
    for start_samp, stop_samp, onset_time in boundaries:
        segments.append(raw.get_data(start=start_samp, stop=stop_samp))
        segment_start_times.append(onset_time)

    return {
        "signals": segments,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _uniform_rates(raw),
        "events": _annotations_to_events(raw),
        "metadata": _build_metadata(raw, "EDF+D", file_path),
        "discontinuous": True,
        "segment_start_times": segment_start_times,
    }


# ────────────────────────────────────────────────────────────────────
# Private: TAL parsing and segment detection
# ────────────────────────────────────────────────────────────────────

def _read_record_params(file_path: str) -> Tuple[float, int]:
    """
    Read data record duration and count from the EDF fixed header.

    EDF fixed header layout:
        Offset 236: number of data records (8 bytes, ASCII)
        Offset 244: duration of data record in seconds (8 bytes, ASCII)
        Offset 252: number of signals (4 bytes, ASCII)
    """
    with open(file_path, "rb") as fh:
        fh.seek(236)
        raw_nrecords = fh.read(8).decode("ascii").strip()
        raw_duration = fh.read(8).decode("ascii").strip()

    n_records = int(raw_nrecords)
    duration = float(raw_duration)
    return duration, n_records


def _parse_tal_onsets(file_path: str) -> List[float]:
    """
    Extract the onset time from the first TAL in each data record.

    In EDF+, the last signal in each data record is the annotation
    channel. Each annotation channel starts with a TAL whose onset
    is the absolute time of that data record:

        +<seconds>\x14(\x14)\x00...

    This function reads only the annotation bytes — signal data is
    never loaded into memory.
    """
    with open(file_path, "rb") as fh:
        # Read the number of signals from the fixed header.
        fh.seek(252)
        n_signals = int(fh.read(4).decode("ascii").strip())

        # Read the fixed header size (256 + 256 * n_signals bytes).
        header_size = 256 + 256 * n_signals

        # Read per-signal "number of samples per record" (last field
        # in the signal headers, 8 bytes each, starting at offset
        # 256 + 216*n_signals).
        fh.seek(256 + 216 * n_signals)
        samples_per_record = []
        for _ in range(n_signals):
            samples_per_record.append(
                int(fh.read(8).decode("ascii").strip())
            )

        # The annotation signal is the last one whose label contains
        # "Annotation" (EDF+ convention). Find it by reading labels.
        fh.seek(256)
        labels = []
        for _ in range(n_signals):
            labels.append(fh.read(16).decode("ascii", errors="ignore").strip())

        annot_idx = None
        for i, label in enumerate(labels):
            if "annotation" in label.lower():
                annot_idx = i
        if annot_idx is None:
            return []

        # Compute byte sizes per record.
        # Each sample is 2 bytes (16-bit integer).
        signal_sizes = [s * 2 for s in samples_per_record]
        record_size = sum(signal_sizes)
        annot_offset_in_record = sum(signal_sizes[:annot_idx])
        annot_bytes = signal_sizes[annot_idx]

        # Read record duration and count.
        record_duration, n_records = _read_record_params(file_path)

        # Parse onset from each record's annotation channel.
        onsets = []
        tal_onset_pattern = re.compile(rb"\+?(-?\d+\.?\d*)\x14")

        fh.seek(header_size)
        for _ in range(n_records):
            record_start = fh.tell()
            fh.seek(record_start + annot_offset_in_record)
            annot_data = fh.read(annot_bytes)
            fh.seek(record_start + record_size)

            match = tal_onset_pattern.match(annot_data)
            if match:
                onsets.append(float(match.group(1)))

    return onsets


def _segment_boundaries(
    tal_onsets: List[float],
    record_duration: float,
    sfreq: float,
    n_records: int,
) -> List[Tuple[int, int, float]]:
    """
    Given per-record TAL onsets, identify continuous segments by
    detecting where consecutive onsets jump by more than one record
    duration (indicating a gap).

    Returns (start_sample, stop_sample, onset_time) tuples.
    Sample indices are relative to MNE's concatenated data (gaps
    stripped). onset_time is the absolute TAL time of the segment's
    first record.
    """
    if not tal_onsets:
        return [(0, int(n_records * record_duration * sfreq), 0.0)]

    tolerance = record_duration * 1.5
    samples_per_record = int(record_duration * sfreq)

    segments = []
    seg_start_record = 0

    for i in range(1, len(tal_onsets)):
        expected_onset = tal_onsets[i - 1] + record_duration
        actual_onset = tal_onsets[i]

        if abs(actual_onset - expected_onset) > tolerance:
            # Gap detected between record i-1 and record i.
            n_records_in_seg = i - seg_start_record
            start_samp = seg_start_record * samples_per_record
            stop_samp = start_samp + n_records_in_seg * samples_per_record
            segments.append((start_samp, stop_samp, tal_onsets[seg_start_record]))
            seg_start_record = i

    # Final segment.
    n_records_in_seg = len(tal_onsets) - seg_start_record
    start_samp = seg_start_record * samples_per_record
    stop_samp = start_samp + n_records_in_seg * samples_per_record
    segments.append((start_samp, stop_samp, tal_onsets[seg_start_record]))

    return segments


# ────────────────────────────────────────────────────────────────────
# Private: shared helpers
# ────────────────────────────────────────────────────────────────────

def _channel_units(raw) -> List[str]:
    """Map MNE unit codes to string labels."""
    units = []
    for ch in raw.info["chs"]:
        code = ch.get("unit", 0)
        if code == 107:
            units.append("V")
        elif code == 201:
            units.append("T")
        else:
            units.append("uV")
    return units


def _uniform_rates(raw) -> np.ndarray:
    """Return per-channel sampling rate array (uniform)."""
    n = len(raw.ch_names)
    return np.full(n, raw.info["sfreq"], dtype=np.float64)


def _build_metadata(raw, source_format: str, file_path: str) -> Dict[str, Any]:
    """Assemble the metadata dictionary from MNE info."""
    meas_date = raw.info.get("meas_date")
    subject_info = raw.info.get("subject_info") or {}
    device_info = raw.info.get("device_info") or {}

    return {
        "start_datetime": meas_date.isoformat() if meas_date else None,
        "subject_id": subject_info.get("his_id", None),
        "equipment": device_info.get("type", None),
        "source_format": source_format,
        "source_file": Path(file_path).name,
    }


def _annotations_to_events(raw) -> Dict[str, Any]:
    """Convert MNE annotations to the standardized events dict."""
    annot = raw.annotations
    if annot is None or len(annot) == 0:
        return None

    return {
        "onsets": list(annot.onset),
        "durations": list(annot.duration),
        "descriptions": list(annot.description),
    }
