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

import numpy as np
import mne
from pathlib import Path
from typing import Dict, Any, List, NamedTuple, Tuple


class Segment(NamedTuple):
    """
    One continuous stretch of an EDF+D recording.

    start_sample and stop_sample index MNE's concatenated data, which
    has the gaps already removed, with stop exclusive. onset is the
    wall-clock time of the segment's first record and is the only
    real-time information that survives the gap stripping.
    """

    start_sample: int
    stop_sample: int
    onset: float


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
    _reject_mixed_sampling_rates(file_path)
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
    _reject_mixed_sampling_rates(file_path)
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

    Annotations are parsed directly from TAL bytes (not from MNE)
    because MNE drops annotations outside its concatenated timeline.
    """
    file_path = str(Path(file_path).resolve())
    _reject_mixed_sampling_rates(file_path)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    sfreq = raw.info["sfreq"]
    record_duration, n_records = _read_record_params(file_path)
    tal_onsets, tal_annotations = _parse_tals(file_path)
    boundaries = _segment_boundaries(tal_onsets, record_duration, sfreq, n_records)

    segments = []
    segment_start_times = []
    for start_samp, stop_samp, onset_time in boundaries:
        segments.append(raw.get_data(start=start_samp, stop=stop_samp))
        segment_start_times.append(onset_time)

    # Build events from TAL annotations (excluding empty time-keeping ones)
    if tal_annotations:
        events = {
            "onsets": [a["onset"] for a in tal_annotations],
            "durations": [a["duration"] for a in tal_annotations],
            "descriptions": [a["description"] for a in tal_annotations],
        }
    else:
        events = None

    return {
        "signals": segments,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _uniform_rates(raw),
        "events": events,
        "metadata": _build_metadata(raw, "EDF+D", file_path),
        "discontinuous": True,
        "segment_start_times": segment_start_times,
    }


# ────────────────────────────────────────────────────────────────────
# Private: header inspection
# ────────────────────────────────────────────────────────────────────

def _channel_sampling_rates(file_path: str) -> List[float]:
    """
    Read the sampling rate of each ordinary signal from the EDF header.

    EDF stores a samples-per-record count per signal, so channels may
    run at different rates. Annotation channels are excluded: their
    sample count is a byte budget for text, not a sampling rate.
    """
    with open(file_path, "rb") as fh:
        fh.seek(244)
        record_duration = float(fh.read(8).decode("ascii").strip())
        n_signals = int(fh.read(4).decode("ascii").strip())

        fh.seek(256)
        labels = [
            fh.read(16).decode("ascii", errors="ignore").strip()
            for _ in range(n_signals)
        ]

        fh.seek(256 + 216 * n_signals)
        samples_per_record = [
            int(fh.read(8).decode("ascii").strip()) for _ in range(n_signals)
        ]

    # Files holding only annotations may declare a record duration of 0
    # (EDF+ spec 2.1.2). There are no ordinary signals to compare.
    if record_duration == 0:
        return []

    return [
        samples / record_duration
        for samples, label in zip(samples_per_record, labels)
        if label != "EDF Annotations"
    ]


def _reject_mixed_sampling_rates(file_path: str) -> None:
    """
    Raise if the channels do not share a single sampling rate.

    MNE cannot return such files faithfully. With preload=False it
    upsamples slow channels lazily, so values near chunk boundaries
    depend on how the file is sliced. With preload=True it upsamples
    eagerly, fabricating samples that were never recorded. Refusing is
    preferable to writing data that looks plausible but is wrong.
    See known_issues.txt.
    """
    rates = sorted(set(_channel_sampling_rates(file_path)))
    if len(rates) > 1:
        raise ValueError(
            f"{Path(file_path).name} has mixed per-channel sampling rates "
            f"({rates} Hz). This cannot be ingested losslessly and is not "
            f"yet supported."
        )


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


def _parse_tals(file_path: str):
    """
    Parse all TAL content from every data record's annotation channel.

    Returns
    -------
    record_onsets : list of float
        The time-keeping onset from the first TAL in each record.
    annotations : list of dict
        Each dict has keys 'onset' (float), 'duration' (float),
        'description' (str). Collected from all non-time-keeping
        TALs across all records.
    """
    with open(file_path, "rb") as fh:
        fh.seek(252)
        n_signals = int(fh.read(4).decode("ascii").strip())

        header_size = 256 + 256 * n_signals

        fh.seek(256 + 216 * n_signals)
        samples_per_record = []
        for _ in range(n_signals):
            samples_per_record.append(
                int(fh.read(8).decode("ascii").strip())
            )

        fh.seek(256)
        labels = []
        for _ in range(n_signals):
            labels.append(fh.read(16).decode("ascii", errors="ignore").strip())

        # The label is fixed by EDF+ spec 2.2.2. Spec 2.2.4 puts
        # time-keeping in the FIRST annotation signal only, and further
        # annotation signals need not carry one at all, so reading the
        # wrong signal would yield event times in place of record start
        # times. Refuse rather than guess. See known_issues.txt.
        annot_indices = [
            i for i, label in enumerate(labels) if label == "EDF Annotations"
        ]
        if not annot_indices:
            return [], []
        if len(annot_indices) > 1:
            raise ValueError(
                f"{Path(file_path).name} declares {len(annot_indices)} "
                f"'EDF Annotations' signals (indices {annot_indices}). "
                f"Only one is supported."
            )
        annot_idx = annot_indices[0]

        signal_sizes = [s * 2 for s in samples_per_record]
        record_size = sum(signal_sizes)
        annot_offset_in_record = sum(signal_sizes[:annot_idx])
        annot_bytes = signal_sizes[annot_idx]

        _, n_records = _read_record_params(file_path)

        record_onsets = []
        annotations = []

        # Every data record has identical layout, so a record's position
        # can be computed rather than tracked with tell().
        for record_idx in range(n_records):
            record_start = header_size + record_idx * record_size
            fh.seek(record_start + annot_offset_in_record)
            annot_data = fh.read(annot_bytes)

            # TALs are packed at the front of the block and separated by
            # \x00, with the remaining bytes zero-filled. Dropping the
            # trailing zeros first keeps the split from producing one
            # empty piece per padding byte.
            tals = annot_data.rstrip(b"\x00").split(b"\x00")

            for tal_idx, tal in enumerate(tals):
                # A record whose block holds no TALs at all rstrips to
                # b"" and yields a single empty piece.
                if not tal:
                    continue

                # Parse TAL: +<onset>(\x15<duration>)?\x14(<text>\x14)*
                # Split on \x14 to get [onset_part, text1, text2, ...]
                parts = tal.split(b"\x14")
                if not parts[0]:
                    continue

                onset_part = parts[0]

                # Extract onset time and optional duration
                if b"\x15" in onset_part:
                    time_str, dur_str = onset_part.split(b"\x15", 1)
                    dur = float(dur_str) if dur_str else 0.0
                else:
                    time_str = onset_part
                    dur = 0.0

                # Parse onset time
                try:
                    onset_time = float(time_str)
                except ValueError:
                    continue

                # First TAL in the record is time-keeping
                if tal_idx == 0:
                    record_onsets.append(onset_time)

                # Extract annotation text (parts after onset, skip empties)
                texts = [
                    p.decode("utf-8", errors="replace")
                    for p in parts[1:]
                    if p
                ]

                for text in texts:
                    annotations.append({
                        "onset": onset_time,
                        "duration": dur,
                        "description": text,
                    })

    return record_onsets, annotations


def _segment_boundaries(
    tal_onsets: List[float],
    record_duration: float,
    sfreq: float,
    n_records: int,
) -> List[Segment]:
    """
    Given per-record TAL onsets, identify continuous segments. A record
    that begins more than half a record duration later than the
    previous record ended is treated as starting a new segment.

    Records appear in the file in chronological order (EDF+ spec 2.1.2),
    so comparing each onset against the previous one is enough to find
    every gap.
    """
    if not tal_onsets:
        return [Segment(0, int(n_records * record_duration * sfreq), 0.0)]

    gap_threshold = record_duration * 0.5
    samples_per_record = int(record_duration * sfreq)

    segments = []
    segment_start_record = 0

    for record_idx in range(1, len(tal_onsets)):
        expected_onset = tal_onsets[record_idx - 1] + record_duration
        actual_onset = tal_onsets[record_idx]

        if abs(actual_onset - expected_onset) <= gap_threshold:
            continue

        # The gap sits between record_idx-1 and record_idx, so the
        # segment being built ends at record_idx-1 and the next one
        # begins at record_idx.
        segments.append(Segment(
            start_sample=segment_start_record * samples_per_record,
            stop_sample=record_idx * samples_per_record,
            onset=tal_onsets[segment_start_record],
        ))
        segment_start_record = record_idx

    # No gap follows the last record, so the final segment is closed
    # against the end of the list rather than by a detected gap.
    segments.append(Segment(
        start_sample=segment_start_record * samples_per_record,
        stop_sample=len(tal_onsets) * samples_per_record,
        onset=tal_onsets[segment_start_record],
    ))

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
