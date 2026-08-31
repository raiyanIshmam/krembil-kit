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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, NamedTuple, Optional


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


@dataclass(frozen=True)
class EdfHeader:
    """
    Everything the readers need from an EDF header, parsed once.

    Vocabulary follows the spec. A SIGNAL is an entry in the header,
    including the 'EDF Annotations' signal. A CHANNEL is an ordinary
    signal, one that carries samples. So the signal_ fields have one
    entry per signal, while the channel_ fields cover only the ordinary
    ones — which is why the two can have different lengths.

    Read straight from the file:
        n_records, record_duration, n_signals
        signal_labels, signal_samples

    Derived:
        header_size       where the first data record begins
        record_size       bytes per data record, identical for every one
        annotation_indices  which signals are 'EDF Annotations'
        channel_rates     Hz per ordinary channel. Empty when the file
                          declares record_duration 0, which EDF+ allows
                          for annotations-only files with no ordinary
                          signals at all.
        channel_samples_per_record
                          the samples-per-record shared by every
                          ordinary channel, or None when they disagree.
                          Since rate = samples / record_duration and
                          record_duration is one value for the file,
                          equal sample counts and equal rates are the
                          same condition — but tested on integers rather
                          than floats.
    """

    n_records: int
    record_duration: float
    n_signals: int

    signal_labels: List[str]
    signal_samples: List[int]

    header_size: int
    record_size: int
    annotation_indices: List[int]
    channel_rates: List[float]
    channel_samples_per_record: Optional[int]


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
    """Read a plain EDF file."""
    return _read_continuous_recording(file_path, "EDF")


def read_edf_plus_c(file_path: str) -> Dict[str, Any]:
    """Read an EDF+C file (continuous, with annotations)."""
    return _read_continuous_recording(file_path, "EDF+C")


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
    header = _read_header(file_path)
    _reject_mixed_sampling_rates(header, Path(file_path).name)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    tal_onsets, tal_annotations = _parse_tals(file_path, header)
    boundaries = _segment_boundaries(tal_onsets, header)

    segments = []
    segment_start_times = []
    for start_samp, stop_samp, onset_time in boundaries:
        segments.append(raw.get_data(start=start_samp, stop=stop_samp))
        segment_start_times.append(onset_time)

    events = _events_dict(
        [a["onset"] for a in tal_annotations],
        [a["duration"] for a in tal_annotations],
        [a["description"] for a in tal_annotations],
    )

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

def _read_header(file_path: str) -> EdfHeader:
    """
    Parse an EDF header into a single object.

    This is the only function that knows where fields sit in the file.
    Everything else works from the returned EdfHeader.

    Fixed header, 256 bytes:

        offset  field                width
        0       version              8
        8       patient             80
        88      recording           80
        168     startdate            8
        176     starttime            8
        184     header bytes         8    declared size; we compute it
        192     reserved            44    carries EDF+C / EDF+D
        236     number of records    8
        244     record duration      8
        252     number of signals    4

    Signal headers follow, 256 bytes per signal, stored FIELD-MAJOR:
    every signal's label, then every signal's transducer, and so on. So
    a field's block begins at 256 plus the summed width of all earlier
    fields times the signal count:

        field        width    block begins at
        label           16    256
        transducer      80
        dimension        8
        phys min         8
        phys max         8
        dig min          8
        dig max          8
        prefilter       80
        samples          8    256 + 216 * n_signals
        reserved        32

    where 216 = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80.

    Samples are always 2-byte integers. Spec 2.1.2 and 2.1.3 leave no
    choice, which is why no header field states the width.
    """
    with open(file_path, "rb") as fh:
        # 236, 244 and 252 are adjacent, and 252 + 4 lands exactly on
        # 256 where the label block begins, so a single seek covers all
        # four reads below.
        fh.seek(236)
        n_records = int(fh.read(8).decode("ascii").strip())
        record_duration = float(fh.read(8).decode("ascii").strip())
        n_signals = int(fh.read(4).decode("ascii").strip())

        signal_labels = [
            fh.read(16).decode("ascii", errors="ignore").strip()
            for _ in range(n_signals)
        ]

        fh.seek(256 + 216 * n_signals)
        signal_samples = [
            int(fh.read(8).decode("ascii").strip()) for _ in range(n_signals)
        ]

    # The label is fixed by spec 2.2.2, so an exact match is correct
    # rather than fragile.
    annotation_indices = [
        i for i, label in enumerate(signal_labels)
        if label == "EDF Annotations"
    ]

    channel_samples = [
        samples
        for samples, label in zip(signal_samples, signal_labels)
        if label != "EDF Annotations"
    ]

    # An annotations-only file may declare a record duration of 0
    # (spec 2.1.2). There are no ordinary channels to rate.
    if record_duration == 0:
        channel_rates = []
    else:
        channel_rates = [s / record_duration for s in channel_samples]

    distinct_samples = set(channel_samples)
    if len(distinct_samples) == 1:
        channel_samples_per_record = distinct_samples.pop()
    else:
        channel_samples_per_record = None

    return EdfHeader(
        n_records=n_records,
        record_duration=record_duration,
        n_signals=n_signals,
        signal_labels=signal_labels,
        signal_samples=signal_samples,
        header_size=256 + 256 * n_signals,
        record_size=sum(s * 2 for s in signal_samples),
        annotation_indices=annotation_indices,
        channel_rates=channel_rates,
        channel_samples_per_record=channel_samples_per_record,
    )


def _reject_mixed_sampling_rates(header: EdfHeader, file_name: str) -> None:
    """
    Raise if the channels do not share a single sampling rate.

    MNE cannot return such files faithfully. With preload=False it
    upsamples slow channels lazily, so values near chunk boundaries
    depend on how the file is sliced. With preload=True it upsamples
    eagerly, fabricating samples that were never recorded. Refusing is
    preferable to writing data that looks plausible but is wrong.
    See known_issues.txt.

    An annotations-only file has no ordinary channels, so channel_rates
    is empty and nothing is rejected.
    """
    rates = sorted(set(header.channel_rates))
    if len(rates) > 1:
        raise ValueError(
            f"{file_name} has mixed per-channel sampling rates "
            f"({rates} Hz). This cannot be ingested losslessly and is not "
            f"yet supported."
        )


# ────────────────────────────────────────────────────────────────────
# Private: TAL parsing and segment detection
# ────────────────────────────────────────────────────────────────────

def _parse_tals(file_path: str, header: EdfHeader):
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
    # Spec 2.2.4 puts time-keeping in the FIRST annotation signal only,
    # and further annotation signals need not carry one at all, so
    # reading the wrong one would yield event times in place of record
    # start times. Refuse rather than guess. See known_issues.txt.
    if not header.annotation_indices:
        return [], []
    if len(header.annotation_indices) > 1:
        raise ValueError(
            f"{Path(file_path).name} has "
            f"{len(header.annotation_indices)} 'EDF Annotations' signals "
            f"(at positions {header.annotation_indices}). Only one is "
            f"supported."
        )
    annot_idx = header.annotation_indices[0]

    signal_sizes = [s * 2 for s in header.signal_samples]
    annot_offset_in_record = sum(signal_sizes[:annot_idx])
    annot_bytes = signal_sizes[annot_idx]

    record_onsets = []
    annotations = []

    with open(file_path, "rb") as fh:
        # Every data record has identical layout, so a record's position
        # can be computed rather than tracked with tell().
        for record_idx in range(header.n_records):
            record_start = (
                header.header_size + record_idx * header.record_size
            )
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
    header: EdfHeader,
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
        raise ValueError(
            "This file does not contain any record start times. Every "
            "EDF+ file must store them in an 'EDF Annotations' signal "
            "(EDF+ specification, section 2.2.1)."
        )

    record_duration = header.record_duration
    gap_threshold = record_duration * 0.5
    samples_per_record = header.channel_samples_per_record

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

def _read_continuous_recording(
    file_path: str,
    source_format: str,
) -> Dict[str, Any]:
    """
    Read a continuous EDF recording — plain EDF or EDF+C.

    There is no discontinuous counterpart to this function. EDF+D is
    handled by read_edf_plus_d on its own, because it has to parse TALs
    and work out segment boundaries and so shares nothing with these two.
    This helper exists only because plain EDF and EDF+C turned out to be
    the same job.

    The Raw object is passed on with preload=False so the schema writer
    can stream it to disk in chunks.
    """
    file_path = str(Path(file_path).resolve())
    header = _read_header(file_path)
    _reject_mixed_sampling_rates(header, Path(file_path).name)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    return {
        "signals": raw,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _uniform_rates(raw),
        "events": _annotations_to_events(raw),
        "metadata": _build_metadata(raw, source_format, file_path),
        "discontinuous": False,
        "segment_start_times": None,
    }


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


def _events_dict(
    onsets,
    durations,
    descriptions,
) -> Optional[Dict[str, Any]]:
    """
    Build the events dictionary the schema writer expects, or None when
    there are no events.

    len() rather than a truth test because MNE hands back numpy arrays,
    and the truth value of an empty array is ambiguous. len() behaves the
    same on arrays and on lists.
    """
    if len(onsets) == 0:
        return None

    return {
        "onsets": list(onsets),
        "durations": list(durations),
        "descriptions": list(descriptions),
    }


def _annotations_to_events(raw) -> Optional[Dict[str, Any]]:
    """Convert MNE annotations to the standardized events dict."""
    annot = raw.annotations
    if annot is None or len(annot) == 0:
        return None

    return {
        "onsets": list(annot.onset),
        "durations": list(annot.duration),
        "descriptions": list(annot.description),
    }
