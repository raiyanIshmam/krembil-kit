"""
EDF / EDF+C / EDF+D Readers
============================

Reads the three EDF variants and returns a standardized dictionary for
the schema writer. No preprocessing is applied.

Which variant a file is comes from the 44-byte reserved field at offset
192 of the header:

    starts with "EDF+C"    continuous, with annotations
    starts with "EDF+D"    discontinuous
    anything else          plain EDF


VOCABULARY

Two words that are easy to confuse, used here the way the spec uses
them:

    signal     an entry in the EDF header, including the
               'EDF Annotations' signal. A file holding 80 EEG traces
               plus annotations has 81 signals.
    channel    an ordinary signal, one that carries samples. That same
               file has 80 channels, and MNE's raw.ch_names has 80
               entries.

So signal_ names cover all of them and channel_ names only the ordinary
ones, which means the two can have different lengths. Anything pairing
one with the other has to account for that.


FILE LAYOUT

    file      [ fixed header ][ signal headers ][ record 0 ][ record 1 ] ...
    record    [ channel 0 samples ][ channel 1 samples ] ... [ annotations ]

Every record is the same size, fixed once by the header, so a record's
position is arithmetic rather than a search:

    record i begins at   header_size + i * record_size

That is what makes the file seekable, and what lets MNE hand back a
slice of samples without loading the whole recording.

The annotation channel is an ordinary channel as far as the header is
concerned — it gets a label and a samples-per-record count like any
other. Only the reading differs: its bytes hold text rather than
integers, so its sample count is really a text budget.


WHAT MNE DOES AND WHAT WE DO

MNE reads samples. It handles the digital-to-physical scaling and gives
bounded-memory access into a file of any size, which is why the Raw
object is passed on with preload=False for the schema writer to stream.

We read the header ourselves. MNE parses it correctly but then
normalizes it for analysis: per-channel sampling rates collapse to one,
unit strings collapse to one code, annotations are filtered to MNE's own
timeline, and the gaps in a discontinuous recording are stripped out.
Those are all reasonable for an analysis library and wrong for archival,
so every header fact here comes from _read_header, which is the only
function that knows a byte offset.

For EDF+D that goes further: both the segment boundaries and the
annotations are parsed from TAL bytes rather than taken from MNE,
because MNE drops annotations falling outside its gap-stripped timeline.

Spec: https://www.edfplus.info/specs/edfplus.html
      Kemp & Olivan, Clinical Neurophysiology 114 (2003) 1755-1761
"""

import numpy as np
import mne
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, NamedTuple, Optional, Tuple


# A file opens with the fixed header, then one header block per signal.
# Both are 256 bytes, but they are different things, so they are named
# separately rather than written as one number used twice.
_FIXED_HEADER_BYTES = 256
_SIGNAL_HEADER_BYTES = 256

# Samples are always 2-byte integers. Spec sections 2.1.2 and 2.1.3
# leave no choice, which is why no header field states the width.
_BYTES_PER_SAMPLE = 2

# MNE's own unit codes, from the FIFF constants. Not EDF values — EDF
# states each signal's unit as text in the header instead.
_FIFF_UNIT_V = 107
_FIFF_UNIT_T = 201

# Offset of the samples-per-record field within a signal header:
# label 16, transducer 80, dimension 8, phys min 8, phys max 8,
# dig min 8, dig max 8, prefiltering 80. Written as the sum so it can be
# checked against the field table in _read_header without adding up.
_SAMPLES_FIELD_OFFSET = 16 + 80 + 8 + 8 + 8 + 8 + 8 + 80    # 216


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
    Read the reserved field of the EDF header, bytes 192 to 235, and
    classify the file as 'edf', 'edf+c' or 'edf+d'.

    Spec 2.1.1 requires the field to START with the identifier, so a
    prefix test is what the format actually promises. A plain EDF file
    leaves the field blank.
    """
    with open(file_path, "rb") as fh:
        fh.seek(192)
        reserved = fh.read(44).decode("ascii", errors="ignore").strip()

    if reserved.startswith("EDF+D"):
        return "edf+d"
    if reserved.startswith("EDF+C"):
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

    Segment boundaries come from the TAL onset times in each data
    record — see _parse_tals and _segment_boundaries.

    Every segment is held in memory before any of them is written, so
    peak usage is the size of the whole recording rather than of one
    segment. Writing each segment and discarding it as we go would need
    the reader and writer restructured.
    """
    file_path = str(Path(file_path).resolve())
    header = _read_header(file_path)
    _reject_mixed_sampling_rates(header, Path(file_path).name)
    raw = mne.io.read_raw_edf(file_path, preload=False, verbose=False)

    record_onsets, tal_annotations = _parse_tals(file_path, header)
    boundaries = _segment_boundaries(record_onsets, header)

    segment_data = []
    segment_start_times = []
    for segment in boundaries:
        segment_data.append(
            raw.get_data(start=segment.start_sample, stop=segment.stop_sample)
        )
        segment_start_times.append(segment.onset)

    events = _events_dict(
        [a["onset"] for a in tal_annotations],
        [a["duration"] for a in tal_annotations],
        [a["description"] for a in tal_annotations],
    )

    return {
        "signals": segment_data,
        "channel_names": raw.ch_names,
        "channel_units": _channel_units(raw),
        "sampling_rates": _sampling_rates(raw),
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

    Signal headers follow, one _SIGNAL_HEADER_BYTES block per signal,
    stored FIELD-MAJOR: every signal's label, then every signal's
    transducer, and so on.

        field        width
        label           16
        transducer      80
        dimension        8
        phys min         8
        phys max         8
        dig min          8
        dig max          8
        prefilter       80
        samples          8
        reserved        32

    Reading down the widths gives _SAMPLES_FIELD_OFFSET, defined at the
    top of the file. The label block starts at _FIXED_HEADER_BYTES.
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

        # Field-major layout means a field's block sits past every
        # earlier field's block, so its offset within a signal header
        # times the signal count gives where the block starts.
        samples_block = (
            _FIXED_HEADER_BYTES + _SAMPLES_FIELD_OFFSET * n_signals
        )
        fh.seek(samples_block)
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
        header_size=_FIXED_HEADER_BYTES + _SIGNAL_HEADER_BYTES * n_signals,
        record_size=sum(s * _BYTES_PER_SAMPLE for s in signal_samples),
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

def _parse_tals(
    file_path: str,
    header: EdfHeader,
) -> Tuple[List[float], List[Dict[str, Any]]]:
    r"""
    Parse all TAL content from every data record's annotation channel.

    A TAL, or Time-stamped Annotation List, is one timestamp and the
    annotations sharing it:

        +onset [\x15 duration] \x14 text \x14 text \x14 ... \x00

    \x15 is byte 21, \x14 is byte 20 and \x00 is byte 0. All three are
    control bytes, which is why they can never occur inside the UTF-8
    annotation text. The duration is optional, and when it is absent its
    \x15 is absent too.

    TALs are packed at the front of the channel's bytes for a record and
    the rest is zero-filled. The first TAL of every record is the
    time-keeping one (spec 2.2.4): its timestamp says when that record
    starts and its text is empty, which is why two \x14 appear in a row
    there.

    Returns
    -------
    record_onsets : list of float
        The time-keeping onset from the first TAL in each record, so an
        index into this list is a record number.
    annotations : list of dict
        Each dict has keys 'onset' (float), 'duration' (float),
        'description' (str). Collected from all non-time-keeping
        TALs across all records.
    """
    # Spec 2.2.4 puts time-keeping in the FIRST annotation signal only,
    # and further annotation signals need not carry one at all, so
    # reading the wrong one would yield event times in place of record
    # start times. Refuse rather than guess.
    if not header.annotation_indices:
        return [], []
    if len(header.annotation_indices) > 1:
        raise ValueError(
            f"{Path(file_path).name} has "
            f"{len(header.annotation_indices)} 'EDF Annotations' signals "
            f"(at positions {header.annotation_indices}). Only one is "
            f"supported."
        )
    annotation_idx = header.annotation_indices[0]

    signal_sizes = [s * _BYTES_PER_SAMPLE for s in header.signal_samples]
    annotation_offset_in_record = sum(signal_sizes[:annotation_idx])
    annotation_bytes = signal_sizes[annotation_idx]

    record_onsets = []
    annotations = []

    with open(file_path, "rb") as fh:
        # Every data record has identical layout, so a record's position
        # can be computed rather than tracked with tell().
        for record_idx in range(header.n_records):
            record_start = (
                header.header_size + record_idx * header.record_size
            )
            fh.seek(record_start + annotation_offset_in_record)
            annotation_data = fh.read(annotation_bytes)

            # Dropping the trailing zeros first stops the split from
            # producing one empty piece per padding byte.
            tals = annotation_data.rstrip(b"\x00").split(b"\x00")

            for tal_idx, tal in enumerate(tals):
                # A record whose block holds no TALs at all rstrips to
                # b"" and yields a single empty piece.
                if not tal:
                    continue

                parts = tal.split(b"\x14")
                if not parts[0]:
                    continue

                onset_part = parts[0]

                # A TAL may leave out its duration. Recording 0.0 makes
                # "not given" and "zero" indistinguishable downstream.
                # That matches MNE and is right for an instantaneous
                # event, but it is a choice rather than a fact.
                if b"\x15" in onset_part:
                    time_str, dur_str = onset_part.split(b"\x15", 1)
                    dur = float(dur_str) if dur_str else 0.0
                else:
                    time_str = onset_part
                    dur = 0.0

                try:
                    onset_time = float(time_str)
                except ValueError:
                    continue

                # Position carries the meaning: the first TAL of a
                # record holds that record's start time.
                if tal_idx == 0:
                    record_onsets.append(onset_time)

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
    record_onsets: List[float],
    header: EdfHeader,
) -> List[Segment]:
    """
    Given per-record onsets, identify the continuous stretches. A record
    beginning more than half a record duration after the previous one
    ended starts a new segment.

    Records appear in the file in chronological order (spec 2.1.2), so
    comparing each onset against the previous one finds every gap.

    Two timelines are in play, and that is the subtle part. The onsets
    are wall-clock seconds and include the gaps. MNE returns the sample
    data with the gaps already removed, so record N always sits at
    sample N * samples_per_record however much real time preceded it.
    That is why the sample arithmetic below is plain multiplication, and
    why each Segment carries an onset — it is the only wall-clock
    information that survives.
    """
    if not record_onsets:
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

    for record_idx in range(1, len(record_onsets)):
        expected_onset = record_onsets[record_idx - 1] + record_duration
        actual_onset = record_onsets[record_idx]

        if abs(actual_onset - expected_onset) <= gap_threshold:
            continue

        # The gap sits between record_idx-1 and record_idx, so the
        # segment being built ends at record_idx-1 and the next one
        # begins at record_idx.
        segments.append(Segment(
            start_sample=segment_start_record * samples_per_record,
            stop_sample=record_idx * samples_per_record,
            onset=record_onsets[segment_start_record],
        ))
        segment_start_record = record_idx

    # No gap follows the last record, so the final segment is closed
    # against the end of the list rather than by a detected gap.
    segments.append(Segment(
        start_sample=segment_start_record * samples_per_record,
        stop_sample=len(record_onsets) * samples_per_record,
        onset=record_onsets[segment_start_record],
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
        "sampling_rates": _sampling_rates(raw),
        "events": _annotations_to_events(raw),
        "metadata": _build_metadata(raw, source_format, file_path),
        "discontinuous": False,
        "segment_start_times": None,
    }


def _channel_units(raw) -> List[str]:
    """
    Map MNE's per-channel unit code to a string.

    In practice MNE reports volts for every EDF channel, so this returns
    "V" throughout and the other two branches never run. The EDF header
    states each signal's unit as text and we ignore it, which is wrong
    for any channel that is not a voltage.
    """
    units = []
    for ch in raw.info["chs"]:
        code = ch.get("unit", 0)
        if code == _FIFF_UNIT_V:
            units.append("V")
        elif code == _FIFF_UNIT_T:
            units.append("T")
        else:
            units.append("uV")
    return units


def _sampling_rates(raw) -> np.ndarray:
    """
    One sampling rate per channel, which is what the schema stores.

    Every entry holds the same value. That is only correct because
    _reject_mixed_sampling_rates has already refused any file whose
    channels disagree.
    """
    n = len(raw.ch_names)
    return np.full(n, raw.info["sfreq"], dtype=np.float64)


def _build_metadata(raw, source_format: str, file_path: str) -> Dict[str, Any]:
    """Assemble the metadata dictionary from MNE info."""
    meas_date = raw.info.get("meas_date")
    subject_info = raw.info.get("subject_info") or {}
    device_info = raw.info.get("device_info") or {}

    return {
        "start_datetime": meas_date.isoformat() if meas_date else None,
        "subject_id": subject_info.get("his_id"),
        "equipment": device_info.get("type"),
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
    annotations = raw.annotations
    if annotations is None:
        return None

    return _events_dict(
        annotations.onset, annotations.duration, annotations.description
    )
