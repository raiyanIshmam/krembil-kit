"""
Test configuration and shared fixtures.

Real data tests are skipped automatically if the test files are not
present on disk. This allows the test suite to run anywhere (CI,
other machines) without failing due to missing large files.

Synthetic data tests always run.
"""

import pytest
import numpy as np
import shutil
import tempfile
from pathlib import Path

# Base directory for test data (relative to repo root)
DATA_DIR = Path(__file__).parent.parent / "data" / "test"

# The file generators below are plain functions rather than fixtures,
# because one test may need several files and because some generators
# call others. That puts their temporary directories outside pytest's
# tmp_path teardown, so they are collected here and removed when the
# session ends.
_TEMP_DIRS = []


def _temp_dir() -> Path:
    path = Path(tempfile.mkdtemp())
    _TEMP_DIRS.append(path)
    return path


@pytest.fixture(scope="session", autouse=True)
def _remove_temp_dirs():
    yield
    for path in _TEMP_DIRS:
        # A directory left behind by a still-open file handle is not
        # worth failing an otherwise green run over.
        shutil.rmtree(path, ignore_errors=True)


# ────────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────────

def require_file(path):
    """Skip test if file does not exist."""
    if not Path(path).exists():
        pytest.skip(f"Test file not found: {path}")


def require_dir(path):
    """Skip test if directory does not exist."""
    if not Path(path).is_dir():
        pytest.skip(f"Test directory not found: {path}")


# ────────────────────────────────────────────────────────────────────
# Synthetic EDF generation
# ────────────────────────────────────────────────────────────────────

def create_synthetic_edf(n_channels=3, duration_sec=5, sfreq=256,
                         annotations=None, edf_plus=True):
    """
    Create a synthetic EDF file holding sine waves.

    edf_plus picks the subtype: True writes EDF+C, False writes plain
    EDF. The two differ only in the reserved header field, and that
    field is what the readers dispatch on.

    Returns (file_path, channel_names, sfreq). The sample values are not
    returned — tests that check them read the file back through MNE, so
    a mistake in this generator cannot cancel out against itself.
    """
    import pyedflib

    file_path = _temp_dir() / "synthetic.edf"

    n_samples = int(duration_sec * sfreq)
    channel_names = [f"CH{i}" for i in range(n_channels)]

    # Deterministic signal: sine waves with different frequencies
    data = np.zeros((n_channels, n_samples))
    for i in range(n_channels):
        t = np.arange(n_samples) / sfreq
        freq = (i + 1) * 2.0  # 2 Hz, 4 Hz, 6 Hz, ...
        data[i] = 100.0 * np.sin(2 * np.pi * freq * t)
        # Add small unique offset so channels are distinguishable
        data[i] += i * 10.0

    # Write using pyedflib
    # file_type: 0 = plain EDF, 1 = EDF+C
    file_type = 1 if edf_plus else 0
    writer = pyedflib.EdfWriter(str(file_path), n_channels, file_type=file_type)

    for i in range(n_channels):
        writer.setSignalHeader(i, {
            "label": channel_names[i],
            "dimension": "uV",
            "sample_frequency": sfreq,
            "physical_min": -500.0,
            "physical_max": 500.0,
            "digital_min": -32768,
            "digital_max": 32767,
            "transducer": "",
            "prefilter": "",
        })

    if annotations:
        for onset, duration, desc in annotations:
            writer.writeAnnotation(onset, duration, desc)

    writer.writeSamples(data)
    writer.close()

    return str(file_path), channel_names, sfreq


def create_edfd_without_annotation_channel():
    """
    Create a file marked EDF+D that has no 'EDF Annotations' signal.

    Such a file is malformed: EDF+ requires the signal because it is
    where each data record's start time is stored, and without those a
    discontinuous recording cannot be split into segments.

    Built by writing a plain EDF and overwriting the reserved field,
    since that field is the only thing distinguishing the two.

    Returns the file path.
    """
    source, _, _ = create_synthetic_edf(edf_plus=False)

    target = _temp_dir() / "edfd_without_annotations.edf"

    original = Path(source).read_bytes()
    target.write_bytes(
        original[:192] + b"EDF+D".ljust(44) + original[236:]
    )
    return str(target)


def create_edfd_with_two_annotation_channels():
    """
    Create an EDF+D file declaring two 'EDF Annotations' signals.

    EDF+ permits more than one, but only the first carries the record
    start times (EDF+ specification, section 2.2.4), and the others need
    not carry them at all. Reading the wrong one would give event times
    where record start times belong.

    Built by relabelling an ordinary channel in a normal synthetic file.
    Only the 16-byte label changes, so the signal count and record size
    stay as they were and no other part of the file needs adjusting. The
    relabelled channel still holds ordinary samples rather than text,
    which does not matter because the file is rejected on its header.

    Returns the file path.
    """
    source, _, _, _, _ = create_synthetic_edfd()

    target = _temp_dir() / "edfd_two_annotation_channels.edf"

    patched = bytearray(Path(source).read_bytes())
    patched[256:272] = b"EDF Annotations".ljust(16)
    target.write_bytes(bytes(patched))
    return str(target)


def create_synthetic_edfd(n_channels=3, sfreq=256, annotations=None):
    """
    Create a synthetic EDF+D file with known gaps.

    Creates 3 segments: 2 seconds, gap, 2 seconds, gap, 1 second.

    Parameters
    ----------
    annotations : list of (float, float, str), optional
        Annotations as (absolute_onset, duration, description).
        These are written into the TAL of the appropriate data record.

    Returns (file_path, segments_data, segment_start_times,
             channel_names, sfreq).
    """
    file_path = _temp_dir() / "synthetic_d.edf"

    channel_names = [f"CH{i}" for i in range(n_channels)]

    # Define segments with gaps
    segment_defs = [
        (0.0, 2.0),    # segment 0: starts at 0s, lasts 2s
        (5.0, 2.0),    # segment 1: starts at 5s (3s gap), lasts 2s
        (10.0, 1.0),   # segment 2: starts at 10s (3s gap), lasts 1s
    ]

    # Generate deterministic data for each segment
    segments_data = []
    segment_start_times = []

    for seg_idx, (start_time, duration) in enumerate(segment_defs):
        n_samples = int(duration * sfreq)
        seg = np.zeros((n_channels, n_samples))
        for ch in range(n_channels):
            t = np.arange(n_samples) / sfreq
            freq = (ch + 1) * 3.0
            seg[ch] = 50.0 * np.sin(2 * np.pi * freq * t)
            seg[ch] += seg_idx * 20.0 + ch * 5.0
        segments_data.append(seg)
        segment_start_times.append(start_time)

    # Write EDF+D manually using raw bytes (no library writes EDF+D)
    _write_edfd_bytes(
        str(file_path), segments_data, segment_start_times,
        channel_names, sfreq, annotations
    )

    return (str(file_path), segments_data, segment_start_times,
            channel_names, sfreq)


def _write_edfd_bytes(file_path, segments, start_times,
                      channel_names, sfreq, annotations=None):
    """
    Write a minimal EDF+D file from raw bytes, because no library
    writes EDF+D.

    Minimal means it carries only what the readers need: a 256-byte
    fixed header, one 256-byte header per signal, then data records each
    holding every signal's samples followed by the annotation channel.
    Patient and recording fields are placeholders, and every channel
    shares one physical range. It is a valid EDF+D file, not a complete
    exercise of the format.

    Each annotation is written into the record whose time range contains
    its onset.
    """
    n_channels = len(channel_names)
    n_signals = n_channels + 1  # +1 for EDF Annotations signal
    record_duration = 1.0  # 1 second per data record
    samples_per_record = int(sfreq * record_duration)

    # Annotation channel: reserve 256 bytes (128 "samples") per record
    annot_samples_per_record = 128

    # EDF holds whole data records only. A segment that is not a whole
    # number of records would lose its tail here without saying so, and
    # the file would then disagree with the segments handed back to the
    # test, so refuse it instead.
    for seg_idx, seg in enumerate(segments):
        if seg.shape[1] % samples_per_record:
            raise ValueError(
                f"Segment {seg_idx} holds {seg.shape[1]} samples, which is "
                f"not a whole number of {samples_per_record}-sample records. "
                f"Pick a segment duration that divides evenly."
            )

    records_per_segment = [
        seg.shape[1] // samples_per_record for seg in segments
    ]
    total_records = sum(records_per_segment)

    # Build a mapping of record index → list of annotations for that record
    annot_by_record = {}
    if annotations:
        for seg_idx, seg_start in enumerate(start_times):
            first_record = sum(records_per_segment[:seg_idx])
            for rec in range(records_per_segment[seg_idx]):
                rec_onset = seg_start + rec * record_duration
                rec_end = rec_onset + record_duration
                for onset, dur, desc in annotations:
                    if rec_onset <= onset < rec_end:
                        annot_by_record.setdefault(
                            first_record + rec, []
                        ).append((onset, dur, desc))

    # Physical range for signals
    phys_min = -500.0
    phys_max = 500.0
    dig_min = -32768
    dig_max = 32767

    # Build fixed header (256 bytes)
    header = ""
    header += "0       "                          # version (8)
    header += "X X X X".ljust(80)                 # patient info (80)
    header += "Startdate X X X X".ljust(80)       # recording info (80)
    header += "01.01.20"                          # start date (8)
    header += "00.00.00"                          # start time (8)
    header += str(256 + 256 * n_signals).ljust(8) # header bytes (8)
    header += "EDF+D".ljust(44)                   # reserved (44)
    header += str(total_records).ljust(8)         # n_records (8)
    header += f"{record_duration:.6f}"[:8].ljust(8)  # duration (8)
    header += str(n_signals).ljust(4)             # n_signals (4)

    # Signal headers: labels (16 each)
    for name in channel_names:
        header += name.ljust(16)
    header += "EDF Annotations".ljust(16)

    # Transducer type (80 each)
    for _ in range(n_signals):
        header += "".ljust(80)

    # Physical dimension (8 each)
    for _ in range(n_channels):
        header += "uV".ljust(8)
    header += "".ljust(8)

    # Physical minimum (8 each)
    for _ in range(n_channels):
        header += str(phys_min).ljust(8)
    header += "-1".ljust(8)

    # Physical maximum (8 each)
    for _ in range(n_channels):
        header += str(phys_max).ljust(8)
    header += "1".ljust(8)

    # Digital minimum (8 each)
    for _ in range(n_channels):
        header += str(dig_min).ljust(8)
    header += "-32768".ljust(8)

    # Digital maximum (8 each)
    for _ in range(n_channels):
        header += str(dig_max).ljust(8)
    header += "32767".ljust(8)

    # Prefiltering (80 each)
    for _ in range(n_signals):
        header += "".ljust(80)

    # Samples per record (8 each)
    for _ in range(n_channels):
        header += str(samples_per_record).ljust(8)
    header += str(annot_samples_per_record).ljust(8)

    # Reserved per signal (32 each)
    for _ in range(n_signals):
        header += "".ljust(32)

    header_bytes = header.encode("ascii")

    # Write file
    with open(file_path, "wb") as f:
        f.write(header_bytes)

        # Write data records
        scale = (phys_max - phys_min) / (dig_max - dig_min)

        for seg_idx, seg in enumerate(segments):
            first_record = sum(records_per_segment[:seg_idx])
            seg_start = start_times[seg_idx]

            for rec in range(records_per_segment[seg_idx]):
                # Write ordinary signal samples
                rec_start = rec * samples_per_record
                rec_end = rec_start + samples_per_record

                for ch in range(n_channels):
                    chunk = seg[ch, rec_start:rec_end]
                    # Convert physical to digital
                    digital = np.round(
                        (chunk - phys_min) / scale + dig_min
                    ).astype(np.int16)
                    f.write(digital.tobytes())

                # Write annotation channel
                onset_time = seg_start + rec * record_duration
                annot_byte_size = annot_samples_per_record * 2

                # Time-keeping TAL
                tal_content = f"+{onset_time}\x14\x14\x00"

                # Event TALs for this record
                record_idx = first_record + rec
                if record_idx in annot_by_record:
                    for evt_onset, evt_dur, evt_desc in annot_by_record[record_idx]:
                        if evt_dur > 0:
                            tal_content += f"+{evt_onset}\x15{evt_dur}\x14{evt_desc}\x14\x00"
                        else:
                            tal_content += f"+{evt_onset}\x14{evt_desc}\x14\x00"

                tal_bytes = tal_content.encode("ascii")
                padded = tal_bytes + b"\x00" * (annot_byte_size - len(tal_bytes))
                f.write(padded)
