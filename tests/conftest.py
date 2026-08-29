"""
Test configuration and shared fixtures.

Real data tests are skipped automatically if the test files are not
present on disk. This allows the test suite to run anywhere (CI,
other machines) without failing due to missing large files.

Synthetic data tests always run.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path

# Base directory for test data (relative to repo root)
DATA_DIR = Path(__file__).parent.parent / "data" / "test"


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
    Create a synthetic EDF+C file with known values.

    Returns (file_path, expected_data, channel_names, sfreq).
    The file is written to a temporary directory.
    """
    import pyedflib

    tmp_dir = tempfile.mkdtemp()
    file_path = Path(tmp_dir) / "synthetic.edf"

    n_samples = int(duration_sec * sfreq)
    channel_names = [f"CH{i}" for i in range(n_channels)]

    # Deterministic signal: sine waves with different frequencies
    np.random.seed(42)
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

    return str(file_path), data, channel_names, sfreq


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
             channel_names, sfreq, annotations).
    """
    tmp_dir = tempfile.mkdtemp()
    file_path = Path(tmp_dir) / "synthetic_d.edf"

    channel_names = [f"CH{i}" for i in range(n_channels)]

    # Define segments with gaps
    segment_defs = [
        (0.0, 2.0),    # segment 0: starts at 0s, lasts 2s
        (5.0, 2.0),    # segment 1: starts at 5s (3s gap), lasts 2s
        (10.0, 1.0),   # segment 2: starts at 10s (3s gap), lasts 1s
    ]

    # Generate deterministic data for each segment
    np.random.seed(99)
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
            channel_names, sfreq, annotations)


def _write_edfd_bytes(file_path, segments, start_times,
                      channel_names, sfreq, annotations=None):
    """
    Write a minimal EDF+D file from raw bytes.

    Follows the EDF/EDF+ specification exactly:
    - Fixed header (256 bytes)
    - Signal headers (256 * n_signals bytes)
    - Data records with TAL annotation channel

    Annotations are placed in the TAL of the data record whose
    time range contains the annotation onset.
    """
    n_channels = len(channel_names)
    n_signals = n_channels + 1  # +1 for EDF Annotations signal
    record_duration = 1.0  # 1 second per data record
    samples_per_record = int(sfreq * record_duration)

    # Annotation channel: reserve 256 bytes (128 "samples") per record
    annot_samples_per_record = 128

    # Count total data records across all segments
    total_records = 0
    records_per_segment = []
    for seg in segments:
        n_recs = seg.shape[1] // samples_per_record
        records_per_segment.append(n_recs)
        total_records += n_recs

    # Build a mapping of record index → list of annotations for that record
    annot_by_record = {}
    if annotations:
        record_idx = 0
        for seg_idx, (seg_start, seg) in enumerate(zip(start_times, segments)):
            n_recs = records_per_segment[seg_idx]
            for rec in range(n_recs):
                rec_onset = seg_start + rec * record_duration
                rec_end = rec_onset + record_duration
                for onset, dur, desc in annotations:
                    if rec_onset <= onset < rec_end:
                        annot_by_record.setdefault(record_idx + rec, []).append(
                            (onset, dur, desc)
                        )
            record_idx += n_recs

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
        global_rec = 0

        for seg_idx, seg in enumerate(segments):
            n_recs = records_per_segment[seg_idx]
            seg_start = start_times[seg_idx]

            for rec in range(n_recs):
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
                if global_rec in annot_by_record:
                    for evt_onset, evt_dur, evt_desc in annot_by_record[global_rec]:
                        if evt_dur > 0:
                            tal_content += f"+{evt_onset}\x15{evt_dur}\x14{evt_desc}\x14\x00"
                        else:
                            tal_content += f"+{evt_onset}\x14{evt_desc}\x14\x00"

                tal_bytes = tal_content.encode("ascii")
                padded = tal_bytes + b"\x00" * (annot_byte_size - len(tal_bytes))
                f.write(padded)

                global_rec += 1
