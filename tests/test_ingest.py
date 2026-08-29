"""
Ingestion tests for all supported EDF format variants and BrainVision.

Each test ingests a file, opens the resulting HDF5, and verifies:
  - Schema structure (version, groups, channel names, sampling rates)
  - Signal data (every channel, every sample, zero tolerance)
  - Annotations where applicable

Real data tests auto-skip when test files are absent.
Synthetic tests always run.
"""

import numpy as np
import h5py
import mne
import pytest
from pathlib import Path

from krembil_kit.io import ingest, SCHEMA_VERSION
from krembil_kit.io._readers_edf import _channel_sampling_rates
from conftest import (
    DATA_DIR, require_file, require_dir,
    create_synthetic_edf, create_synthetic_edfd,
)


# ────────────────────────────────────────────────────────────────────
# Shared verification
# ────────────────────────────────────────────────────────────────────

CHUNK_SECONDS = 10.0


def verify_structure(h5_path, expected_channels, expected_sfreq,
                     expect_discontinuous=False):
    """Check schema version, groups, channel names, rates."""
    with h5py.File(h5_path, "r") as f:
        assert f.attrs["schema_version"] == SCHEMA_VERSION
        for grp in ("signals", "channels", "events", "metadata"):
            assert grp in f

        names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["channels"]["names"][:]
        ]
        assert names == list(expected_channels)
        assert np.all(f["channels"]["sampling_rates"][:] == expected_sfreq)
        assert f["signals"].attrs["discontinuous"] == expect_discontinuous


def verify_signals(h5_path, raw):
    """Stream-compare every sample in HDF5 against MNE Raw. Zero tolerance."""
    sfreq = raw.info["sfreq"]
    n_samples = raw.n_times
    chunk = int(CHUNK_SECONDS * sfreq)

    with h5py.File(h5_path, "r") as f:
        ds = f["signals"]["data"]
        assert ds.shape == (len(raw.ch_names), n_samples)

        start = 0
        while start < n_samples:
            stop = min(start + chunk, n_samples)
            edf_chunk = raw.get_data(start=start, stop=stop).astype(np.float32)
            h5_chunk = ds[:, start:stop]
            assert np.max(np.abs(h5_chunk - edf_chunk)) == 0.0
            start = stop


def verify_segments(h5_path, ref_paths):
    """Compare each HDF5 segment against a reference EDF+C file."""
    with h5py.File(h5_path, "r") as f:
        n_segments = f["signals"].attrs["n_segments"]
        assert n_segments == len(ref_paths)

        h5_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["channels"]["names"][:]
        ]

        for i in range(n_segments):
            h5_data = f["signals"][f"segment_{i}"][:]
            ref = mne.io.read_raw_edf(str(ref_paths[i]), preload=False, verbose=False)
            ref_names = ref.ch_names
            common = [ch for ch in h5_names if ch in ref_names]
            assert len(common) > 0

            n_samp = min(h5_data.shape[1], ref.n_times)
            for ch in common:
                h5_ch = h5_data[h5_names.index(ch), :n_samp]
                ref_ch = ref.get_data(
                    picks=[ref_names.index(ch)], start=0, stop=n_samp
                )[0].astype(np.float32)
                assert np.max(np.abs(h5_ch - ref_ch)) == 0.0


def verify_events(h5_path, raw):
    """Check annotations match MNE's parsing."""
    annot = raw.annotations
    with h5py.File(h5_path, "r") as f:
        onsets = f["events"]["onsets"][:]
        durations = f["events"]["durations"][:]
        descriptions = [
            d.decode() if isinstance(d, bytes) else d
            for d in f["events"]["descriptions"][:]
        ]

    if annot is not None and len(annot) > 0:
        assert len(onsets) == len(annot)
        for i in range(len(annot)):
            assert abs(onsets[i] - annot.onset[i]) < 0.001
            assert abs(durations[i] - annot.duration[i]) < 0.001
            assert descriptions[i] == annot.description[i]
    else:
        assert len(onsets) == 0


# ────────────────────────────────────────────────────────────────────
# Synthetic tests
# ────────────────────────────────────────────────────────────────────

class TestSyntheticEDF:

    def test_structure(self, tmp_path):
        path, _, ch_names, sfreq = create_synthetic_edf(edf_plus=False)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=False)

    def test_signals(self, tmp_path):
        path, _, _, _ = create_synthetic_edf(edf_plus=False)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        with h5py.File(h5, "r") as f:
            h5_data = f["signals"]["data"][:]
        edf_data = raw.get_data().astype(np.float32)
        assert np.max(np.abs(h5_data - edf_data)) == 0.0


class TestSyntheticEDFC:

    def test_structure(self, tmp_path):
        path, _, ch_names, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=False)

    def test_signals(self, tmp_path):
        path, _, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        # EDF 16-bit quantization: compare against MNE's reading
        with h5py.File(h5, "r") as f:
            h5_data = f["signals"]["data"][:]
        edf_data = raw.get_data().astype(np.float32)
        assert np.max(np.abs(h5_data - edf_data)) == 0.0

    def test_annotations(self, tmp_path):
        annotations = [(1.0, 0.0, "event_A"), (3.5, 0.5, "event_B")]
        path, _, _, _ = create_synthetic_edf(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        verify_events(h5, raw)


class TestSyntheticEDFD:

    def test_structure(self, tmp_path):
        path, _, _, ch_names, sfreq, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=True)

    def test_segment_count(self, tmp_path):
        path, segments, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        with h5py.File(h5, "r") as f:
            assert f["signals"].attrs["n_segments"] == len(segments)

    def test_segment_start_times(self, tmp_path):
        path, _, start_times, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        with h5py.File(h5, "r") as f:
            for i, expected in enumerate(start_times):
                actual = f["signals"][f"segment_{i}"].attrs["start_time_seconds"]
                assert abs(actual - expected) < 0.01

    def test_signals(self, tmp_path):
        path, _, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        with h5py.File(h5, "r") as f:
            total = 0
            for i in range(f["signals"].attrs["n_segments"]):
                seg = f["signals"][f"segment_{i}"][:]
                mne_seg = raw.get_data(start=total, stop=total + seg.shape[1]).astype(np.float32)
                assert np.max(np.abs(seg - mne_seg)) == 0.0
                total += seg.shape[1]

    def test_annotations(self, tmp_path):
        annotations = [
            (0.5, 0.0, "stim_on"),     # in segment 0
            (1.5, 0.0, "stim_off"),    # in segment 0
            (5.5, 2.0, "seizure"),     # in segment 1, with duration
            (10.2, 0.0, "spike"),      # in segment 2
        ]
        path, _, _, _, _, _ = create_synthetic_edfd(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with h5py.File(h5, "r") as f:
            h5_onsets = f["events"]["onsets"][:]
            h5_durations = f["events"]["durations"][:]
            h5_descs = [
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            ]

        for onset, dur, desc in annotations:
            matched = any(
                h5_descs[i] == desc
                and abs(h5_onsets[i] - onset) < 0.01
                and abs(h5_durations[i] - dur) < 0.01
                for i in range(len(h5_onsets))
            )
            assert matched, (
                f"Annotation not found: t={onset}s \"{desc}\""
            )


# ────────────────────────────────────────────────────────────────────
# Real data: plain EDF
# ────────────────────────────────────────────────────────────────────

EDF_DIR = DATA_DIR / "test_edf"
_edf_files = sorted(EDF_DIR.glob("*.edf")) if EDF_DIR.is_dir() else []


def _has_mixed_rates(path):
    """True if the file's channels do not share one sampling rate."""
    return len(set(_channel_sampling_rates(str(path)))) > 1


class TestRealEDF:

    @pytest.mark.parametrize("edf_file", _edf_files, ids=[f.name for f in _edf_files])
    def test_signals(self, edf_file, tmp_path):
        require_file(edf_file)
        if _has_mixed_rates(edf_file):
            pytest.skip("mixed sampling rates — see test_mixed_rates_rejected")

        h5 = str(tmp_path / "out.h5")
        ingest(str(edf_file), output_path=h5)
        raw = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False)
        verify_structure(h5, raw.ch_names, raw.info["sfreq"], expect_discontinuous=False)
        verify_signals(h5, raw)

    @pytest.mark.parametrize("edf_file", _edf_files, ids=[f.name for f in _edf_files])
    def test_mixed_rates_rejected(self, edf_file, tmp_path):
        """
        Files with per-channel sampling rates cannot be ingested
        losslessly, so ingest must refuse them rather than write
        resampled data. See known_issues.txt item 1.
        """
        require_file(edf_file)
        if not _has_mixed_rates(edf_file):
            pytest.skip("uniform sampling rates")

        with pytest.raises(ValueError, match="mixed per-channel sampling rates"):
            ingest(str(edf_file), output_path=str(tmp_path / "out.h5"))


# ────────────────────────────────────────────────────────────────────
# Real data: EDF+C
# ────────────────────────────────────────────────────────────────────

EDFC_DIR = DATA_DIR / "test_edfc"
_edfc_files = sorted(EDFC_DIR.glob("*.edf")) if EDFC_DIR.is_dir() else []


class TestRealEDFC:

    @pytest.mark.parametrize("edf_file", _edfc_files, ids=[f.name for f in _edfc_files])
    def test_signals(self, edf_file, tmp_path):
        require_file(edf_file)
        h5 = str(tmp_path / "out.h5")
        ingest(str(edf_file), output_path=h5)
        raw = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False)
        verify_structure(h5, raw.ch_names, raw.info["sfreq"], expect_discontinuous=False)
        verify_signals(h5, raw)

    @pytest.mark.parametrize("edf_file", _edfc_files, ids=[f.name for f in _edfc_files])
    def test_events(self, edf_file, tmp_path):
        require_file(edf_file)
        h5 = str(tmp_path / "out.h5")
        ingest(str(edf_file), output_path=h5)
        raw = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False)
        verify_events(h5, raw)


# ────────────────────────────────────────────────────────────────────
# Real data: EDF+D
# ────────────────────────────────────────────────────────────────────

EDFD_DIR = DATA_DIR / "test_edfd"
EDFD_FILE = EDFD_DIR / "discontinuous.EDF"
EDFD_REF_DIR = EDFD_DIR / "reference_segments"


class TestRealEDFD:

    def test_signals(self, tmp_path):
        require_file(EDFD_FILE)
        require_dir(EDFD_REF_DIR)

        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        raw = mne.io.read_raw_edf(str(EDFD_FILE), preload=False, verbose=False)
        verify_structure(h5, raw.ch_names, raw.info["sfreq"], expect_discontinuous=True)

        ref_files = sorted(EDFD_REF_DIR.glob("*.edf"), key=lambda p: p.name)
        assert len(ref_files) > 0
        verify_segments(h5, ref_files)

    def test_events(self, tmp_path):
        """
        Verify annotations by comparing against reference segments.
        Each reference segment's annotations are offset by the
        corresponding HDF5 segment start time to get absolute times.
        Every reference annotation must have a match in the HDF5.
        """
        require_file(EDFD_FILE)
        require_dir(EDFD_REF_DIR)

        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        ref_files = sorted(EDFD_REF_DIR.glob("*.edf"), key=lambda p: p.name)

        with h5py.File(h5, "r") as f:
            h5_onsets = f["events"]["onsets"][:]
            h5_descs = [
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            ]

            # Collect reference annotations with absolute times
            n_segments = f["signals"].attrs["n_segments"]
            seg_starts = [
                f["signals"][f"segment_{i}"].attrs["start_time_seconds"]
                for i in range(n_segments)
            ]

        ref_annotations = []
        for seg_idx, ref_file in enumerate(ref_files):
            if seg_idx >= len(seg_starts):
                break
            ref_raw = mne.io.read_raw_edf(str(ref_file), preload=False, verbose=False)
            annot = ref_raw.annotations
            if annot is None or len(annot) == 0:
                continue
            for onset, desc in zip(annot.onset, annot.description):
                ref_annotations.append((seg_starts[seg_idx] + onset, desc))

        assert len(ref_annotations) > 0, "No reference annotations found"

        # Every reference annotation must match one in the HDF5
        tolerance = 0.5
        for ref_onset, ref_desc in ref_annotations:
            matched = any(
                h5_descs[i] == ref_desc and abs(h5_onsets[i] - ref_onset) < tolerance
                for i in range(len(h5_onsets))
            )
            assert matched, (
                f"Annotation not found in HDF5: t={ref_onset:.1f}s \"{ref_desc}\""
            )


# ────────────────────────────────────────────────────────────────────
# Real data: BrainVision
# ────────────────────────────────────────────────────────────────────

BV_DIR = DATA_DIR / "test_brainvision"
_bv_files = sorted(BV_DIR.glob("*.vhdr")) if BV_DIR.is_dir() else []


class TestRealBrainVision:

    @pytest.mark.parametrize("vhdr_file", _bv_files, ids=[f.name for f in _bv_files])
    def test_signals(self, vhdr_file, tmp_path):
        require_file(vhdr_file)
        h5 = str(tmp_path / "out.h5")
        ingest(str(vhdr_file), output_path=h5)
        raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=False, verbose=False)
        verify_structure(h5, raw.ch_names, raw.info["sfreq"], expect_discontinuous=False)
        verify_signals(h5, raw)

    @pytest.mark.parametrize("vhdr_file", _bv_files, ids=[f.name for f in _bv_files])
    def test_events(self, vhdr_file, tmp_path):
        require_file(vhdr_file)
        h5 = str(tmp_path / "out.h5")
        ingest(str(vhdr_file), output_path=h5)
        raw = mne.io.read_raw_brainvision(str(vhdr_file), preload=False, verbose=False)
        verify_events(h5, raw)
