"""
Loader tests.

Verifies that load() correctly exposes all data from ingested HDF5 files:
  - Metadata properties (channel names, rate, duration, events)
  - Signal access (full and sliced)
  - Segment access for discontinuous files
  - Error handling for invalid inputs

Uses the same synthetic and real files as the ingestion tests.
"""

import numpy as np
import h5py
import mne
import pytest
from pathlib import Path

from krembil_kit.io import ingest, load, SCHEMA_VERSION
from conftest import (
    DATA_DIR, require_file, require_dir,
    create_synthetic_edf, create_synthetic_edfd,
)


# ────────────────────────────────────────────────────────────────────
# Synthetic: continuous
# ────────────────────────────────────────────────────────────────────

class TestLoaderSyntheticContinuous:

    def test_properties(self, tmp_path):
        path, _, ch_names, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        assert data.channel_names == ch_names
        assert data.sampling_rate == sfreq
        assert data.n_channels == len(ch_names)
        assert data.discontinuous is False
        assert data.n_segments is None
        assert data.n_samples == int(5 * sfreq)
        assert abs(data.duration_seconds - 5.0) < 0.01
        data.close()

    def test_get_signals_full(self, tmp_path):
        path, _, _, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        expected = raw.get_data().astype(np.float32)

        signals = data.get_signals()
        assert signals.shape == expected.shape
        assert np.max(np.abs(signals - expected)) == 0.0
        data.close()

    def test_get_signals_sliced(self, tmp_path):
        path, _, _, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

        start, stop = 100, 500
        expected = raw.get_data(start=start, stop=stop).astype(np.float32)
        signals = data.get_signals(start=start, stop=stop)
        assert signals.shape == expected.shape
        assert np.max(np.abs(signals - expected)) == 0.0
        data.close()

    def test_get_signals_by_channel(self, tmp_path):
        path, _, ch_names, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        subset = [ch_names[0], ch_names[2]]
        signals = data.get_signals(channels=subset, start=0, stop=100)
        assert signals.shape == (2, 100)
        data.close()

    def test_events(self, tmp_path):
        annotations = [(1.0, 0.0, "event_A"), (3.5, 0.5, "event_B")]
        path, _, _, _ = create_synthetic_edf(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        assert data.events is not None
        assert len(data.events["descriptions"]) >= 2
        data.close()

    def test_context_manager(self, tmp_path):
        path, _, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as data:
            assert data.n_channels == 3


# ────────────────────────────────────────────────────────────────────
# Synthetic: discontinuous
# ────────────────────────────────────────────────────────────────────

class TestLoaderSyntheticDiscontinuous:

    def test_properties(self, tmp_path):
        path, segments, start_times, ch_names, sfreq, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        assert data.channel_names == ch_names
        assert data.sampling_rate == sfreq
        assert data.discontinuous is True
        assert data.n_segments == len(segments)

        # Recorded samples only. The fixture spans 11 s of wall clock but
        # holds 5 s of data, so this would fail if duration ever became
        # the span instead.
        recorded_samples = sum(seg.shape[1] for seg in segments)
        assert data.n_samples == recorded_samples
        assert abs(data.duration_seconds - recorded_samples / sfreq) < 1e-9

        data.close()

    def test_segment_start_times(self, tmp_path):
        path, _, start_times, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        actual = data.segment_start_times
        assert len(actual) == len(start_times)
        for a, e in zip(actual, start_times):
            assert abs(a - e) < 0.01
        data.close()

    def test_get_segment(self, tmp_path):
        path, _, _, _, sfreq, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

        # Segment 0: 2 seconds = 512 samples at 256 Hz
        seg = data.get_segment(0)
        expected = raw.get_data(start=0, stop=seg.shape[1]).astype(np.float32)
        assert np.max(np.abs(seg - expected)) == 0.0
        data.close()

    def test_get_segment_by_channel(self, tmp_path):
        path, _, _, ch_names, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        seg = data.get_segment(1, channels=[ch_names[0]])
        assert seg.shape[0] == 1
        data.close()

    def test_events(self, tmp_path):
        annotations = [(0.5, 0.0, "stim_on"), (5.5, 0.0, "stim_off")]
        path, _, _, _, _, _ = create_synthetic_edfd(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        assert data.events is not None
        descs = data.events["descriptions"]
        assert "stim_on" in descs
        assert "stim_off" in descs
        data.close()


# ────────────────────────────────────────────────────────────────────
# Error handling
# ────────────────────────────────────────────────────────────────────

class TestLoaderErrors:

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load("nonexistent.h5")

    def test_major_version_mismatch_is_refused(self, tmp_path):
        bad_h5 = str(tmp_path / "bad.h5")
        with h5py.File(bad_h5, "w") as f:
            f.attrs["schema_version"] = "99.0"
        with pytest.raises(ValueError, match="schema version 99.0"):
            load(bad_h5)

    def test_minor_version_still_loads(self, tmp_path):
        """
        Minor schema changes are additive by rule, so a file written under
        a different minor version must still open.
        """
        path, _, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with h5py.File(h5, "r+") as f:
            f.attrs["schema_version"] = "1.99"

        with load(h5) as recording:
            assert recording.n_channels > 0

    def test_missing_schema(self, tmp_path):
        bad_h5 = str(tmp_path / "bad.h5")
        with h5py.File(bad_h5, "w") as f:
            f.create_group("signals")
        with pytest.raises(ValueError, match="no schema_version"):
            load(bad_h5)

    def test_get_signals_on_discontinuous(self, tmp_path):
        path, _, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        with pytest.raises(ValueError, match="discontinuous"):
            data.get_signals()
        data.close()

    def test_get_segment_on_continuous(self, tmp_path):
        path, _, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        with pytest.raises(ValueError, match="continuous"):
            data.get_segment(0)
        data.close()

    def test_segment_index_out_of_range(self, tmp_path):
        path, _, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        data = load(h5)
        with pytest.raises(IndexError):
            data.get_segment(99)
        data.close()


# ────────────────────────────────────────────────────────────────────
# Real data: continuous (EDF+C)
# ────────────────────────────────────────────────────────────────────

EDFC_DIR = DATA_DIR / "test_edfc"
_edfc_files = sorted(EDFC_DIR.glob("*.edf")) if EDFC_DIR.is_dir() else []


class TestLoaderRealContinuous:

    @pytest.mark.parametrize("edf_file", _edfc_files, ids=[f.name for f in _edfc_files])
    def test_properties_and_signals(self, edf_file, tmp_path):
        require_file(edf_file)
        h5 = str(tmp_path / "out.h5")
        ingest(str(edf_file), output_path=h5)

        raw = mne.io.read_raw_edf(str(edf_file), preload=False, verbose=False)
        data = load(h5)

        assert data.channel_names == list(raw.ch_names)
        assert data.sampling_rate == raw.info["sfreq"]
        assert data.n_samples == raw.n_times
        assert data.discontinuous is False

        # Spot-check: first 10 seconds
        n = min(int(10 * data.sampling_rate), data.n_samples)
        expected = raw.get_data(start=0, stop=n).astype(np.float32)
        actual = data.get_signals(start=0, stop=n)
        assert np.max(np.abs(actual - expected)) == 0.0
        data.close()


# ────────────────────────────────────────────────────────────────────
# Real data: discontinuous (EDF+D)
# ────────────────────────────────────────────────────────────────────

EDFD_FILE = DATA_DIR / "test_edfd" / "discontinuous.EDF"


class TestLoaderRealDiscontinuous:

    def test_properties_and_segments(self, tmp_path):
        require_file(EDFD_FILE)
        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        data = load(h5)
        assert data.discontinuous is True
        assert data.n_segments == 7
        assert len(data.segment_start_times) == 7

        # Spot-check segment 0: first 1000 samples
        seg = data.get_segment(0)
        assert seg.shape[0] == data.n_channels
        assert seg.shape[1] > 0

        # Verify against MNE reading of original
        raw = mne.io.read_raw_edf(str(EDFD_FILE), preload=False, verbose=False)
        n = min(1000, seg.shape[1])
        expected = raw.get_data(start=0, stop=n).astype(np.float32)
        assert np.max(np.abs(seg[:, :n] - expected)) == 0.0
        data.close()
