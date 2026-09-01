"""
Loader tests.

Checks that load() gives back everything ingest() put into the HDF5:
  - Metadata properties (channel names, rate, duration, events)
  - Signal access, whole and sliced, by position and by channel name
  - Segment access for discontinuous files
  - Refusals for inputs that cannot be served

Uses the same synthetic and real files as the ingestion tests.
"""

import numpy as np
import h5py
import mne
import pytest

from krembil_kit.io import ingest, load, SCHEMA_VERSION
# Imported as a module so the large-read threshold can be lowered.
from krembil_kit.io import _loader
from conftest import (
    DATA_DIR, require_file,
    create_synthetic_edf, create_synthetic_edfd,
)


# ────────────────────────────────────────────────────────────────────
# Synthetic: continuous
# ────────────────────────────────────────────────────────────────────

class TestLoaderSyntheticContinuous:

    def test_properties(self, tmp_path):
        path, ch_names, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            assert recording.channel_names == ch_names
            assert recording.sampling_rate == sfreq
            assert recording.n_channels == len(ch_names)
            assert recording.discontinuous is False
            assert recording.n_segments is None
            assert recording.n_samples == int(5 * sfreq)
            assert recording.duration_seconds == pytest.approx(5.0)

            # Length not content: the readers report "V" for every
            # channel whatever the file says, so asserting "V" here
            # would lock in behaviour we intend to correct.
            assert len(recording.channel_units) == len(ch_names)

            assert recording.metadata["source_format"] == "EDF+C"
            assert f"{len(ch_names)}ch" in repr(recording)

    def test_get_signals_full(self, tmp_path):
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        expected = raw.get_data().astype(np.float32)

        with load(h5) as recording:
            signals = recording.get_signals()
            assert signals.shape == expected.shape
            assert np.max(np.abs(signals - expected)) == 0.0

    def test_get_signals_sliced(self, tmp_path):
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        start, stop = 100, 500
        expected = raw.get_data(start=start, stop=stop).astype(np.float32)

        with load(h5) as recording:
            signals = recording.get_signals(start=start, stop=stop)
            assert signals.shape == expected.shape
            assert np.max(np.abs(signals - expected)) == 0.0

    def test_get_signals_by_channel(self, tmp_path):
        """
        Selecting channels must return those channels, in the order asked
        for. Checking only the shape would pass even if the wrong rows
        came back.
        """
        path, ch_names, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            everything = recording.get_signals(start=0, stop=100)

            subset = recording.get_signals(
                channels=[ch_names[0], ch_names[2]], start=0, stop=100
            )
            assert subset.shape == (2, 100)
            assert np.array_equal(subset[0], everything[0])
            assert np.array_equal(subset[1], everything[2])

            # Asked for in reverse, they must come back in reverse. An
            # implementation that sorted the indices would fail here and
            # pass every assertion above.
            flipped = recording.get_signals(
                channels=[ch_names[2], ch_names[0]], start=0, stop=100
            )
            assert np.array_equal(flipped[0], everything[2])
            assert np.array_equal(flipped[1], everything[0])

    def test_large_read_warns(self, tmp_path, monkeypatch):
        """
        A read large enough to strain memory should say so rather than
        just doing it. The threshold is lowered here instead of building
        a file big enough to cross the real one.
        """
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        monkeypatch.setattr(_loader, "_LARGE_FILE_SAMPLES", 10)

        with load(h5) as recording:
            with pytest.warns(UserWarning, match="Pass start and stop"):
                recording.get_signals()

    def test_events(self, tmp_path):
        """
        The two annotations written into the file must come back with
        their onsets, durations and text intact, in onset order.
        """
        annotations = [(1.0, 0.0, "event_A"), (3.5, 0.5, "event_B")]
        path, _, _ = create_synthetic_edf(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            events = recording.events
            assert events is not None
            assert list(events["descriptions"]) == ["event_A", "event_B"]
            assert events["onsets"] == pytest.approx([1.0, 3.5], abs=1e-3)
            assert events["durations"] == pytest.approx([0.0, 0.5], abs=1e-3)

    def test_context_manager(self, tmp_path):
        """
        Leaving the with block closes the file. Reading after that must
        fail rather than return stale data.
        """
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            assert recording.n_channels == 3

        with pytest.raises(Exception):
            recording.get_signals()


# ────────────────────────────────────────────────────────────────────
# Synthetic: discontinuous
# ────────────────────────────────────────────────────────────────────

class TestLoaderSyntheticDiscontinuous:

    def test_properties(self, tmp_path):
        path, segments, _, ch_names, sfreq = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            assert recording.channel_names == ch_names
            assert recording.sampling_rate == sfreq
            assert recording.discontinuous is True
            assert recording.n_segments == len(segments)

            # Recorded samples only. The fixture spans 11 s of wall
            # clock but holds 5 s of data, so this would fail if
            # duration ever became the span instead.
            recorded_samples = sum(seg.shape[1] for seg in segments)
            assert recording.n_samples == recorded_samples
            assert recording.duration_seconds == pytest.approx(
                recorded_samples / sfreq
            )

            # A discontinuous recording reports segments where a
            # continuous one reports samples.
            assert f"{len(segments)} segments" in repr(recording)

    def test_segment_start_times(self, tmp_path):
        """
        Start times come back exactly. They travel through the file as
        ASCII in each record's first annotation, and the fixture uses
        whole seconds, so there is no rounding to allow for.
        """
        path, _, start_times, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            assert recording.segment_start_times == pytest.approx(start_times)

    def test_get_segment(self, tmp_path):
        path, segments, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)

        with load(h5) as recording:
            seg = recording.get_segment(0)
            assert seg.shape[1] == segments[0].shape[1]

            # Segment 0 begins at the start of the file, so MNE's view
            # of the same span is directly comparable.
            expected = raw.get_data(
                start=0, stop=seg.shape[1]
            ).astype(np.float32)
            assert np.max(np.abs(seg - expected)) == 0.0

    def test_get_segment_by_channel(self, tmp_path):
        path, _, _, ch_names, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            everything = recording.get_segment(1)
            subset = recording.get_segment(1, channels=[ch_names[2]])
            assert subset.shape == (1, everything.shape[1])
            assert np.array_equal(subset[0], everything[2])

    def test_events(self, tmp_path):
        """
        Annotations sit in the record covering their onset, and one of
        these falls in the second segment, so both must survive the
        split into segments.
        """
        annotations = [(0.5, 0.0, "stim_on"), (5.5, 0.0, "stim_off")]
        path, _, _, _, _ = create_synthetic_edfd(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            events = recording.events
            assert events is not None
            assert list(events["descriptions"]) == ["stim_on", "stim_off"]
            assert events["onsets"] == pytest.approx([0.5, 5.5], abs=1e-3)


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
        Minor schema changes are additive by rule, so a file written
        under a different minor version must still open. The major part
        is taken from the current version so this keeps testing a minor
        difference after the schema is bumped.
        """
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        major = SCHEMA_VERSION.split(".")[0]
        with h5py.File(h5, "r+") as f:
            f.attrs["schema_version"] = f"{major}.99"

        with load(h5) as recording:
            assert recording.n_channels > 0

    def test_missing_schema(self, tmp_path):
        bad_h5 = str(tmp_path / "bad.h5")
        with h5py.File(bad_h5, "w") as f:
            f.create_group("signals")
        with pytest.raises(ValueError, match="no schema_version"):
            load(bad_h5)

    def test_unparseable_schema_version(self, tmp_path):
        """
        The version is compared by major part, which means parsing it.
        A value that is not a version number must be refused clearly
        rather than raising from int().
        """
        bad_h5 = str(tmp_path / "bad.h5")
        with h5py.File(bad_h5, "w") as f:
            f.attrs["schema_version"] = "not-a-version"
        with pytest.raises(ValueError, match="not a version number"):
            load(bad_h5)

    def test_unknown_channel_name(self, tmp_path):
        """
        Asking for a channel that does not exist must say so, and say
        where to find the real ones.
        """
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            with pytest.raises(ValueError, match="does not have these"):
                recording.get_signals(channels=["NoSuchChannel"])

    def test_get_signals_on_discontinuous(self, tmp_path):
        """
        A discontinuous recording has no single sample axis, so serving
        get_signals would mean silently joining segments across gaps.
        """
        path, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            with pytest.raises(ValueError, match="discontinuous"):
                recording.get_signals()

    def test_get_segment_on_continuous(self, tmp_path):
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            with pytest.raises(ValueError, match="continuous"):
                recording.get_segment(0)

    def test_segment_index_out_of_range(self, tmp_path):
        path, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with load(h5) as recording:
            with pytest.raises(IndexError):
                recording.get_segment(99)


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

        with load(h5) as recording:
            assert recording.channel_names == list(raw.ch_names)
            assert recording.sampling_rate == raw.info["sfreq"]
            assert recording.n_samples == raw.n_times
            assert recording.discontinuous is False

            # First 10 seconds rather than the whole file, which can run
            # to gigabytes.
            n = min(int(10 * recording.sampling_rate), recording.n_samples)
            expected = raw.get_data(start=0, stop=n).astype(np.float32)
            actual = recording.get_signals(start=0, stop=n)
            assert np.max(np.abs(actual - expected)) == 0.0


# ────────────────────────────────────────────────────────────────────
# Real data: discontinuous (EDF+D)
# ────────────────────────────────────────────────────────────────────

EDFD_FILE = DATA_DIR / "test_edfd" / "discontinuous.EDF"

# The reference export of this file, produced by EDFbrowser's
# edfplusdcnv, is seven separate EDFs, so seven is the count to expect.
EDFD_SEGMENTS = 7


class TestLoaderRealDiscontinuous:

    def test_properties_and_segments(self, tmp_path):
        require_file(EDFD_FILE)
        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        raw = mne.io.read_raw_edf(str(EDFD_FILE), preload=False, verbose=False)

        with load(h5) as recording:
            assert recording.discontinuous is True
            assert recording.n_segments == EDFD_SEGMENTS
            assert len(recording.segment_start_times) == EDFD_SEGMENTS

            # Every sample belongs to exactly one segment, so the
            # segments must account for the whole recording. A segment
            # read that silently returned nothing would fail here.
            total = sum(
                recording.get_segment(i).shape[1]
                for i in range(EDFD_SEGMENTS)
            )
            assert total == recording.n_samples

            seg = recording.get_segment(0)
            assert seg.shape[0] == recording.n_channels

            # Segment 0 begins at the start of the file. Comparing its
            # first 1000 samples is enough to catch a channel or scaling
            # mistake without reading the whole file twice.
            n = min(1000, seg.shape[1])
            expected = raw.get_data(start=0, stop=n).astype(np.float32)
            assert np.max(np.abs(seg[:, :n] - expected)) == 0.0
