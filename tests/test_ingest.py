"""
Ingestion tests for plain EDF, EDF+C, EDF+D and BrainVision.

Most tests ingest a file, open the resulting HDF5 and check it against
the source:
  - Schema structure: version, groups, channel names, sampling rates
  - Signal data: every channel, every sample, at zero tolerance
  - Annotations, where the format carries them

The rest cover what ingest does with a path it cannot use, which files
the readers refuse outright, and what write_hdf5 rejects before it opens
an output file.

Real data tests skip themselves when the files are absent, so the suite
runs on a machine without them. Synthetic tests always run.
"""

from collections import Counter
from pathlib import Path

import h5py
import mne
import numpy as np
import pytest

from krembil_kit.io import ingest, SCHEMA_VERSION
from krembil_kit.io._readers_edf import _parse_tals, _read_header
from krembil_kit.io._schema import write_hdf5
from conftest import (
    DATA_DIR, require_file, require_dir,
    create_edfd_with_two_annotation_channels,
    create_edfd_without_annotation_channel,
    create_synthetic_edf, create_synthetic_edfd,
)


# ────────────────────────────────────────────────────────────────────
# Shared verification
# ────────────────────────────────────────────────────────────────────

# Signals are compared in chunks so that a large file is never held in
# memory twice. This is deliberately not the same size the writer uses,
# so a mistake tied to chunk boundaries cannot cancel itself out between
# the write and the read.
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

        # The three channel datasets are parallel: entry i of each
        # describes the same channel. Length is the part worth asserting;
        # what the units actually say is a separate question.
        assert len(f["channels"]["units"][:]) == len(names)
        assert len(f["channels"]["sampling_rates"][:]) == len(names)


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
    """
    Compare every segment against its reference file, every channel and
    every sample, at zero tolerance.

    The channel and length assertions are deliberately strict. Comparing
    only the channels whose names appear in both files, over only the
    shorter of the two segments, would let this pass while checking a
    fraction of the data — so a mismatch in either is a failure rather
    than something to work around.
    """
    with h5py.File(h5_path, "r") as f:
        n_segments = f["signals"].attrs["n_segments"]
        assert n_segments == len(ref_paths)

        h5_names = [
            n.decode() if isinstance(n, bytes) else n
            for n in f["channels"]["names"][:]
        ]

        for i in range(n_segments):
            h5_data = f["signals"][f"segment_{i}"][:]
            ref = mne.io.read_raw_edf(
                str(ref_paths[i]), preload=False, verbose=False
            )
            ref_names = ref.ch_names

            absent = [ch for ch in h5_names if ch not in ref_names]
            assert not absent, (
                f"segment {i}: reference file has no channel named {absent}"
            )
            assert h5_data.shape[1] == ref.n_times, (
                f"segment {i}: {h5_data.shape[1]} samples in the HDF5 but "
                f"{ref.n_times} in the reference"
            )

            for ch in h5_names:
                h5_channel = h5_data[h5_names.index(ch), :]
                ref_channel = ref.get_data(
                    picks=[ref_names.index(ch)]
                )[0].astype(np.float32)
                assert np.max(np.abs(h5_channel - ref_channel)) == 0.0, (
                    f"segment {i}, channel {ch}: samples differ"
                )


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

        # Onsets and durations are decimal text in the EDF file, parsed
        # to float separately by us and by MNE, so allow the last digit
        # to differ rather than demanding identical floats.
        tolerance = 0.001
        for i in range(len(annot)):
            assert abs(onsets[i] - annot.onset[i]) < tolerance
            assert abs(durations[i] - annot.duration[i]) < tolerance
            assert descriptions[i] == annot.description[i]
    else:
        assert len(onsets) == 0


# ────────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────────

class TestIngestEntryPoint:
    """
    What ingest does with the path it is given, before any format
    specific reader runs.
    """

    def test_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            ingest(str(tmp_path / "absent.edf"))

    def test_unsupported_extension(self, tmp_path):
        """
        The message names the extension that was rejected and lists the
        ones that work, so a caller who passed the wrong file learns
        both facts at once.
        """
        other = tmp_path / "recording.txt"
        other.write_text("not a recording")

        with pytest.raises(ValueError, match=r"Cannot read '\.txt'"):
            ingest(str(other))

    def test_default_output_path(self):
        """
        With no output_path, ingest writes beside the source file with
        the extension swapped for .h5. Every other test passes an
        explicit path, so this is the only check on the default.
        """
        source, ch_names, sfreq = create_synthetic_edf()

        result = ingest(source)

        assert result.resolve() == Path(source).with_suffix(".h5").resolve()
        assert result.exists()
        verify_structure(str(result), ch_names, sfreq,
                         expect_discontinuous=False)

    def test_brainvision_eeg_without_header(self, tmp_path):
        """
        BrainVision splits one recording across .vhdr, .eeg and .vmrk.
        Handed the .eeg alone, the reader goes looking for the .vhdr
        beside it — copying one file of the three is an easy mistake, so
        the error names the file it could not find.
        """
        orphan = tmp_path / "recording.eeg"
        orphan.write_bytes(b"\x00" * 32)

        with pytest.raises(FileNotFoundError, match="header file not found"):
            ingest(str(orphan))


# ────────────────────────────────────────────────────────────────────
# Synthetic tests
# ────────────────────────────────────────────────────────────────────

class TestSyntheticEDF:

    def test_structure(self, tmp_path):
        path, ch_names, sfreq = create_synthetic_edf(edf_plus=False)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=False)

    def test_signals(self, tmp_path):
        path, _, _ = create_synthetic_edf(edf_plus=False)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        verify_signals(h5, raw)

    def test_no_events(self, tmp_path):
        """
        A plain EDF has no annotation channel, so /events must exist and
        be empty rather than be absent.
        """
        path, _, _ = create_synthetic_edf(edf_plus=False)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        verify_events(h5, raw)


class TestMalformedEDF:
    """
    Files the readers must refuse rather than convert.
    """

    def test_edfd_without_annotation_channel(self, tmp_path):
        """
        A discontinuous file with no 'EDF Annotations' signal has nowhere
        to record when each data record starts, so it cannot be split
        into segments. Ingest must say so rather than guess.
        """
        path = create_edfd_without_annotation_channel()
        with pytest.raises(ValueError, match="record start times"):
            ingest(path, output_path=str(tmp_path / "out.h5"))

    def test_two_annotation_channels(self, tmp_path):
        """
        Only the first 'EDF Annotations' signal carries record start
        times, so with two of them the reader cannot tell which to trust.
        Reading the wrong one would give event times in place of record
        start times, and the segment boundaries computed from them would
        be wrong with nothing to indicate it.
        """
        path = create_edfd_with_two_annotation_channels()
        with pytest.raises(ValueError, match="'EDF Annotations' signals"):
            ingest(path, output_path=str(tmp_path / "out.h5"))


class TestSchemaWriter:
    """
    Checks write_hdf5 performs directly, without going through a reader.
    """

    def test_channel_arrays_must_agree_in_length(self, tmp_path):
        """
        Names, units and rates describe the same channels by position, so
        a length mismatch would mislabel every channel past the first
        disagreement. write_hdf5 refuses before opening the file.
        """
        with pytest.raises(ValueError, match="does not line up"):
            write_hdf5(
                output_path=str(tmp_path / "out.h5"),
                signals=None,
                channel_names=["A", "B", "C"],
                channel_units=["V", "V"],
                sampling_rates=np.array([256.0, 256.0, 256.0]),
            )

    def test_nothing_is_written_when_channels_disagree(self, tmp_path):
        """
        The check runs before h5py opens the file, which truncates on
        open — so a rejected call must not leave a file behind.
        """
        out = tmp_path / "out.h5"
        with pytest.raises(ValueError):
            write_hdf5(
                output_path=str(out),
                signals=None,
                channel_names=["A", "B"],
                channel_units=["V"],
                sampling_rates=np.array([256.0, 256.0]),
            )
        assert not out.exists()

    def test_discontinuous_requires_segment_start_times(self, tmp_path):
        """
        Segment start times are the only record of when each segment
        began — nothing else in the file carries it. Writing a
        discontinuous recording without them would lose the timing
        permanently, so the writer refuses, and refuses before opening
        the file so no partial output is left behind.
        """
        out = tmp_path / "out.h5"
        with pytest.raises(ValueError, match="segment_start_times is required"):
            write_hdf5(
                output_path=str(out),
                signals=[np.zeros((2, 8), dtype=np.float32)],
                channel_names=["A", "B"],
                channel_units=["V", "V"],
                sampling_rates=np.array([256.0, 256.0]),
                discontinuous=True,
            )
        assert not out.exists()


class TestSyntheticEDFC:

    def test_structure(self, tmp_path):
        path, ch_names, sfreq = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=False)

    def test_signals(self, tmp_path):
        path, _, _ = create_synthetic_edf()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        verify_signals(h5, raw)

    def test_annotations(self, tmp_path):
        annotations = [(1.0, 0.0, "event_A"), (3.5, 0.5, "event_B")]
        path, _, _ = create_synthetic_edf(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        verify_events(h5, raw)


class TestSyntheticEDFD:

    def test_structure(self, tmp_path):
        path, _, _, ch_names, sfreq = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        verify_structure(h5, ch_names, sfreq, expect_discontinuous=True)

    def test_segment_count(self, tmp_path):
        path, segments, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        with h5py.File(h5, "r") as f:
            assert f["signals"].attrs["n_segments"] == len(segments)

    def test_segment_start_times(self, tmp_path):
        """
        Each segment's start time is stored on its dataset. The fixture
        uses whole seconds, which survive the trip through ASCII in the
        annotation channel exactly, so there is no rounding to allow for.
        """
        path, _, start_times, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        with h5py.File(h5, "r") as f:
            actual = [
                f["signals"][f"segment_{i}"].attrs["start_time_seconds"]
                for i in range(len(start_times))
            ]
        assert actual == pytest.approx(start_times)

    def test_signals(self, tmp_path):
        path, _, _, _, _ = create_synthetic_edfd()
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)
        raw = mne.io.read_raw_edf(path, preload=False, verbose=False)
        # MNE reads EDF+D as one continuous block and ignores the gaps,
        # so segment i sits at the sum of the lengths before it.
        with h5py.File(h5, "r") as f:
            offset = 0
            for i in range(f["signals"].attrs["n_segments"]):
                seg = f["signals"][f"segment_{i}"][:]
                mne_seg = raw.get_data(
                    start=offset, stop=offset + seg.shape[1]
                ).astype(np.float32)
                assert np.max(np.abs(seg - mne_seg)) == 0.0
                offset += seg.shape[1]

    def test_annotations(self, tmp_path):
        annotations = [
            (0.5, 0.0, "stim_on"),     # in segment 0
            (1.5, 0.0, "stim_off"),    # in segment 0
            (5.5, 2.0, "seizure"),     # in segment 1, with duration
            (10.2, 0.0, "spike"),      # in segment 2
        ]
        path, _, _, _, _ = create_synthetic_edfd(annotations=annotations)
        h5 = str(tmp_path / "out.h5")
        ingest(path, output_path=h5)

        with h5py.File(h5, "r") as f:
            h5_onsets = f["events"]["onsets"][:]
            h5_durations = f["events"]["durations"][:]
            h5_descs = [
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            ]

        # Compared element by element, not searched for. Each annotation
        # goes into the record covering its onset and the records are
        # written in time order, so the four come back in the order they
        # were given. Searching instead would pass even if one were
        # duplicated and another dropped.
        assert h5_descs == [desc for _, _, desc in annotations]
        assert h5_onsets == pytest.approx(
            [onset for onset, _, _ in annotations], abs=1e-3
        )
        assert h5_durations == pytest.approx(
            [dur for _, dur, _ in annotations], abs=1e-3
        )


# ────────────────────────────────────────────────────────────────────
# Real data: plain EDF
# ────────────────────────────────────────────────────────────────────

EDF_DIR = DATA_DIR / "test_edf"
_edf_files = sorted(EDF_DIR.glob("*.edf")) if EDF_DIR.is_dir() else []


def _has_mixed_rates(path):
    """True if the file's channels do not share one sampling rate."""
    return len(set(_read_header(str(path)).channel_rates)) > 1


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
        resampled data.
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
        The ingested annotations must match the reference segments
        exactly — same count, same descriptions, same times, both ways.

        The reference annotations are read with our own TAL parser rather
        than through MNE. MNE discards annotations it judges out of range,
        11 of the 356 here, so comparing against its count would measure
        our complete output against an already incomplete number. The
        comparison is still meaningful because edfplusdcnv wrote the bytes
        being parsed, and MNE's own reading agrees with ours on the 345 it
        keeps.

        Reference onsets are relative to a whole second. EDF headers store
        start times only to the second, so edfplusdcnv gives each segment
        a whole-second header time and carries the remainder in the first
        TAL. An absolute time is therefore int(start) + onset rather than
        start + onset, and using the latter shifts every annotation by the
        fractional part of its segment's start.

        Sorting both sides and comparing element by element rather than
        searching for matches: ten annotations here share one onset and
        two of those are exact duplicates, so a per-item search would pass
        even if a duplicate went missing.
        """
        require_file(EDFD_FILE)
        require_dir(EDFD_REF_DIR)

        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        with h5py.File(h5, "r") as f:
            descriptions = [
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            ]
            ingested = sorted(zip(f["events"]["onsets"][:], descriptions))

            segment_starts = [
                f["signals"][f"segment_{i}"].attrs["start_time_seconds"]
                for i in range(f["signals"].attrs["n_segments"])
            ]

        ref_files = sorted(EDFD_REF_DIR.glob("*.edf"), key=lambda p: p.name)
        assert len(ref_files) == len(segment_starts)

        reference = []
        for start, ref_file in zip(segment_starts, ref_files):
            _, annotations = _parse_tals(
                str(ref_file), _read_header(str(ref_file))
            )
            for annotation in annotations:
                reference.append(
                    (int(start) + annotation["onset"],
                     annotation["description"])
                )
        reference.sort()

        assert len(ingested) == len(reference), (
            f"{len(ingested)} annotations ingested but {len(reference)} "
            f"across the reference segments"
        )

        # One sample at 1024 Hz is close to 0.001 s. Onsets are decimal
        # text in both files, so allow the last digit to differ.
        tolerance = 0.001
        for (onset, description), (ref_onset, ref_description) in zip(
            ingested, reference
        ):
            assert description == ref_description
            assert abs(onset - ref_onset) < tolerance, (
                f"{description!r}: ingested at {onset}, "
                f"reference at {ref_onset}"
            )

    def test_events_include_everything_mne_finds(self, tmp_path):
        """
        Our TAL parser must recover at least what MNE recovers.

        test_events parses the reference segments with our own parser,
        which would not catch a mistake that misread both sides the same
        way. MNE is a separate implementation reading the same bytes, so
        it is an independent check — over the 345 of 356 annotations it
        keeps, discarding the rest as outside the data range.

        Descriptions rather than times, because the failure this guards
        against is mis-splitting TAL fields, which would corrupt or split
        the text. The times are checked against the reference bytes in
        test_events.
        """
        require_file(EDFD_FILE)
        require_dir(EDFD_REF_DIR)

        h5 = str(tmp_path / "out.h5")
        ingest(str(EDFD_FILE), output_path=h5)

        with h5py.File(h5, "r") as f:
            ours = Counter(
                d.decode() if isinstance(d, bytes) else d
                for d in f["events"]["descriptions"][:]
            )

        from_mne = Counter()
        for ref_file in sorted(EDFD_REF_DIR.glob("*.edf")):
            raw = mne.io.read_raw_edf(
                str(ref_file), preload=False, verbose=False
            )
            if raw.annotations is not None:
                from_mne.update(raw.annotations.description)

        assert from_mne, "MNE found no annotations in the reference segments"

        # Counter subtraction keeps only descriptions MNE found more of
        # than we did, which is exactly what "we lost something" means.
        missing = from_mne - ours
        assert not missing, f"MNE found annotations we did not: {dict(missing)}"


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
