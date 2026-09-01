"""
Central Dispatcher — Format Routing
====================================

The sole entry point for data ingestion. Looks at the file extension,
hands the file to the reader for that format, and writes the result to a
standardized HDF5 file.


ADDING A FORMAT

    1. Write a reader that takes a file path and returns the dictionary
       described below.
    2. Register it in _EXTENSION_MAP.

Nothing else changes. The schema writer, the loader, and everything
downstream work from that dictionary alone, which is why readers do not
need to know about each other.


WHAT A READER MUST RETURN

    signals              an MNE Raw object for a continuous recording,
                         or a list of arrays, one per segment, if
                         discontinuous
    channel_names        list of str, one per channel
    channel_units        list of str, one per channel
    sampling_rates       numpy array, one rate per channel
    events               dict holding onsets, durations and descriptions,
                         each a list of equal length; or None when the
                         recording has no events
    metadata             dict, written as HDF5 attributes
    discontinuous        bool, saying which of the two signal layouts
                         above is being returned
    segment_start_times  list of wall-clock start times, one per segment;
                         None when continuous

Everything describing channels — names, units, sampling rates, and the
rows of the signal array — must be the same length and in the same order.
A channel is identified by its position, not by its label.
"""

from pathlib import Path
from typing import Optional

from ._readers_edf import (
    detect_edf_subtype,
    read_edf,
    read_edf_plus_c,
    read_edf_plus_d,
)
from ._readers_brainvision import read_brainvision
from ._schema import write_hdf5


_EXTENSION_MAP = {
    ".vhdr": read_brainvision,
    ".eeg": read_brainvision,
    ".vmrk": read_brainvision,
}

_EDF_READERS = {
    "edf": read_edf,
    "edf+c": read_edf_plus_c,
    "edf+d": read_edf_plus_d,
}


def ingest(
    file_path: str,
    output_path: Optional[str] = None,
) -> Path:
    """
    Universal ingestion entry point.

    Accepts a raw recording file, works out its format, extracts the data
    without altering it, and writes a standardized HDF5 file.

    Parameters
    ----------
    file_path : str
        Path to the source recording file (.edf, .vhdr, .eeg, .vmrk).
    output_path : str, optional
        Destination path for the output .h5 file. If None, the HDF5
        file is written alongside the source with the same stem:
            input:  /data/subject/recording.edf
            output: /data/subject/recording.h5

    Returns
    -------
    Path
        Path to the written HDF5 file. Absolute when output_path was not
        given, since it is derived from the resolved source path.
        Otherwise exactly the path that was passed in.

    Raises
    ------
    FileNotFoundError
        If the source file does not exist.
    ValueError
        If the extension is not supported, or if the reader refuses the
        file — for example an EDF recording whose channels do not share a
        sampling rate.

    Examples
    --------
    >>> from krembil_kit.io import ingest
    >>> h5_path = ingest("data/Subject01/Subject01.edf")
    >>> print(h5_path)
    data/Subject01/Subject01.h5

    >>> h5_path = ingest("data/Subject02/recording.vhdr",
    ...                  output_path="output/Subject02.h5")
    """
    source = Path(file_path).resolve()

    if not source.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    extension = source.suffix.lower()

    # EDF needs a second lookup. The extension does not say whether the
    # file is plain EDF, EDF+C or EDF+D — that is in the header.
    if extension == ".edf":
        reader = _EDF_READERS[detect_edf_subtype(str(source))]
    elif extension in _EXTENSION_MAP:
        reader = _EXTENSION_MAP[extension]
    else:
        supported = sorted({*_EXTENSION_MAP, ".edf"})
        raise ValueError(
            f"Cannot read '{extension}' files. Supported extensions: "
            f"{supported}"
        )

    data = reader(str(source))

    if output_path is None:
        output_path = str(source.with_suffix(".h5"))

    result_path = write_hdf5(
        output_path=output_path,
        signals=data["signals"],
        channel_names=data["channel_names"],
        channel_units=data["channel_units"],
        sampling_rates=data["sampling_rates"],
        events=data["events"],
        metadata=data["metadata"],
        discontinuous=data["discontinuous"],
        segment_start_times=data["segment_start_times"],
    )

    return result_path
