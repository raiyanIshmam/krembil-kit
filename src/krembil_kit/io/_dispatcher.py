"""
Central Dispatcher — Format Routing
====================================

The sole entry point for data ingestion. Inspects the incoming file
extension, delegates to the appropriate isolated reader, and writes
the result to a standardized Version 1.0 HDF5 file.

Adding support for a new format requires:
    1. Writing an isolated reader function.
    2. Registering it in the _EXTENSION_MAP below.
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


# Maps file extensions to reader functions.
# EDF requires sub-dispatch based on header content.
_EXTENSION_MAP = {
    ".vhdr": read_brainvision,
    ".eeg": read_brainvision,
    ".vmrk": read_brainvision,
}

# EDF subtypes resolved dynamically
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

    Accepts a raw recording file, detects its format, extracts all
    data losslessly, and writes a Version 1.0 HDF5 file.

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
        Absolute path to the generated HDF5 file.

    Raises
    ------
    ValueError
        If the file extension is not supported.
    FileNotFoundError
        If the source file does not exist.

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

    # ── Route to appropriate reader ─────────────────────────────
    if extension == ".edf":
        subtype = detect_edf_subtype(str(source))
        reader = _EDF_READERS[subtype]
    elif extension in _EXTENSION_MAP:
        reader = _EXTENSION_MAP[extension]
    else:
        supported = sorted(
            set(list(_EXTENSION_MAP.keys()) + [".edf"])
        )
        raise ValueError(
            f"Unsupported file format: '{extension}'. "
            f"Supported extensions: {supported}"
        )

    # ── Execute reader ──────────────────────────────────────────
    data = reader(str(source))

    # ── Determine output path ───────────────────────────────────
    if output_path is None:
        output_path = str(source.with_suffix(".h5"))

    # ── Write to HDF5 ──────────────────────────────────────────
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
