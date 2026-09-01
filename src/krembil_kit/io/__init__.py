"""
krembil_kit.io — Data Ingestion and Standardization
====================================================

Universal translator that reads raw clinical recording files and converts
them into a standardized HDF5 format, applying no preprocessing.

Supported formats:
    - EDF   (standard European Data Format)
    - EDF+C (continuous EDF+)
    - EDF+D (discontinuous EDF+, segment-aware)
    - BrainVision (.vhdr / .eeg / .vmrk)

Usage:
    >>> from krembil_kit.io import ingest, load
    >>> h5_path = ingest("data/Subject01/Subject01.edf")
    >>> with load(h5_path) as recording:
    ...     recording.channel_names
    ...     chunk = recording.get_signals(start=0, stop=5000)
    ['Fp1', 'Fz', 'F3', ...]

SCHEMA_VERSION is the version of the HDF5 layout this release writes.
"""

from ._dispatcher import ingest
from ._loader import load, Recording
from ._schema import SCHEMA_VERSION

__all__ = ["ingest", "load", "Recording", "SCHEMA_VERSION"]
