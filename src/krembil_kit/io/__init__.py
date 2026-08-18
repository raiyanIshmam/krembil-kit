"""
krembil_kit.io — Data Ingestion and Standardization
====================================================

Universal translator that reads raw clinical recording files and
converts them into a standardized Version 1.0 HDF5 format with
zero preprocessing applied.

Supported formats:
    - EDF   (standard European Data Format)
    - EDF+C (continuous EDF+)
    - EDF+D (discontinuous EDF+, segment-aware)
    - BrainVision (.vhdr / .eeg / .vmrk)

Usage:
    >>> from krembil_kit.io import ingest, load
    >>> h5_path = ingest("data/Subject01/Subject01.edf")
    >>> data = load(h5_path)
    >>> data.channel_names
    ['Fp1', 'Fz', 'F3', ...]
    >>> chunk = data.get_signals(start=0, stop=5000)
"""

from ._dispatcher import ingest
from ._loader import load, Recording
from ._schema import SCHEMA_VERSION

__all__ = ["ingest", "load", "Recording", "SCHEMA_VERSION"]
