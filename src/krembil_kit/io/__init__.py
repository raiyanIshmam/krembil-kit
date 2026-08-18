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
    >>> from krembil_kit.io import ingest
    >>> h5_path = ingest("data/Subject01/Subject01.edf")
    >>> h5_path = ingest("data/Subject02/recording.vhdr",
    ...                  output_path="output/Subject02.h5")
"""

from ._dispatcher import ingest
from ._schema import SCHEMA_VERSION

__all__ = ["ingest", "SCHEMA_VERSION"]
