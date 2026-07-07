"""Low-level HDF5 / npy access helpers (Ingestion layer).

All h5py / pandas / numpy-IO usage is concentrated here so the rest of the
ingestion layer reads as domain-level intent. Everything is READ-ONLY.
"""

from __future__ import annotations

import pickle
from typing import Any, Dict, List

import h5py
import numpy as np
import pandas as pd


class ChunkLayoutError(Exception):
    """Raised when an HDF5 dataset we intend to stream has a chunk layout that
    is incompatible with sequential (row-range) access -- e.g. a single-row
    chunk along the iteration axis, which would cause chunk thrashing."""


def read_dataframe(path: str, key: str) -> pd.DataFrame:
    """Read a pandas/PyTables table. Catalog/noise/mbhb stores are written in
    'fixed' format, which must be read whole; callers project columns/rows in
    memory afterwards."""
    return pd.read_hdf(path, key=key)


def hdf5_keys(path: str) -> List[str]:
    with h5py.File(path, "r") as f:
        return list(f.keys())


def validate_streamable(path: str) -> None:
    """Fail-fast chunk inspection. Raises :class:`ChunkLayoutError` if any
    multi-row dataset is chunked with a single row along its first
    (sequential) axis. Contiguous datasets (``chunks is None``) are fine."""
    with h5py.File(path, "r") as f:
        def visit(name: str, obj: object) -> None:
            if isinstance(obj, h5py.Dataset) and obj.chunks is not None:
                if obj.ndim >= 1 and obj.shape[0] > 1 and obj.chunks[0] == 1:
                    raise ChunkLayoutError(
                        f"Dataset '{name}' in '{path}' is chunked with a single "
                        f"row along the sequential axis (chunks={obj.chunks}); "
                        f"this is incompatible with bounded sequential reads."
                    )
        f.visititems(visit)


def read_pickle(path: str) -> Any:
    """Read a pickled pipeline artifact (e.g. ``fstats.pkl``). Read-only."""
    with open(path, "rb") as fh:
        return pickle.load(fh)


def open_npy_chain(path: str) -> np.ndarray:
    """Memory-map a structured-array chain so that slicing reads only the
    requested draw range from disk (Lazy Load / Virtual Proxy)."""
    return np.load(path, mmap_mode="r")


def structured_field_groups(dtype: np.dtype) -> Dict[int, Dict[str, str]]:
    """Group structured-chain fields by trailing ``:<source>`` index.

    ``fr:0, fdot:0, ... , logL, logP`` -> {0: {"fr": "fr:0", ...}}. Fields with
    no ``:`` suffix (logL, logP) are returned under source key ``-1``.
    """
    groups: Dict[int, Dict[str, str]] = {}
    for field in dtype.names or ():
        if ":" in field:
            base, idx = field.rsplit(":", 1)
            try:
                src = int(idx)
            except ValueError:
                src = -1
        else:
            base, src = field, -1
        groups.setdefault(src, {})[base] = field
    return groups
