"""Bounded, block-wise streaming reader for pandas/PyTables catalogs.

``central_catalog.h5`` is written in pandas 'fixed' format, which pandas itself
cannot row-slice. But the numeric columns live in a 2-D ``cat/blockN_values``
dataset, and h5py *can* read an arbitrary row hyperslab from it. This module
yields the requested numeric columns one fixed-size row-block at a time, so peak
memory is bounded by the block size rather than the catalog length.

READ-ONLY: only ever reads.
"""

from __future__ import annotations

from typing import Dict, Iterator, Sequence, Tuple

import h5py
import numpy as np


def _decode(x) -> str:
    return x.decode() if isinstance(x, bytes) else str(x)


def iter_catalog_blocks(
    path: str,
    key: str,
    wanted: Sequence[str],
    block_size: int = 50_000,
) -> Iterator[Tuple[Dict[str, np.ndarray], int, int]]:
    """Yield ``(columns, start, total)`` where ``columns`` maps each found
    on-disk column name in ``wanted`` to a 1-D float array for rows
    ``[start:start+block_size]``. Object/1-D blocks (e.g. pickled string columns)
    are skipped. Only ``block_size`` rows are resident at a time."""
    wanted = set(wanted)
    with h5py.File(path, "r") as f:
        g = f[key]
        # Discover the 2-D numeric blocks and where each wanted column lives.
        colmap: Dict[str, Tuple[str, int]] = {}
        k = 0
        while f"block{k}_values" in g:
            vals = g[f"block{k}_values"]
            if vals.ndim == 2:
                items = [_decode(x) for x in g[f"block{k}_items"][:]]
                for j, name in enumerate(items):
                    if name in wanted:
                        colmap[name] = (f"block{k}_values", j)
            k += 1

        total = int(g["axis1"].shape[0]) if "axis1" in g else 0
        if total == 0 and colmap:
            total = int(g[next(iter(colmap.values()))[0]].shape[0])

        # Group columns by source dataset so each block slice is read once.
        by_ds: Dict[str, list] = {}
        for name, (ds, j) in colmap.items():
            by_ds.setdefault(ds, []).append((name, j))

        for start in range(0, total, block_size):
            stop = min(start + block_size, total)
            out: Dict[str, np.ndarray] = {}
            for ds, cols in by_ds.items():
                chunk = g[ds][start:stop, :]  # hyperslab: only these rows hit RAM
                for name, j in cols:
                    out[name] = chunk[:, j].astype(float, copy=False)
            yield out, start, total
