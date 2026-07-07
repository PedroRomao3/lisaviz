"""Quantify the memory wasted by NaN-padding a trans-dimensional model into a
dense (xarray / ArviZ ``InferenceData``) tensor, versus the library's
padding-free, variable-cardinality representation.

The trans-dimensional galactic-binary sampler produces a *variable* number of
sources per draw / per sub-band. xarray and ArviZ's ``InferenceData`` are built on dense,
rectangular ``DataArray``s, so to store a "source" axis every draw must be
padded out to the global maximum source count, with the unused slots filled by
NaN. The waste is therefore::

    padding_fraction = 1 - mean_sources / max_sources

which is large precisely because the GB source-count distribution is heavily
skewed (most bands sparse, a few confused bands very dense).

Run it::

    python benchmarks/padding_memory.py
    python benchmarks/padding_memory.py --catalog /path/to/gfrun

Everything here is read-only; it never writes to the run data.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

BYTES_PER_FLOAT64 = 8
N_GB_PARAMS = 8


# --------------------------------------------------------------------------- #
#  Core accounting                                                            #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PaddingResult:
    label: str
    n_groups: int          # number of non-empty bands / draws
    mean_count: float
    max_count: int
    p99_count: float
    total_sources: int

    @property
    def padding_fraction(self) -> float:
        return 1.0 - self.mean_count / self.max_count

    @property
    def dense_bytes(self) -> int:
        # rectangular tensor: (n_groups, max_count, n_params) float64
        return self.n_groups * self.max_count * N_GB_PARAMS * BYTES_PER_FLOAT64

    @property
    def ragged_bytes(self) -> int:
        # only the real sources are stored
        return self.total_sources * N_GB_PARAMS * BYTES_PER_FLOAT64

    def as_row(self) -> str:
        return (
            f"{self.label:<22} | groups={self.n_groups:>6} | "
            f"sources/group mean={self.mean_count:6.1f} max={self.max_count:>4d} "
            f"p99={self.p99_count:>5.0f} | NaN pad={self.padding_fraction * 100:4.1f}% | "
            f"dense={self.dense_bytes / 1e6:8.1f} MB  vs ragged={self.ragged_bytes / 1e6:7.1f} MB "
            f"({self.dense_bytes / max(self.ragged_bytes, 1):.1f}x)"
        )


def summarise_counts(label: str, counts: Sequence[int]) -> PaddingResult:
    nz = np.asarray([c for c in counts if c > 0], dtype=int)
    return PaddingResult(
        label=label,
        n_groups=int(nz.size),
        mean_count=float(nz.mean()),
        max_count=int(nz.max()),
        p99_count=float(np.percentile(nz, 99)),
        total_sources=int(nz.sum()),
    )


def counts_per_band(frequencies: np.ndarray, width_hz: float) -> np.ndarray:
    edges = np.arange(frequencies.min(), frequencies.max() + width_hz, width_hz)
    counts, _ = np.histogram(frequencies, bins=edges)
    return counts


# --------------------------------------------------------------------------- #
#  Data sources                                                               #
# --------------------------------------------------------------------------- #
def load_real_frequencies(run_dir: str) -> Optional[np.ndarray]:
    """Read catalog frequencies through the read-only adapter, if available."""
    try:
        from lisaviz.ingestion.hdf5_adapter import HDF5RepositoryAdapter

        repo = HDF5RepositoryAdapter(run_dir)
        cat = repo.get_catalog()
        f = np.array([s.binary.intrinsic.f0 for s in cat.sources], dtype=float)
        return f[np.isfinite(f)]
    except Exception as exc:  # noqa: BLE001 - benchmark, degrade to synthetic
        print(f"(could not read real catalog from {run_dir!r}: {exc}; using synthetic)\n")
        return None


def synthetic_frequencies(n: int = 11000, seed: int = 0) -> np.ndarray:
    """A stand-in catalog with a realistic galactic-plane frequency clustering."""
    rng = np.random.default_rng(seed)
    f = np.concatenate([
        rng.uniform(0.3e-3, 43e-3, n // 2),               # broad spread
        rng.normal(3e-3, 0.5e-3, n // 2).clip(0.3e-3, 43e-3),  # a dense knot
    ])
    return f


# --------------------------------------------------------------------------- #
#  Report                                                                      #
# --------------------------------------------------------------------------- #
def per_band_report(frequencies: np.ndarray, widths_uhz=(4, 16, 64, 256)) -> None:
    print(f"catalog sources: {frequencies.size}   "
          f"span: {frequencies.min() * 1e3:.2f}-{frequencies.max() * 1e3:.2f} mHz\n")
    print("Padding a (band x source x param) tensor to the per-band max source count:")
    for w in widths_uhz:
        counts = counts_per_band(frequencies, w * 1e-6)
        print("  " + summarise_counts(f"width = {w:>4} uHz", counts).as_row())


def survey_scale_report(mean_count=6.0, max_count=60, params=N_GB_PARAMS) -> None:
    print("\nProjected to survey scale (representative skew max ~ 10x mean):")
    for tag, n in [("Sangria resolvable ~1e4", 10_000), ("Mojito Light 15.5M", 15_539_324)]:
        n_bands = n / mean_count
        dense = n_bands * max_count * params * BYTES_PER_FLOAT64
        ragged = n * params * BYTES_PER_FLOAT64
        print(f"  {tag:<24}: dense={dense / 1e9:6.2f} GB  ragged={ragged / 1e9:5.2f} GB  "
              f"wasted on NaN={ (dense - ragged) / 1e9:6.2f} GB ({(1 - ragged / dense) * 100:.0f}%)")


def chain_draws_report(n_draws=100000, mean_sources=4.0, max_sources=10, seed=42) -> None:
    print("\nPadding a (draw x source x param) MCMC chain to the max active source count:")
    rng = np.random.default_rng(seed)
    counts = rng.poisson(mean_sources, n_draws)
    counts = np.clip(counts, 1, max_sources)
    res = summarise_counts("trans-dim draw axis", counts)
    print("  " + res.as_row())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--catalog", default="gfrun_old",
                    help="Global Fit run directory to read real catalog frequencies "
                         "from (falls back to a synthetic catalog when absent)")
    args = ap.parse_args()

    freqs = load_real_frequencies(args.catalog)
    if freqs is None:
        freqs = synthetic_frequencies()

    per_band_report(freqs)
    chain_draws_report()
    survey_scale_report()


if __name__ == "__main__":
    main()
