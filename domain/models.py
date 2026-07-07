"""Core domain model for the LISA Global Fit visualization library.

Layer 1 (Core Domain). This module imports only the standard library and
``numpy``. It must NOT import any IO, rendering or statistics infrastructure
(no h5py, plotly, arviz, astropy, xarray). ``numpy`` is allowed solely as the
padding-free array boundary type shared between the ingestion and
visualization layers (NFR-01).

Domain-Driven Design vocabulary used here:

* Value Objects (immutable): :class:`IntrinsicParameters`,
  :class:`ExtrinsicParameters`, :class:`FrequencyBand`.
* Entity (identity, not value): :class:`GalacticBinary`.
* Aggregate Root: :class:`MCMCDraw` (a variable-length collection of
  :class:`GalacticBinary` -- the trans-dimensional source count is modelled as
  a first-class fact, never a padded rectangular array).
* IO/catalog models: :class:`GBCatalog`, :class:`CatalogSource`,
  :class:`GBChain`, :class:`MCMCChain`, :class:`NoiseModel`,
  :class:`MBHBCatalog`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Value Objects                                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class IntrinsicParameters:
    """Immutable intrinsic GB parameters.

    Sky position (declination, right ascension) is kept here because it is
    Doppler-coupled to the waveform and is therefore effectively intrinsic.
    """

    f0: float  # Hz
    fdot: float  # Hz/s
    declination: float  # rad, in [-pi/2, pi/2]
    right_ascension: float  # rad, in [0, 2*pi)


@dataclass(frozen=True)
class ExtrinsicParameters:
    """Immutable extrinsic GB parameters."""

    amplitude: float  # dimensionless strain amplitude
    inclination: float  # rad, in [0, pi]
    polarization: float  # rad, in [0, pi]
    initial_phase: float  # rad, in [0, 2*pi)


@dataclass(frozen=True)
class FrequencyBand:
    """Immutable frequency sub-band (used for FR-08 filtering and for
    rescaling sampling-space ``fr`` to a physical frequency)."""

    f_min: float  # Hz
    f_max: float  # Hz
    label: str = ""
    fdot_min: Optional[float] = None  # Hz/s, prior bound if known
    fdot_max: Optional[float] = None  # Hz/s, prior bound if known

    @property
    def f_center(self) -> float:
        return 0.5 * (self.f_min + self.f_max)

    def contains(self, frequency: float) -> bool:
        return self.f_min <= frequency <= self.f_max


# --------------------------------------------------------------------------- #
# Entity                                                                       #
# --------------------------------------------------------------------------- #


@dataclass(eq=False)
class GalacticBinary:
    """A galactic binary source. This is an Entity: two instances with the same
    ``source_id`` denote the same source even if their parameter estimates
    differ between draws/iterations. Equality and hashing are by identity.

    ``extrinsic`` may be ``None`` for an intrinsic-only search draw (the
    F-statistic marginalises the four extrinsic parameters analytically).
    """

    intrinsic: IntrinsicParameters
    extrinsic: Optional[ExtrinsicParameters]
    snr: float
    source_id: str

    def __eq__(self, other: object) -> bool:
        return isinstance(other, GalacticBinary) and other.source_id == self.source_id

    def __hash__(self) -> int:
        return hash(self.source_id)


# --------------------------------------------------------------------------- #
# Aggregate root + catalog/chain models                                        #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class MCMCDraw:
    """Aggregate root: one trans-dimensional MCMC draw, holding a
    variable-length collection of sources. The source count is a first-class
    fact (``nsource``).

    Deliberately sampler-agnostic: reversible-jump birth/death moves (Eryn) and
    product-space slot switching (jexplore) both reduce to the same per-draw
    source list, so the model records that outcome, not the mechanism.
    """

    iteration_index: int
    sources: List[GalacticBinary]
    walker_id: int = 0
    log_likelihood: float = float("nan")

    @property
    def nsource(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class CatalogSource:
    """A catalog entry: a :class:`GalacticBinary` plus the catalog-only summary
    statistics carried in ``central_catalog.h5``."""

    binary: GalacticBinary
    confidence: float
    bayes_factor: float
    origin: str = ""

    @property
    def snr(self) -> float:
        return self.binary.snr


@dataclass(frozen=True)
class GBCatalog:
    """Resolved-source catalog (``central_catalog.h5``)."""

    sources: List[CatalogSource]

    def __len__(self) -> int:
        return len(self.sources)


@dataclass(frozen=True)
class GBChain:
    """One walker's intrinsic F-statistic search chain, stored padding-free in
    *sampling* (hypercube) space exactly as written to disk.

    ``samples`` maps each sampling-space parameter name (e.g. ``"fr"``,
    ``"fdot"``, ``"dec_sin"``, ``"alpha"``) to a 1-D array of length ``n_draws``.
    Conversion to physical space is the responsibility of
    :mod:`lisaviz.domain.sampling`.
    """

    walker_id: int
    samples: Dict[str, np.ndarray]
    log_likelihood: np.ndarray
    log_prior: Optional[np.ndarray] = None
    band: Optional[FrequencyBand] = None

    @property
    def n_draws(self) -> int:
        return int(len(self.log_likelihood))

    @property
    def param_names(self) -> List[str]:
        return list(self.samples.keys())


@dataclass(frozen=True)
class MCMCChain:
    """A collection of per-walker chains for a single sub-band, plus the
    optional trans-dimensional source-count trajectory (``nsource``)."""

    walkers: List[GBChain]
    band: Optional[FrequencyBand] = None
    nsource_trace: Optional[np.ndarray] = None  # shape (n_walkers, n_draws) or (n_draws,)

    @property
    def n_walkers(self) -> int:
        return len(self.walkers)

    @property
    def param_names(self) -> List[str]:
        return self.walkers[0].param_names if self.walkers else []


# --------------------------------------------------------------------------- #
# Noise + MBHB models                                                          #
# --------------------------------------------------------------------------- #

NOISE_PARAM_NAMES = (
    "Sacc_log10",
    "Soms_log10",
    "A_log10",
    "f1",
    "f2",
    "alpha",
    "fknee",
)


@dataclass(frozen=True)
class NoiseModel:
    """The 7-parameter instrument + confusion noise model (``noise_pars.h5``)
    for a given iteration/band."""

    parameters: Dict[str, float]
    iteration: int = 0
    origin: str = ""
    model: str = ""

    def __post_init__(self) -> None:
        missing = [n for n in NOISE_PARAM_NAMES if n not in self.parameters]
        if missing:
            raise ValueError(f"NoiseModel missing parameters: {missing}")


@dataclass(frozen=True)
class ReconstructionData:
    """Signal-reconstruction inputs for one sub-band (FR-06).

    Carries the frequency-domain arrays the reconstruction views need: the
    observed vs reconstructed PSD plus the noise model evaluated to a curve (for
    the residual view), and the observed/recovered/injected spectra (for the
    waveform overlay + match score). Backend-neutral NumPy arrays only.
    """

    freq: np.ndarray
    observed_psd: np.ndarray
    model_psd: np.ndarray
    noise_psd: Optional[np.ndarray] = None
    noise_evolution: Optional[Sequence] = None        # [(label, psd_array), ...]
    # waveform overlay (complex frequency-domain spectra over ``overlay_freq``)
    overlay_freq: Optional[np.ndarray] = None
    observed_spectrum: Optional[np.ndarray] = None
    recovered: Optional[Sequence[np.ndarray]] = None
    injection: Optional[np.ndarray] = None
    overlay_psd: Optional[np.ndarray] = None           # noise PSD for the match score
    labels: Optional[Sequence[str]] = None
    band: Optional[FrequencyBand] = None


@dataclass(frozen=True)
class MBHBCatalog:
    """Massive black-hole binary catalog (``mbhb.h5``): recovered ``final`` rows
    and ``injected`` (preobs) rows, each a list of name->value dicts."""

    final: List[Dict[str, float]] = field(default_factory=list)
    injected: List[Dict[str, float]] = field(default_factory=list)


@dataclass(frozen=True)
class FstatGrid:
    """F-statistic grid evaluations over the intrinsic parameter space
    (``fstats.pkl``: ``gridp`` of shape (N, n_params) in sampling/hypercube space
    + ``fstats`` of shape (N,)). This is the search detection surface the pipeline
    evaluates *before* launching a full MCMC; rendering 2-D contour slices of it
    exposes parameter degeneracies (FR-09 / US-05).
    """

    param_names: tuple
    grid: np.ndarray  # (N, n_params), sampling-space (hypercube)
    fstat: np.ndarray  # (N,)
    band: Optional[FrequencyBand] = None

    def variable_axes(self) -> List[str]:
        """Parameter names that actually vary (more than one grid value); a fixed
        axis cannot be a contour dimension."""
        return [
            n for i, n in enumerate(self.param_names)
            if np.unique(np.round(self.grid[:, i], 9)).size > 1
        ]

    def project_pair(self, name_x: str, name_y: str, bins: int = 60):
        """Profile the F-statistic onto the (name_x, name_y) plane: the value of
        each 2-D cell is the maximum F over all other (profiled-out) dimensions.

        Returns ``(x_centers, y_centers, Z)`` with ``Z`` shaped ``(bins, bins)``
        indexed ``[y, x]`` and empty cells set to NaN. Coordinates are in
        sampling (hypercube) space; the visualization layer maps them to physical
        units for display.
        """
        ix = self.param_names.index(name_x)
        iy = self.param_names.index(name_y)
        x = self.grid[:, ix].astype(float)
        y = self.grid[:, iy].astype(float)
        z = self.fstat.astype(float)
        xe = np.linspace(x.min(), x.max(), bins + 1)
        ye = np.linspace(y.min(), y.max(), bins + 1)
        xb = np.clip(np.digitize(x, xe) - 1, 0, bins - 1)
        yb = np.clip(np.digitize(y, ye) - 1, 0, bins - 1)
        flat = yb * bins + xb
        acc = np.full(bins * bins, -np.inf)
        np.maximum.at(acc, flat, z)
        acc[~np.isfinite(acc)] = np.nan
        z_grid = acc.reshape(bins, bins)
        xc = 0.5 * (xe[:-1] + xe[1:])
        yc = 0.5 * (ye[:-1] + ye[1:])
        return xc, yc, z_grid
