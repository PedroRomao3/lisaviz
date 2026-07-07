"""Abstract repository interface (Dependency Inversion boundary).

The Domain layer declares *what* it needs ("give me draw N -> return an
MCMCDraw"); concrete adapters (HDF5, in-memory mock) live outside the Domain
and implement *how*. All reads are bounded/streaming -- there is intentionally
no ``all()`` that would materialise an entire 1.3 TB run.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import (
    FrequencyBand,
    FstatGrid,
    GBCatalog,
    MBHBCatalog,
    MCMCChain,
    MCMCDraw,
    NoiseModel,
    ReconstructionData,
)


class IGalacticBinaryRepository(ABC):
    """Read-only repository over a global-fit run."""

    @abstractmethod
    def list_subbands(self) -> List[FrequencyBand]:
        """Available GB sub-bands (FR-08 frequency filter index)."""

    @abstractmethod
    def get_catalog(self, band: Optional[FrequencyBand] = None) -> GBCatalog:
        """Resolved-source catalog, optionally restricted to one sub-band."""

    @abstractmethod
    def get_chain(
        self, subband_id: str, max_draws: Optional[int] = None, source_index: int = 0
    ) -> MCMCChain:
        """Per-walker MCMC chains for one source slot of a sub-band, bounded to
        ``max_draws``.

        ``source_index`` selects which source group to trace: the chains store
        sources in numbered groups and ``0`` is the always-present *primary*
        slot. Higher indices exist only where the trans-dimensional sampler had
        that many sources; see :meth:`source_indices`.
        """

    def source_indices(self, subband_id: str) -> List[int]:
        """Source slot indices available to :meth:`get_chain` for a sub-band.

        Default: ``[0]`` (primary slot only). Adapters that store multiple
        source groups override this so a UI can offer a source selector.
        """
        return [0]

    def has_chain(self, subband_id: str) -> bool:
        """Whether this sub-band actually has MCMC chains to plot. Default
        ``True``; adapters where only some bands were sampled override this so a
        UI can hide / filter sources that have no chain behind them."""
        return True

    def list_iterations(self) -> List[int]:
        """Available global-fit iterations (the outer refinement loop; each is a
        full catalog/noise/chain snapshot). Default ``[]`` = not
        iteration-organised (e.g. the mock). Adapters over a real run override
        this so a UI can offer an iteration selector."""
        return []

    @abstractmethod
    def get_draw(self, subband_id: str, draw_index: int) -> MCMCDraw:
        """A single trans-dimensional draw (variable source count) of a sub-band."""

    @abstractmethod
    def get_noise(self, subband_id: Optional[str] = None) -> NoiseModel:
        """The fitted noise model (global or per sub-band)."""

    @abstractmethod
    def get_fstat_grid(self, subband_id: str) -> FstatGrid:
        """The F-statistic grid evaluations for a sub-band (FR-09)."""

    def get_reconstruction(self, subband_id: str) -> "ReconstructionData":
        """Signal-reconstruction inputs for a sub-band (FR-06): observed vs
        reconstructed PSD + noise curve, and the observed/recovered/injected
        spectra for the waveform overlay. Optional capability; adapters that
        cannot supply it raise ``FileNotFoundError`` (or leave this default,
        which signals the view is unavailable)."""
        raise NotImplementedError(f"{type(self).__name__} has no reconstruction data")

    def get_mbhb(self) -> MBHBCatalog:  # optional capability
        """MBHB catalog (final + injected). Optional; default: empty."""
        return MBHBCatalog()
