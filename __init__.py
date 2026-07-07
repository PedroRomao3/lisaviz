"""lisaviz -- an interactive diagnostic library for the LISA Global Fit.

Three layers with dependencies pointing inward (NOT MVC):

* ``lisaviz.domain``        -- core domain model (stdlib + numpy only);
* ``lisaviz.ingestion``     -- repository adapters that hide HDF5;
* ``lisaviz.visualization`` -- plotting modules that hide the rendering backend.

The :class:`LISAViz` facade is a convenience composition for notebooks and the
Streamlit dashboard. Every underlying plotting function is also callable on its
own with plain domain objects.
"""

from __future__ import annotations

from typing import List, Optional

from .domain.models import FrequencyBand
from .domain.repository import IGalacticBinaryRepository
from .ingestion.hdf5_adapter import HDF5RepositoryAdapter
from .ingestion.mock import InMemoryMockRepository

# NOTE: the visualization layer (plotly / matplotlib / arviz-stats) is imported
# lazily -- inside the facade methods and via __getattr__ below -- so that the
# ingestion and domain layers can be used (and tested) with no rendering
# dependencies installed. This is the runtime counterpart to the inward
# dependency rule: importing lisaviz must not force the heavy viz stack.

__all__ = [
    "LISAViz",
    "HDF5RepositoryAdapter",
    "InMemoryMockRepository",
    "heuristic_eval",
    "evaluate_figures",
]

# Names re-exported lazily from the visualization layer on attribute access.
_LAZY_VIZ = {"heuristic_eval", "evaluate_figures"}


def __getattr__(name):  # PEP 562: lazy top-level attribute access
    if name in _LAZY_VIZ:
        import importlib

        return getattr(importlib.import_module(".visualization", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class LISAViz:
    """Thin composition over a repository + the visualization modules."""

    def __init__(self, run_dir: Optional[str] = None, repo: Optional[IGalacticBinaryRepository] = None,
                 iteration: Optional[int] = None):
        if repo is not None:
            self.repo = repo
        elif run_dir is not None:
            self.repo = HDF5RepositoryAdapter(run_dir, iteration=iteration)
        else:
            raise ValueError("Provide either run_dir or a repository instance.")

    @classmethod
    def mock(cls, **kwargs) -> "LISAViz":
        """A facade over synthetic in-memory data (no filesystem, no HDF5)."""
        return cls(repo=InMemoryMockRepository(**kwargs))

    # -- discovery -------------------------------------------------------- #
    def subbands(self) -> List[FrequencyBand]:
        return self.repo.list_subbands()

    def iterations(self) -> List[int]:
        """Available global-fit iterations (outer refinement loop). Empty when the
        source is not iteration-organised (e.g. the mock)."""
        return self.repo.list_iterations()

    def current_iteration(self) -> Optional[int]:
        """The iteration currently loaded (None when not applicable)."""
        return getattr(self.repo, "iteration", None)

    def sampled_bands(self) -> List[str]:
        """Labels of sub-bands that actually have MCMC chains (a clicked catalog
        source only has a trace/corner if it lands in one of these)."""
        return [b.label for b in self.repo.list_subbands() if self.repo.has_chain(b.label)]

    # -- population ------------------------------------------------------- #
    def sky_map(self, band: Optional[FrequencyBand] = None, **kwargs):
        from .visualization import plot_sky_map
        return plot_sky_map(self.repo.get_catalog(band), **kwargs)

    def sky_map_streaming(self, band: Optional[FrequencyBand] = None, block_size: int = 50_000, **kwargs):
        """Bounded-memory sky map: streams the catalog in fixed row-blocks into a
        fixed density grid + bounded outlier reservoir. Peak RAM is set by
        block_size, not catalog size. Requires an HDF5-backed repository."""
        from .visualization import plot_sky_map_streaming
        if not hasattr(self.repo, "iter_catalog_arrays"):
            raise TypeError("sky_map_streaming requires an HDF5RepositoryAdapter repository.")
        return plot_sky_map_streaming(self.repo.iter_catalog_arrays(block_size=block_size, band=band), **kwargs)

    def waterfall(self, band: Optional[FrequencyBand] = None, **kwargs):
        from .visualization import plot_waterfall
        return plot_waterfall(self.repo.get_catalog(band), **kwargs)

    def waterfall_streaming(self, band: Optional[FrequencyBand] = None, y_axis: str = "Amplitude",
                            block_size: int = 50_000, **kwargs):
        """Bounded-memory waterfall: streams the catalog in fixed row-blocks into
        a fixed freq x amplitude/SNR density grid + bounded outlier reservoir.
        Peak RAM is set by block_size, not catalog size. Requires an HDF5-backed
        repository."""
        from .visualization import plot_waterfall_streaming
        if not hasattr(self.repo, "iter_catalog_arrays"):
            raise TypeError("waterfall_streaming requires an HDF5RepositoryAdapter repository.")
        return plot_waterfall_streaming(
            self.repo.iter_catalog_arrays(block_size=block_size, band=band), y_axis=y_axis, **kwargs)

    # -- MCMC health / posterior ----------------------------------------- #
    def sources(self, subband: str) -> List[int]:
        """Source slot indices available for a sub-band's trace/corner/diagnostics
        (slot 0 is the primary source; see :meth:`IGalacticBinaryRepository`)."""
        return self.repo.source_indices(subband)

    def traces(self, subband: str, burn_in: int = 0, max_draws: Optional[int] = None,
               source_index: int = 0, **kwargs):
        from .visualization import plot_traces
        chain = self.repo.get_chain(subband, max_draws=max_draws, source_index=source_index)
        return plot_traces(chain, burn_in=burn_in, source_index=source_index, **kwargs)

    def autocorrelation(self, subband: str, max_draws: Optional[int] = None,
                        source_index: int = 0, **kwargs):
        from .visualization import plot_autocorrelation
        chain = self.repo.get_chain(subband, max_draws=max_draws, source_index=source_index)
        return plot_autocorrelation(chain, **kwargs)

    def diagnostics(self, subband: str, burn_in: int = 0, max_draws: Optional[int] = None,
                    source_index: int = 0):
        from .visualization import diagnostics_table
        chain = self.repo.get_chain(subband, max_draws=max_draws, source_index=source_index)
        return diagnostics_table(chain, burn_in=burn_in)

    def corner(self, subband: str, burn_in: int = 0, max_draws: Optional[int] = None,
               source_index: int = 0, **kwargs):
        from .visualization import plot_corner
        chain = self.repo.get_chain(subband, max_draws=max_draws, source_index=source_index)
        return plot_corner(chain, burn_in=burn_in, source_index=source_index, **kwargs)

    def fstat_contour(self, subband: str, x: Optional[str] = None, y: Optional[str] = None, **kwargs):
        from .visualization import plot_fstat_contour
        return plot_fstat_contour(self.repo.get_fstat_grid(subband), x=x, y=y, **kwargs)

    # -- signal reconstruction (FR-06) ----------------------------------- #
    def reconstruction(self, subband: str):
        """The :class:`ReconstructionData` for a sub-band (raises
        ``FileNotFoundError`` when the products are unavailable)."""
        return self.repo.get_reconstruction(subband)

    def psd_residual(self, subband: str, **kwargs):
        """PSD residual view (FR-06) for a sub-band, from the repository data."""
        from .visualization.reconstruction import plot_psd_residual
        r = self.repo.get_reconstruction(subband)
        return plot_psd_residual(r.freq, r.observed_psd, r.model_psd, noise_psd=r.noise_psd,
                                 noise_evolution=r.noise_evolution, **kwargs)

    def waveform_overlay(self, subband: str, **kwargs):
        """Frequency-domain waveform overlay + match scores (FR-06) for a sub-band."""
        from .visualization.reconstruction import plot_waveform_overlay
        r = self.repo.get_reconstruction(subband)
        freq = r.overlay_freq if r.overlay_freq is not None else r.freq
        return plot_waveform_overlay(freq, r.observed_spectrum, r.recovered or [],
                                     injection=r.injection, psd=r.overlay_psd,
                                     labels=r.labels, **kwargs)
