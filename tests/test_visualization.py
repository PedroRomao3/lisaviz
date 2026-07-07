"""Visualization-layer tests: every plotting function returns a figure from
plain domain objects, both backends are interchangeable, diagnostics run via
the arviz-stats array interface."""

import numpy as np
import pytest

from lisaviz.ingestion.mock import InMemoryMockRepository
from lisaviz.visualization import stats
from lisaviz.visualization.backends import MatplotlibBackend, PlotlyBackend
from lisaviz.visualization.corner import plot_corner
from lisaviz.visualization.fstat import plot_fstat_contour
from lisaviz.visualization.health import plot_autocorrelation, plot_traces
from lisaviz.visualization.population import (
    StreamingSkyMap,
    StreamingSkyMapBands,
    StreamingWaterfall,
    plot_sky_map,
    plot_sky_map_bands,
    plot_sky_map_streaming,
    plot_waterfall,
    plot_waterfall_streaming,
)
from lisaviz.visualization.reconstruction import match, plot_psd_residual


@pytest.fixture(scope="module")
def repo():
    return InMemoryMockRepository(n_catalog=200, n_draws=300)


@pytest.fixture(scope="module")
def chain(repo):
    return repo.get_chain(repo.list_subbands()[0].label, max_draws=300)


def test_diagnostics_via_arviz_stats(chain):
    diag = stats.diagnostics(chain, burn_in=50)
    assert set(diag) == set(chain.param_names)
    assert np.isfinite(diag["fr"].ess) and diag["fr"].ess > 0
    assert np.isfinite(diag["fr"].rhat)  # multi-walker -> finite
    assert diag["fr"].rhat_is_split is False  # multi-walker, not a split fallback
    assert 0.0 <= stats.acceptance_fraction(chain) <= 1.0


def test_single_walker_uses_split_rhat():
    """A single-walker chain (the real GB F-stat case) yields a finite
    split-R-hat, clearly flagged -- never a silent multi-chain R-hat."""
    repo = InMemoryMockRepository(n_catalog=5, n_walkers=1, n_draws=400)
    chain = repo.get_chain(repo.list_subbands()[0].label)
    assert chain.n_walkers == 1
    diag = stats.diagnostics(chain)
    d = diag["fr"]
    assert d.rhat_is_split is True
    assert np.isfinite(d.rhat)


def test_trace_both_backends(chain):
    assert plot_traces(chain, burn_in=50, backend=PlotlyBackend()).__module__.startswith("plotly")
    assert plot_traces(chain, burn_in=50, backend=MatplotlibBackend()).__module__.startswith("matplotlib")


def test_autocorr(chain):
    assert plot_autocorrelation(chain, max_lag=50) is not None


def test_corner_transforms_to_physical(chain):
    fig = plot_corner(chain, burn_in=50, backend=MatplotlibBackend())
    assert fig.__module__.startswith("matplotlib")


def test_sky_and_waterfall(repo):
    cat = repo.get_catalog()
    assert plot_sky_map(cat) is not None
    assert plot_waterfall(cat, y_axis="SNR") is not None


def test_sky_map_density_reduction_large_catalog():
    repo = InMemoryMockRepository(n_catalog=8000, n_draws=10)
    fig = plot_sky_map(repo.get_catalog())  # exercises Splatterplots path
    assert fig is not None


def _synthetic_blocks(total, block, seed=0):
    rng = np.random.default_rng(seed)
    done = 0
    while done < total:
        n = min(block, total - done)
        yield ({
            "frequency": np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), n)),
            "right_ascension": rng.uniform(0, 2 * np.pi, n),
            "declination": np.clip(rng.normal(0, 0.3, n), -1.5, 1.5),
            "snr": np.exp(rng.uniform(np.log(5), np.log(50), n)),
            "confidence": rng.uniform(0, 1, n),
            "amplitude": np.exp(rng.uniform(np.log(1e-23), np.log(1e-21), n)),
        }, done, total)
        done += n


def test_streaming_sky_aggregator_is_bounded_and_renders():
    # The reservoir and grid stay fixed-size no matter how many sources stream in.
    agg = StreamingSkyMap(gridsize=64, max_points=500)
    for arrays, _s, _t in _synthetic_blocks(20_000, 1000):
        agg.add_block(arrays)
    assert agg.total_seen == 20_000
    assert agg._rx.size <= 500              # reservoir bounded
    assert agg.counts.shape == (64, 64)     # grid fixed
    assert int(agg.counts.sum()) == 20_000  # every source binned exactly once
    assert agg.figure(backend=PlotlyBackend()).__module__.startswith("plotly")
    assert agg.figure(backend=MatplotlibBackend()).__module__.startswith("matplotlib")


def test_plot_sky_map_streaming_helper():
    fig = plot_sky_map_streaming(_synthetic_blocks(5000, 777), gridsize=48, max_points=200)
    assert fig is not None


def test_streaming_sky_bands_is_bounded_and_renders():
    # Splitting by sub-band keeps memory O(nbands * grid): per-band count grids
    # plus per-band bounded reservoirs, every source counted once across bands.
    agg = StreamingSkyMapBands(gridsize=48, max_points=200)
    for arrays, _s, _t in _synthetic_blocks(20_000, 1000):
        agg.add_block(arrays)
    assert agg.total_seen == 20_000
    assert agg.counts.shape == (agg.nbands, 48, 48)         # one grid per band
    assert int(agg.counts.sum()) == 20_000                  # binned once across bands
    assert all(r.size <= 200 for r in agg._rx)              # every reservoir bounded
    sels = agg.selections()
    assert len(sels) == agg.nbands + 1                      # "All bands" + each band
    # "All bands" count grid (empty cells are NaN/0) still sums to every source.
    assert int(np.nansum(sels[0]["count"])) == 20_000
    assert agg.figure(backend=PlotlyBackend()).__module__.startswith("plotly")
    # Matplotlib renders the same grid as a hexagonal tessellation.
    assert agg.figure(backend=MatplotlibBackend()).__module__.startswith("matplotlib")


def test_plot_sky_map_bands_helper_active_band():
    fig = plot_sky_map_bands(_synthetic_blocks(5000, 777), gridsize=48, max_points=200, active=2)
    assert fig is not None


def test_streaming_waterfall_is_bounded_and_renders():
    # Same bounded-aggregator guarantee as the sky map, but in freq x amplitude.
    agg = StreamingWaterfall(y_axis="Amplitude", gridsize=64, max_points=500)
    for arrays, _s, _t in _synthetic_blocks(20_000, 1000):
        agg.add_block(arrays)
    assert agg.total_seen == 20_000
    assert agg._rx.size <= 500              # reservoir bounded
    assert agg.counts.shape == (64, 64)     # grid fixed
    assert int(agg.counts.sum()) == 20_000  # every source binned exactly once
    assert agg.figure(backend=PlotlyBackend()).__module__.startswith("plotly")
    assert agg.figure(backend=MatplotlibBackend()).__module__.startswith("matplotlib")


def test_plot_waterfall_streaming_helper_snr_axis():
    fig = plot_waterfall_streaming(_synthetic_blocks(5000, 123), y_axis="SNR",
                                   gridsize=48, max_points=200)
    assert fig is not None


def test_match_self_is_one():
    rng = np.random.default_rng(0)
    a = rng.standard_normal(128) + 1j * rng.standard_normal(128)
    psd = np.ones(128)
    assert match(a, a, psd, 1.0) == pytest.approx(1.0)


def test_psd_residual():
    f = np.linspace(1e-3, 2e-3, 100)
    assert plot_psd_residual(f, np.ones(100), np.ones(100)) is not None


def test_fstat_contour_both_backends(repo):
    grid = repo.get_fstat_grid(repo.list_subbands()[0].label)
    assert plot_fstat_contour(grid, backend=MatplotlibBackend()).__module__.startswith("matplotlib")
    assert plot_fstat_contour(grid, x="fr", y="alpha", backend=PlotlyBackend()).__module__.startswith("plotly")


def test_fstat_project_pair_profiles_max(repo):
    grid = repo.get_fstat_grid(repo.list_subbands()[0].label)
    xc, yc, z = grid.project_pair("fr", "dec_sin", bins=30)
    assert z.shape == (30, 30)
    assert np.nanmax(z) <= grid.fstat.max() + 1e-9  # profile is a max over hidden axes
