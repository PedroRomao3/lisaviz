"""Tests for the computer-assisted heuristic evaluation layer (Zhu and
Gumieniak). The automated checks run on declarative Plotly figures only; a
Matplotlib export is rejected. The library's own interactive figures should pass
every automatic check, and a deliberately bad figure should trip each one."""

import numpy as np
import pytest

pytest.importorskip("plotly")

import plotly.graph_objects as go  # noqa: E402

from lisaviz.ingestion.mock import InMemoryMockRepository  # noqa: E402
from lisaviz.visualization.backends import MatplotlibBackend, PlotlyBackend  # noqa: E402
from lisaviz.visualization.fstat import plot_fstat_contour  # noqa: E402
from lisaviz.visualization.health import plot_autocorrelation, plot_traces  # noqa: E402
from lisaviz.visualization.heuristic_eval import evaluate_figures, heuristic_eval  # noqa: E402
from lisaviz.visualization.population import (  # noqa: E402
    plot_sky_map_streaming,
    plot_waterfall_streaming,
)
from lisaviz.visualization.reconstruction import plot_psd_residual  # noqa: E402


@pytest.fixture(scope="module")
def repo():
    return InMemoryMockRepository(n_catalog=200, n_draws=300)


def _blocks(total, block, seed=0):
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


def test_rejects_matplotlib_figure(repo):
    """The method reads declarative plot JSON, so an imperative Matplotlib figure
    has nothing to introspect and must be refused with a clear message."""
    grid = repo.get_fstat_grid(repo.list_subbands()[0].label)
    mpl_fig = plot_fstat_contour(grid, backend=MatplotlibBackend())
    with pytest.raises(ValueError, match="declarative"):
        heuristic_eval(mpl_fig)


def test_accepts_json_string_and_dict(repo):
    grid = repo.get_fstat_grid(repo.list_subbands()[0].label)
    fig = plot_fstat_contour(grid, x="fr", y="alpha", backend=PlotlyBackend())
    assert heuristic_eval(fig.to_plotly_json()).passed
    assert heuristic_eval(fig.to_json()).passed  # JSON string path


def test_library_plotly_figures_pass_all_checks(repo):
    """Every interactive figure the dashboard composes passes all automatic
    checks: titled, labelled, perceptually-uniform colour, no raw overdraw."""
    band = repo.list_subbands()[0].label
    chain = repo.get_chain(band, max_draws=300)
    grid = repo.get_fstat_grid(band)
    recon = repo.get_reconstruction(band)
    figs = {
        "trace": plot_traces(chain, burn_in=50, backend=PlotlyBackend()),
        "autocorrelation": plot_autocorrelation(chain, max_lag=50, backend=PlotlyBackend()),
        "psd_residual": plot_psd_residual(recon.freq, recon.observed_psd, recon.model_psd,
                                          backend=PlotlyBackend()),
        "sky_map": plot_sky_map_streaming(_blocks(20_000, 5000), gridsize=64, max_points=300),
        "waterfall": plot_waterfall_streaming(_blocks(20_000, 5000), gridsize=64, max_points=300),
        "fstat_contour": plot_fstat_contour(grid, x="fr", y="alpha", backend=PlotlyBackend()),
    }
    reports = evaluate_figures(figs)
    for name, report in reports.items():
        assert report.passed, f"{name} unexpectedly warned: {report.summary()}"


def test_bad_figure_trips_every_check():
    """A rainbow, title-less, axis-less raw scatter above the overdraw threshold
    must raise the title, axis, colormap and overdraw warnings together."""
    n = 20_000
    rng = np.random.default_rng(0)
    fig = go.Figure(go.Scatter(
        x=rng.normal(size=n), y=rng.normal(size=n), mode="markers",
        marker=dict(color=rng.normal(size=n), colorscale="Jet", showscale=True),
    ))  # no title, no axis titles, no colorbar title, rainbow scale, 20k points
    report = heuristic_eval(fig)
    raised = {f.rule_id for f in report.warnings}
    assert {"AC-TITLE", "AC-AXES", "AC-COLORMAP", "AC-OVERDRAW", "AC-COLORBAR"} <= raised
    assert not report.passed


def test_size_encoding_emits_advice():
    """When mark size carries a data variable the tool surfaces the polarity
    reminder (advice, not a failure)."""
    fig = go.Figure(go.Scatter(
        x=[1, 2, 3], y=[1, 2, 3], mode="markers",
        marker=dict(size=[5, 10, 20], color=[1, 2, 3], colorscale="Cividis", showscale=True,
                    colorbar=dict(title="f0")),
    ))
    fig.update_layout(title="t")
    fig.update_xaxes(title_text="x")
    fig.update_yaxes(title_text="y")
    report = heuristic_eval(fig)
    assert report.passed  # advice does not fail the figure
    assert any(f.rule_id == "ADV-SIZE" for f in report.advice)
