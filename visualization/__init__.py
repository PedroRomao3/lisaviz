"""Visualization layer (depends on Domain; hides the rendering backend)."""

from .backends import MatplotlibBackend, PlotlyBackend, RenderBackend
from .corner import plot_corner
from .cheatsheets import CheatSheet, cheatsheets_for, cheatsheets_text
from .fstat import plot_fstat_contour
from .health import diagnostics_table, plot_autocorrelation, plot_traces
from .heuristic_eval import (
    HeuristicFinding,
    HeuristicReport,
    evaluate_figures,
    heuristic_eval,
)
from .population import (
    StreamingSkyMap,
    StreamingWaterfall,
    plot_sky_map,
    plot_sky_map_streaming,
    plot_waterfall,
    plot_waterfall_streaming,
)
from .reconstruction import match, plot_psd_residual, plot_waveform_overlay

__all__ = [
    "RenderBackend",
    "PlotlyBackend",
    "MatplotlibBackend",
    "plot_traces",
    "plot_autocorrelation",
    "diagnostics_table",
    "plot_corner",
    "plot_fstat_contour",
    "plot_sky_map",
    "plot_sky_map_streaming",
    "StreamingSkyMap",
    "StreamingWaterfall",
    "plot_waterfall",
    "plot_waterfall_streaming",
    "plot_psd_residual",
    "plot_waveform_overlay",
    "match",
    "heuristic_eval",
    "evaluate_figures",
    "HeuristicReport",
    "HeuristicFinding",
    "cheatsheets_for",
    "cheatsheets_text",
    "CheatSheet",
]
