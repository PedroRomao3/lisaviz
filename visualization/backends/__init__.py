"""Rendering backends (Strategy pattern).

A visualization module prepares backend-neutral, padding-free NumPy arrays from
domain objects and then delegates *rendering* to an :class:`RenderBackend`
strategy. Two interchangeable strategies are provided, one per module:

* :class:`PlotlyBackend`     -- interactive (the dashboard default);
* :class:`MatplotlibBackend` -- static, publication-quality export (FR-07).

Each backend implements the chart kinds it is suited to; unsupported kinds
raise ``NotImplementedError`` with a clear message rather than silently
degrading.
"""

from .base import LineSeries, RenderBackend, TracePanel
from .matplotlib_backend import MatplotlibBackend
from .plotly_backend import PlotlyBackend

__all__ = [
    "LineSeries",
    "TracePanel",
    "RenderBackend",
    "PlotlyBackend",
    "MatplotlibBackend",
]
