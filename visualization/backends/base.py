"""The abstract rendering strategy and the backend-neutral series containers.

A visualization module prepares padding-free NumPy arrays from domain objects
and hands them to a :class:`RenderBackend` together with lightweight
:class:`LineSeries` / :class:`TracePanel` descriptions. The backend owns every
library-specific rendering decision; the modules above it never import Plotly
or Matplotlib directly.

Each concrete backend implements the chart kinds it is suited to; unsupported
kinds raise ``NotImplementedError`` with a clear message rather than silently
degrading.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np


@dataclass
class LineSeries:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: str = "#1f77b4"
    width: float = 0.8
    dash: Optional[str] = None  # e.g. "dash", "dot"
    opacity: float = 1.0


@dataclass
class TracePanel:
    name: str
    series: List[LineSeries] = field(default_factory=list)
    is_integer: bool = False  # the nsource hyperparameter panel


class RenderBackend(ABC):
    name = "abstract"

    @abstractmethod
    def save(self, fig, path: str) -> None: ...

    def trace_panels(self, panels: List[TracePanel], burn_in: int = 0, title: str = ""):
        raise NotImplementedError(f"{self.name} backend has no trace_panels")

    def autocorr_panels(self, curves: Dict[str, np.ndarray], tau: Dict[str, float], title: str = ""):
        raise NotImplementedError(f"{self.name} backend has no autocorr_panels")

    def corner(self, samples: Dict[str, np.ndarray], order: Sequence[str],
               truths: Optional[Dict[str, float]] = None, title: str = ""):
        raise NotImplementedError(f"{self.name} backend has no corner")

    def scatter(self, x, y, color_values, sizes, *, color_label="", axis_labels=("", ""),
                log_x=False, log_y=False, hovertext=None, title="", cmap=None, customdata=None):
        raise NotImplementedError(f"{self.name} backend has no scatter")

    def sky_mollweide(self, x, y, color_values, sizes, *, graticule=None, color_label="",
                      hovertext=None, highlight=None, title="", cmap=None, customdata=None):
        raise NotImplementedError(f"{self.name} backend has no sky_mollweide")

    def spectral_overlay(self, series: List[LineSeries], *, residual=None, axis_labels=("Frequency [Hz]", "ASD"),
                         log_x=True, log_y=True, title=""):
        raise NotImplementedError(f"{self.name} backend has no spectral_overlay")

    def fstat_contour(self, x, y, z, *, axis_labels=("", ""), color_label="F-statistic",
                      title="", cmap=None, peak=None):
        raise NotImplementedError(f"{self.name} backend has no fstat_contour")

    def sky_density(self, hx, hy, hval, *, points=None, graticule=None,
                    color_label="", title="", cmap=None, hex_size=9, point_color="#ff2a2a"):
        raise NotImplementedError(f"{self.name} backend has no sky_density")

    def sky_band_panels(self, selections, *, xc, yc, graticule=None, title="",
                        cmap_density=None, cmap_freq=None, point_color="#ff2a2a", active=0):
        raise NotImplementedError(f"{self.name} backend has no sky_band_panels")

    def sky_hex_panels(self, selections, *, xc, yc, cmap_density=None, cmap_freq=None,
                       point_color="#ff2a2a", title="", active=0, hexrows=110):
        raise NotImplementedError(f"{self.name} backend has no sky_hex_panels")

    def density_panel(self, hx, hy, hval, *, points=None, axis_labels=("", ""),
                      log_x=False, log_y=True, color_label="source count", title="",
                      cmap=None, hex_size=9, point_color="#ff7f0e"):
        raise NotImplementedError(f"{self.name} backend has no density_panel")
