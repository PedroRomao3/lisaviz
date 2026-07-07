#!/usr/bin/env python3
"""End-to-end walkthrough of lisaviz on synthetic (mock) data.

Runs every diagnostic view through the public facade with no filesystem, no
HDF5 and no run data, then writes the results to ``demo_output/``: interactive
Plotly views as HTML, the corner plot as a Matplotlib PNG.

    python examples/demo.py
"""

from __future__ import annotations

import os

from lisaviz import LISAViz


def main() -> None:
    out = "demo_output"
    os.makedirs(out, exist_ok=True)

    viz = LISAViz.mock()
    band = viz.subbands()[0].label
    print(f"mock repository ready; first sub-band: {band}")

    # Population views (interactive Plotly -> HTML).
    viz.sky_map().write_html(os.path.join(out, "sky_map.html"))
    viz.waterfall().write_html(os.path.join(out, "waterfall.html"))

    # MCMC health: traces, autocorrelation, numeric diagnostics.
    viz.traces(band, burn_in=200).write_html(os.path.join(out, "traces.html"))
    viz.autocorrelation(band).write_html(os.path.join(out, "autocorr.html"))
    for name, d in viz.diagnostics(band, burn_in=200).items():
        rhat_kind = "split" if d.rhat_is_split else "between-walker"
        print(f"  {name:>10}: ESS={d.ess:8.1f}  R-hat={d.rhat:.3f} ({rhat_kind})  tau={d.tau:.1f}")

    # Posterior corner in physical units (static Matplotlib PNG).
    corner = viz.corner(band, burn_in=200)
    corner.savefig(os.path.join(out, "corner.png"), bbox_inches="tight", dpi=150)

    # Signal reconstruction (FR-06) from the mock's synthetic products.
    viz.psd_residual(band).write_html(os.path.join(out, "psd_residual.html"))
    viz.waveform_overlay(band).write_html(os.path.join(out, "waveform_overlay.html"))

    # F-statistic degeneracy contour (static Matplotlib PNG).
    fstat = viz.fstat_contour(band)
    fstat.savefig(os.path.join(out, "fstat_contour.png"), bbox_inches="tight", dpi=150)

    print(f"wrote all views to {out}/")


if __name__ == "__main__":
    main()
