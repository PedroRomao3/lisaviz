#!/usr/bin/env python3
"""Generate the thesis/diagnostic figure set into ``figures/`` (PNG).

Every figure is produced by a public lisaviz function (the library is the
artifact); this script only wires data -> plot -> PNG. Run after installing
the package:

    python examples/make_figures.py    # writes to examples/figures/

Outputs (figures/):
  1  straight_line_trace.png   validation trace (m, c)            [val]
  2  straight_line_corner.png  validation corner (2-param)        [val]
  3  sky_map_gb.png            GB sky map                         FR-02
  4  gb_corner.png             4-param GB corner                  FR-03
  5  gb_trace.png              GB trace, converged vs stuck       FR-04
  6  convergence_flag.png      flagged non-converging bands       FR-05
  7  psd_residual.png          PSD residual                       FR-06
  8  waveform_overlay.png      waveform overlay + match scores    FR-06
  9  fstat_contour.png         F-statistic landscape              FR-09
 10  dashboard.png             dashboard layout (composed mock)
 11  overdraw_demo.png         naive scatter vs streaming sky map ch6
"""

from __future__ import annotations

import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import FancyBboxPatch

from lisaviz import LISAViz
from lisaviz.domain import convergence
from lisaviz.domain.models import GBChain, MCMCChain
from lisaviz.visualization import (
    MatplotlibBackend,
    PlotlyBackend,
    plot_fstat_contour,
    plot_sky_map,
    plot_traces,
)
from lisaviz.visualization.reconstruction import plot_psd_residual, plot_waveform_overlay
from lisaviz.validation import run_straight_line
from lisaviz.validation.straight_line import plot_validation

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

mpl = MatplotlibBackend()       # light, publication-quality (static export)
plotly_light = PlotlyBackend()  # white background for standalone plotly PNGs
plotly_dark = PlotlyBackend(dark=True)  # dark/transparent for the dashboard mock


def save_mpl(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  wrote {name}")
    return path


def save_plotly(fig, name, width=900, height=600, scale=2):
    path = os.path.join(OUT, name)
    fig.write_image(path, width=width, height=height, scale=scale)
    print(f"  wrote {name}")
    return path


# --------------------------------------------------------------------------- #
# 1 + 2  Straight-line validation (trace + corner) -- known ground truth
# --------------------------------------------------------------------------- #
def fig_validation():
    print("[1,2] straight-line validation")
    result = run_straight_line(m_true=2.0, c_true=-3.0, seed=1)
    trace_fig, corner_fig = plot_validation(result, backend=mpl)
    save_mpl(trace_fig, "straight_line_trace.pdf")
    save_mpl(corner_fig, "straight_line_corner.pdf")


# --------------------------------------------------------------------------- #
# 3  GB sky map (FR-02) -- Mollweide, colour=f0, size=1-confidence
# --------------------------------------------------------------------------- #
def fig_sky_map(viz):
    print("[3] GB sky map (FR-02)")
    catalog = viz.repo.get_catalog()
    fig = plot_sky_map(catalog, backend=plotly_light)
    save_plotly(fig, "sky_map_gb.png", width=1000, height=560)


# --------------------------------------------------------------------------- #
# 4  GB corner (FR-03) -- 4 physical intrinsic parameters
# --------------------------------------------------------------------------- #
def fig_gb_corner(viz):
    print("[4] GB corner (FR-03)")
    band = viz.subbands()[0].label
    fig = viz.corner(band, burn_in=300, max_draws=3000, backend=mpl)
    save_mpl(fig, "gb_corner.pdf")


# --------------------------------------------------------------------------- #
# 5  GB trace (FR-04) -- converged walkers vs a stuck (flagged) walker
# --------------------------------------------------------------------------- #
def _trace_chain_converged_vs_stuck(band, n_draws=2500, seed=7):
    """Build a multi-walker chain where most walkers converge to a shared
    target and two get stuck off-target -- the case the trace plot exposes."""
    rng = np.random.default_rng(seed)
    pars = ("amp_log10", "fr", "fdot", "dec_sin", "alpha", "iota_cos", "phiL", "phiR")
    targets = {p: float(rng.uniform(0.4, 0.6)) for p in pars}
    t = np.linspace(0, 1, n_draws)
    decay = np.exp(-4.0 * t)
    walkers = []
    n_walk = 1
    stuck = []  # no stuck walkers, as there is only 1 collapsed walker
    for w in range(n_walk):
        samples = {}
        for p in pars:
            start = float(rng.uniform(0.0, 1.0))
            wander = rng.normal(0, 0.02, n_draws).cumsum() * 0.01
            if w in stuck:
                # stuck: stays near its (wrong) start, slow drift, never mixes
                off = targets[p] + (0.35 if w == 3 else -0.3)
                series = np.clip(off + 0.02 * rng.standard_normal(n_draws).cumsum() * 0.05, 0, 1)
            else:
                series = targets[p] + (start - targets[p]) * decay + wander
            samples[p] = np.clip(series, 0, 1)
        if w in stuck:
            loglik = -995.0 + 5.0 * (1 - decay) + rng.normal(0, 1.2, n_draws)  # lower plateau
        else:
            loglik = -950.0 + 25.0 * (1 - decay) + rng.normal(0, 1.0, n_draws)
        walkers.append(GBChain(walker_id=w, samples=samples, log_likelihood=loglik, band=band))
    return MCMCChain(walkers=walkers, band=band), sorted(stuck)


def fig_gb_trace(viz):
    print("[5] GB trace converged vs stuck (FR-04)")
    band = viz.subbands()[0]
    chain, stuck = _trace_chain_converged_vs_stuck(band)
    fig = plot_traces(chain, burn_in=400, backend=mpl, flagged_walkers=stuck)
    fig.suptitle("GB trace (FR-04): single walker trajectory (flattened)", fontsize=24, y=0.99)
    fig.tight_layout()
    save_mpl(fig, "gb_trace.pdf")


# --------------------------------------------------------------------------- #
# 6  Convergence flag (FR-05) -- robust MAD outlier flag over per-band scores
# --------------------------------------------------------------------------- #
def fig_convergence_flag():
    print("[6] convergence flag (FR-05)")
    rng = np.random.default_rng(3)
    labels = [f"gb-{3.00 + 0.004 * i:.3f}" for i in range(14)]
    # most bands healthy (~-925), three anomalously low (non-converging)
    scores = {lab: float(-925 + rng.normal(0, 2.0)) for lab in labels}
    for bad in (3, 8, 11):
        scores[labels[bad]] = float(-962 + rng.normal(0, 1.5))
    report = convergence.flag_low_bands(scores, n_mad=3.0)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    vals = [scores[l] for l in labels]
    colors = ["#d62728" if report.is_flagged(l) else "#4f9bff" for l in labels]
    lo, hi = min(vals), max(vals)
    pad = 0.08 * (hi - lo)
    base = lo - pad  # anchor bars just below the lowest score so dips are visible
    ax.bar(range(len(labels)), [v - base for v in vals], bottom=base,
           color=colors, edgecolor="#222", linewidth=0.4)
    ax.set_ylim(base, hi + pad)
    ax.axhline(report.threshold, color="#d62728", ls="--", lw=1.2,
               label=f"flag threshold (median − 3·MAD) = {report.threshold:.1f}")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=10)
    ax.set_ylabel("tail-mean log L  (final 20%)", fontsize=12)
    ax.set_title("FR-05 convergence candidates — "
                 f"flagged: {', '.join(report.flagged)}", fontsize=13)
    ax.legend(fontsize=10, loc="lower right")
    ax.margins(x=0.01)
    fig.tight_layout()
    save_mpl(fig, "convergence_flag.pdf")


# --------------------------------------------------------------------------- #
# 7 + 8  Signal reconstruction (FR-06) -- PSD residual & waveform overlay+match
# --------------------------------------------------------------------------- #
def _gb_line(freq, f0, width, amp, phase=0.0):
    """A narrow complex GB-like spectral line (Gaussian envelope)."""
    env = amp * np.exp(-0.5 * ((freq - f0) / width) ** 2)
    return env * np.exp(1j * (phase + 2 * np.pi * (freq - f0) / (4 * width)))


def fig_waveform_overlay():
    print("[8] waveform overlay + match (FR-06)")
    rng = np.random.default_rng(11)
    freq = np.linspace(3.0005e-3, 3.0095e-3, 2000)
    f0, width = 3.005e-3, 1.1e-6  # broad enough that the line spans the window
    noise_amp = 1.2e-23
    psd = np.full_like(freq, noise_amp ** 2)
    injection = _gb_line(freq, f0, width, 2.2e-22)
    # Low-level noise so the observed data visibly *tracks* the injection line
    # (rather than sitting on a flat floor that never overlaps the model).
    noise = noise_amp * (rng.standard_normal(freq.size) + 1j * rng.standard_normal(freq.size))
    observed = injection + noise
    recovered = [
        _gb_line(freq, f0, width, 2.18e-22, phase=0.03),                  # good fit
        _gb_line(freq, f0 + 8e-7, width * 1.1, 1.95e-22, phase=0.4),      # offset fit
    ]
    fig = plot_waveform_overlay(freq, observed, recovered, injection=injection, psd=psd,
                                labels=["recovered (best)", "recovered (offset)"], backend=mpl)
    # Crop the y-range so the Gaussian tails (which plunge toward zero) don't
    # stretch the log axis down to ~1e-50 and dwarf the part that matters.
    ax = fig.axes[0]
    ax.set_ylim(5e-24, 4e-22)
    fig.set_size_inches(8, 5)
    save_mpl(fig, "waveform_overlay.pdf")


def fig_psd_residual():
    print("[7] PSD residual (FR-06)")
    rng = np.random.default_rng(5)
    freq = np.geomspace(3e-4, 1e-2, 4000)
    # LISA-like instrument noise: acceleration (low-f) + OMS (high-f) terms,
    # plus the galactic confusion bump that the global fit subtracts down.
    acc = 5.8e-42 * (1 + (4e-4 / freq) ** 2)
    oms = 3.6e-42 * (freq / 3e-3) ** 2
    instrument = acc + oms
    confusion = 1.4e-41 * np.exp(-((np.log(freq / 1.1e-3)) ** 2) / (2 * 0.55 ** 2))
    noise_psd = instrument + confusion
    # A forest of resolved GB lines sitting on the noise (what the model fits).
    rng_lines = np.random.default_rng(7)
    forest = np.zeros_like(freq)
    for _ in range(60):
        lf = float(np.exp(rng_lines.uniform(np.log(5e-4), np.log(8e-3))))
        amp = noise_psd[np.argmin(np.abs(freq - lf))] * rng_lines.uniform(3, 40)
        forest += amp * np.exp(-0.5 * ((freq - lf) / (lf * 6e-4)) ** 2)
    model_psd = noise_psd + forest
    # Periodogram estimates are exponentially distributed per bin (~100% scatter),
    # not a tidy few-percent band -- that is what makes a real PSD look noisy.
    observed_psd = model_psd * rng.exponential(1.0, freq.size)
    # Noise model evolving over global-fit iterations (confusion being dug out).
    noise_evo = [(f"noise iter {k}", instrument + confusion * (1 + 0.35 * (4 - k)))
                 for k in range(1, 5)]
    fig = plot_psd_residual(freq, observed_psd, model_psd, noise_psd=noise_psd,
                            noise_evolution=noise_evo, backend=mpl)
    # Residual of an exponential periodogram scatters over ~[0, several]; clip the
    # view so the median-1 band reads clearly.
    if len(fig.axes) > 1:
        fig.axes[1].set_ylim(0, 4)
    fig.set_size_inches(9, 5.8)
    save_mpl(fig, "psd_residual.pdf")


# --------------------------------------------------------------------------- #
# 9  F-statistic landscape (FR-09) -- profiled degeneracy contour
# --------------------------------------------------------------------------- #
def fig_fstat(viz):
    print("[9] F-statistic contour (FR-09)")
    band = viz.subbands()[0].label
    grid = viz.repo.get_fstat_grid(band)
    # Match the binning to the dec_sin grid resolution (40 unique values) so the
    # profiled surface fills every row -- a coarser/finer mismatch leaves empty
    # bins that render as white stripes.
    fig = plot_fstat_contour(grid, x="fr", y="dec_sin", bins=40, backend=mpl)
    ax = fig.axes[0]
    ax.xaxis.set_major_locator(plt.matplotlib.ticker.MaxNLocator(5))
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, -3), useMathText=True)
    ax.tick_params(axis="x", labelrotation=0)
    fig.set_size_inches(7.2, 5.4)
    save_mpl(fig, "fstat_contour.pdf")


# --------------------------------------------------------------------------- #
# 11  Overdraw demo (ch6) -- naive scatter vs the library's streaming sky map
# --------------------------------------------------------------------------- #
def fig_overdraw():
    """Left panel: the naive path (every source its own mark). Right panel: the
    SAME field through the library's actual bounded render path
    (StreamingSkyMap + MatplotlibBackend), the code path behind sky_map_gb.png,
    so the comparison shows the library's real output rather than a hand-drawn
    imitation of it."""
    print("[11] overdraw demo (ch6): naive scatter vs StreamingSkyMap render")
    from lisaviz.visualization.population import StreamingSkyMap, _project

    rng = np.random.default_rng(0)
    # Synthetic all-sky field in equatorial coordinates: three dense clusters
    # (a galactic-plane-like pile-up) + a sparse isotropic population.
    ra_parts, dec_parts = [], []
    for ra0, dec0, n_c, s in [(4.6, -0.35, 60_000, 0.25),
                              (1.2, 0.15, 45_000, 0.30),
                              (2.9, 0.55, 30_000, 0.18)]:
        ra_parts.append(rng.normal(ra0, s, n_c) % (2 * np.pi))
        dec_parts.append(np.clip(rng.normal(dec0, s * 0.6, n_c), -np.pi / 2, np.pi / 2))
    ra_parts.append(rng.uniform(0, 2 * np.pi, 4000))
    dec_parts.append(np.arcsin(rng.uniform(-1, 1, 4000)))
    ra = np.concatenate(ra_parts)
    dec = np.concatenate(dec_parts)
    n = ra.size
    freq = np.exp(rng.uniform(np.log(1e-4), np.log(1e-2), n))
    snr = np.exp(rng.uniform(np.log(5), np.log(50), n))
    conf = rng.uniform(0.2, 1.0, n)

    # Left: naive scatter of the projected points -- past saturation the ink
    # stops tracking density, which is the failure the reduction answers.
    x, y = _project(ra, dec)
    figL, axL = plt.subplots(figsize=(9, 5))
    axL.scatter(x, y, s=6, c="#1f4e79", alpha=0.35, edgecolors="none")
    axL.set_aspect("equal")
    axL.set_axis_off()
    axL.set_title(f"Naive scatter — {n:,} marks (overdraw hides structure)", fontsize=15)
    left = os.path.join(OUT, "_overdraw_left.png")
    figL.savefig(left, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figL)

    # Right: the SAME field streamed through the library's bounded aggregator
    # and rendered by the Matplotlib backend (identical path to sky_map_gb.png).
    agg = StreamingSkyMap(gridsize=110, max_points=400)
    for start in range(0, n, 50_000):
        sl = slice(start, min(start + 50_000, n))
        agg.add_block({"frequency": freq[sl], "right_ascension": ra[sl],
                       "declination": dec[sl], "snr": snr[sl], "confidence": conf[sl]})
    figR = agg.figure(backend=mpl)
    right = os.path.join(OUT, "_overdraw_right.png")
    figR.savefig(right, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(figR)

    # Composite the two panels side by side (pad to a common height).
    imgL, imgR = mpimg.imread(left), mpimg.imread(right)

    def pad_to(img, h):
        missing = h - img.shape[0]
        if missing <= 0:
            return img
        white = np.ones((missing, img.shape[1], img.shape[2]), dtype=img.dtype)
        return np.vstack([white[: missing // 2], img, white[missing // 2 :]])

    h = max(imgL.shape[0], imgR.shape[0])
    gap = np.ones((h, 30, imgL.shape[2]), dtype=imgL.dtype)
    combo = np.hstack([pad_to(imgL, h), gap, pad_to(imgR, h)])
    plt.imsave(os.path.join(OUT, "overdraw_demo.png"), combo)
    os.remove(left)
    os.remove(right)
    print("     wrote overdraw_demo.png (right panel = StreamingSkyMap render)")


# --------------------------------------------------------------------------- #
# 10  Dashboard (composed from the real dark-themed panels)
# --------------------------------------------------------------------------- #
def fig_dashboard(viz):
    print("[10] dashboard layout (composed mock; no browser available)")
    band_obj = viz.subbands()[0]
    band = band_obj.label
    catalog = viz.repo.get_catalog()

    # Render the real panels the dashboard shows, dark-themed, to temp PNGs.
    tmp = {}
    sky = plot_sky_map(catalog, backend=plotly_dark)
    tmp["sky"] = save_plotly(sky, "_dash_sky.png", width=820, height=460)

    chain, stuck = _trace_chain_converged_vs_stuck(band_obj)
    # plotly trace for the dark card
    trace = plot_traces(chain, burn_in=400, backend=plotly_dark, flagged_walkers=stuck)
    tmp["trace"] = save_plotly(trace, "_dash_trace.png", width=560, height=620)

    grid = viz.repo.get_fstat_grid(band)
    fstat = plot_fstat_contour(grid, x="fr", y="dec_sin", backend=plotly_dark)
    tmp["fstat"] = save_plotly(fstat, "_dash_fstat.png", width=560, height=420)

    # corner is matplotlib (renders as a light card in the real dashboard too)
    corner_path = os.path.join(OUT, "gb_corner.png")
    if not os.path.exists(corner_path):
        cfig = viz.corner(band, burn_in=300, max_draws=3000, backend=mpl)
        save_mpl(cfig, "gb_corner.png")

    BG, CARD, FG, ACCENT, WARN = "#0e1117", "#1c1f29", "#fafafa", "#4f9bff", "#e0a030"
    fig = plt.figure(figsize=(16, 9), facecolor=BG)

    # header
    fig.text(0.012, 0.955, "LISAViz — LISA Global Fit diagnostics (GB)",
             color=FG, fontsize=20, fontweight="bold", va="center")
    fig.add_artist(plt.Line2D([0.01, 0.99], [0.93, 0.93], color=ACCENT, lw=2))

    # sidebar card
    ax_sb = fig.add_axes([0.01, 0.04, 0.165, 0.86]); ax_sb.axis("off")
    ax_sb.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0.02,rounding_size=0.03",
                    transform=ax_sb.transAxes, facecolor=CARD, edgecolor="none"))
    sb_lines = [
        ("Data source", FG, 11, "bold"),
        ("Repository:  Mock", "#bbbbbb", 10, "normal"),
        ("", FG, 6, "normal"),
        ("Frequency sub-band (FR-08)", FG, 11, "bold"),
        (f"Active:  {band}", "#bbbbbb", 9, "normal"),
        ("Burn-in (draws):  400", "#bbbbbb", 10, "normal"),
        ("Max draws:  3000", "#bbbbbb", 10, "normal"),
        ("", FG, 6, "normal"),
        ("⚠ Convergence (FR-05)", WARN, 10, "bold"),
        (f"candidate: {band}", WARN, 8, "normal"),
    ]
    yy = 0.94
    for txt, col, sz, wt in sb_lines:
        ax_sb.text(0.07, yy, txt, color=col, fontsize=sz, fontweight=wt,
                   transform=ax_sb.transAxes, va="top")
        yy -= 0.052 if txt else 0.03

    def panel(rect, img_path, title):
        ax = fig.add_axes(rect); ax.axis("off")
        ax.text(0.0, 1.06, title, color=FG, fontsize=12, fontweight="bold",
                transform=ax.transAxes, va="bottom")
        ax.imshow(mpimg.imread(img_path))

    # main grid: sky (top-left wide), trace (right tall), corner (bottom-left), fstat (bottom-mid)
    panel([0.195, 0.50, 0.40, 0.38], tmp["sky"], "Population — Sky map (FR-02)")
    panel([0.625, 0.06, 0.355, 0.82], tmp["trace"], "MCMC health — Trace (FR-04)")
    panel([0.195, 0.06, 0.20, 0.36], corner_path, "Posterior (FR-03)")
    panel([0.410, 0.06, 0.185, 0.36], tmp["fstat"], "F-stat (FR-09)")

    save_mpl(fig, "dashboard.png")
    for p in tmp.values():  # clean temp panel PNGs
        try:
            os.remove(p)
        except OSError:
            pass


def main():
    print(f"Writing figures to {OUT}")
    viz = LISAViz.mock(n_catalog=1500, n_draws=3000, seed=42)
    fig_validation()
    fig_sky_map(viz)
    fig_gb_corner(viz)
    fig_gb_trace(viz)
    fig_convergence_flag()
    fig_psd_residual()
    fig_waveform_overlay()
    fig_fstat(viz)
    fig_overdraw()
    # fig_dashboard(viz)  # dashboard.png is now captured from the live app
    #   (see dashboard/app.py); the composed mock is retained above but unused.
    print("Done.")


if __name__ == "__main__":
    main()
