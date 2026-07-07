"""The static Matplotlib strategy: publication-quality export (FR-07).

Renders on the ``Agg`` canvas so it works headless (CI, batch export). The
Plotly-only interactive kinds (``scatter``, ``sky_mollweide``,
``sky_band_panels``) are inherited from the base class and raise
``NotImplementedError``.
"""

from __future__ import annotations

import numpy as np

from .base import RenderBackend


def _mpl_color(color):
    """Accept Plotly-style ``rgb()/rgba()`` strings in the Matplotlib backend.

    Series colours are authored backend-neutrally; Plotly takes ``rgba(...)``
    natively but Matplotlib needs 0-1 float tuples, so translate on the way in.
    """
    if isinstance(color, str) and color.startswith(("rgb(", "rgba(")):
        nums = color[color.index("(") + 1 : color.index(")")].split(",")
        vals = [float(n) for n in nums]
        rgb = [v / 255.0 for v in vals[:3]]
        return tuple(rgb + vals[3:4]) if len(vals) == 4 else tuple(rgb)
    return color


class MatplotlibBackend(RenderBackend):
    name = "matplotlib"

    def __init__(self):
        import matplotlib
        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
        self._plt = plt

    def save(self, fig, path: str) -> None:
        fig.savefig(path, dpi=200, bbox_inches="tight")

    def trace_panels(self, panels, burn_in=0, title=""):
        plt = self._plt
        if len(panels) <= 4:
            rows = len(panels)
            cols = 1
            fig, axes = plt.subplots(rows, cols, sharex=True, figsize=(8, 2.2 * rows))
            axes = np.atleast_1d(axes).flatten()
        else:
            cols = 3 if len(panels) >= 7 else 2
            rows = int(np.ceil(len(panels) / cols))
            fig, axes = plt.subplots(rows, cols, figsize=(12, 2.0 * rows))
            axes = np.atleast_1d(axes).flatten()

        for idx, panel in enumerate(panels):
            ax = axes[idx]
            for s in panel.series:
                ax.plot(s.x, s.y, color=s.color, lw=s.width, alpha=s.opacity, label=s.label)
            if burn_in > 0:
                ax.axvspan(0, burn_in, color="red", alpha=0.08)
            ax.set_ylabel(panel.name, fontsize=20, color="#333333")
            ax.tick_params(labelsize=16)
            ax.set_xlabel("draw", fontsize=16)

        for k in range(len(panels), len(axes)):
            axes[k].set_visible(False)

        # One shared legend (full-opacity proxies) so walkers are identifiable.
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            leg = axes[0].legend(handles, labels, fontsize=16, ncol=min(len(labels), 6),
                                 loc="upper right", framealpha=0.9)
            for line in leg.get_lines():
                line.set_alpha(1.0)
                line.set_linewidth(2.0)
        if title:
            fig.suptitle(title, fontsize=22)
        fig.tight_layout()
        return fig

    def autocorr_panels(self, curves, tau, title=""):
        plt = self._plt
        names = list(curves.keys())
        fig, axes = plt.subplots(len(names), 1, sharex=True, figsize=(8, 1.6 * len(names)))
        axes = np.atleast_1d(axes)
        for ax, n in zip(axes, names):
            rho = curves[n]
            ax.bar(np.arange(rho.size), rho, color="#888888", width=1.0)
            ax.axhline(0.0, color="black", lw=1)
            ax.set_ylabel("autocorr", fontsize=9, color="#333333")
            ax.set_title(f"{n}  (tau~{tau.get(n, float('nan')):.1f})", fontsize=9)
        axes[-1].set_xlabel("lag")
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    def corner(self, samples, order, truths=None, title=""):
        import corner

        order = [n for n in order if n in samples]
        data = np.column_stack([np.asarray(samples[n], dtype=float) for n in order])
        truth_vec = None
        if truths is not None:
            truth_vec = [truths.get(n, None) for n in order]
        ndim = data.shape[1]
        # Scale the canvas with parameter count and damp tick crowding so the
        # column titles (top) and shared tick labels (bottom) don't collide.
        fig = corner.corner(
            data, labels=list(order), truths=truth_vec,
            show_titles=True, title_fmt=".3g", color="#1f4e79",
            max_n_ticks=4, labelpad=0.1,
            label_kwargs=dict(fontsize=34), title_kwargs=dict(fontsize=27),
        )
        fig.set_size_inches(3.9 * ndim, 3.9 * ndim)
        axes = fig.get_axes()
        for i in range(ndim):
            diag_ax = axes[i * ndim + i]
            ax_title = diag_ax.get_title()
            if "=" in ax_title:
                parts = ax_title.split("=")
                new_title = parts[1].strip()
                if ax_title.startswith("$") and not new_title.startswith("$"):
                    new_title = "$" + new_title
                diag_ax.set_title(new_title, fontsize=27)

        for ax in axes:
            ax.tick_params(labelsize=23)
            for lab in ax.get_xticklabels():
                lab.set_rotation(35)
                lab.set_horizontalalignment("right")
        # Reserve headroom so the diagonal column titles clear the suptitle; the
        # smaller (low-ndim) canvases need proportionally more room.
        top = 0.94 if ndim >= 4 else 0.88
        fig.subplots_adjust(top=top, bottom=0.10, hspace=0.06, wspace=0.06)
        if title:
            fig.suptitle(title, y=0.995, fontsize=24, va="top")
        return fig

    def spectral_overlay(self, series, *, residual=None, axis_labels=("Frequency [Hz]", "ASD"),
                         log_x=True, log_y=True, title=""):
        plt = self._plt
        rows = 2 if residual is not None else 1
        fig, axes = plt.subplots(rows, 1, sharex=True, figsize=(8, 5),
                                 gridspec_kw={"height_ratios": [3, 1]} if rows == 2 else None)
        axes = np.atleast_1d(axes)
        for s in series:
            axes[0].plot(s.x, s.y, label=s.label, color=_mpl_color(s.color), lw=s.width,
                         ls={"dash": "--", "dot": ":"}.get(s.dash, "-"))
        if log_x:
            axes[0].set_xscale("log")
        if log_y:
            axes[0].set_yscale("log")
        axes[0].set_ylabel(axis_labels[1])
        axes[0].legend(fontsize=8)
        if residual is not None:
            axes[1].plot(residual[0], residual[1], color="#888888", lw=1)
            axes[1].axhline(1.0, color="black", lw=1)
            axes[1].set_ylabel("resid")
        axes[-1].set_xlabel(axis_labels[0])
        if title:
            fig.suptitle(title)
        fig.tight_layout()
        return fig

    def fstat_contour(self, x, y, z, *, axis_labels=("", ""), color_label="F-statistic",
                      title="", cmap=None, peak=None):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(6.5, 5))
        masked = np.ma.masked_invalid(np.asarray(z))
        cs = ax.contourf(x, y, masked, levels=14, cmap=cmap or "cividis")
        ax.contour(x, y, masked, levels=8, colors="white", linewidths=0.4, alpha=0.5)
        fig.colorbar(cs, ax=ax, label=color_label)
        if peak is not None:
            ax.plot(peak[0], peak[1], "x", color="#d62728", markersize=10, label="max F")
            ax.legend(fontsize=8)
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    def sky_density(self, hx, hy, hval, *, points=None, graticule=None,
                    color_label="", title="", cmap=None, hex_size=9, point_color="#ff2a2a"):
        plt = self._plt
        from matplotlib.collections import PolyCollection
        fig, ax = plt.subplots(figsize=(9, 5))
        if graticule is not None:
            for gx, gy in graticule:
                ax.plot(gx, gy, color="#dddddd", lw=0.5, zorder=0)
        hx = np.asarray(hx, dtype=float)
        hy = np.asarray(hy, dtype=float)
        # Draw the cells as a tessellation in DATA coordinates rather than as
        # point-sized markers, so the cells stay in scale with the grid at any
        # figure size. The pitch is recovered from the streamed lattice itself:
        # row pitch from the unique y centres, column pitch from the unique x
        # centres (staggered by half a column between rows).
        uy = np.unique(np.round(hy, 12))
        ux = np.unique(np.round(hx, 12))
        dy = float(np.median(np.diff(uy))) if uy.size > 1 else 0.05
        dx = 2.0 * float(np.median(np.diff(ux))) if ux.size > 1 else dy
        r = 0.62 * max(dx, dy)  # slight overlap so the field closes without gaps
        ang = np.radians([90, 150, 210, 270, 330, 30])
        unit = np.column_stack([np.cos(ang), np.sin(ang)]) * r
        verts = np.column_stack([hx, hy])[:, None, :] + unit[None, :, :]
        pc = PolyCollection(verts, array=np.asarray(hval, dtype=float),
                            cmap=cmap or "cividis", linewidths=0.0, zorder=1)
        ax.add_collection(pc)
        ax.set_xlim(hx.min() - 4 * r, hx.max() + 4 * r)
        ax.set_ylim(hy.min() - 4 * r, hy.max() + 4 * r)
        fig.colorbar(pc, ax=ax, label=color_label)
        if points is not None and len(points[0]):
            import matplotlib.patheffects as path_effects
            from matplotlib.lines import Line2D
            px, py, psize = points[0], points[1], points[2]
            # Red core, white ring, thin black outermost: the white ring lifts
            # the mark off the dark cells while the black rim keeps it visible
            # over the white sky. The black stroke is drawn under the white edge.
            ax.scatter(px, py, s=np.asarray(psize, dtype=float) * 1.8,
                       color=point_color, edgecolors="white", linewidths=0.7,
                       zorder=5,
                       path_effects=[path_effects.withStroke(linewidth=1.6,
                                                             foreground="#111111")])
            # Proxy handle so the legend swatch carries a black edge too,
            # otherwise the white ring fades into the legend background.
            handle = Line2D([], [], linestyle="", marker="o", markersize=5,
                            markerfacecolor=point_color, markeredgecolor="#111111",
                            markeredgewidth=0.6)
            ax.legend([handle], ["outlier marks kept"], fontsize=8, loc="upper left")
        ax.set_axis_off()
        ax.set_aspect("equal")
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    def density_panel(self, hx, hy, hval, *, points=None, axis_labels=("", ""),
                      log_x=False, log_y=True, color_label="source count", title="",
                      cmap=None, hex_size=9, point_color="#ff7f0e"):
        plt = self._plt
        fig, ax = plt.subplots(figsize=(8, 5))
        sc = ax.scatter(hx, hy, marker="h", s=(hex_size * 2.0) ** 2, c=hval,
                        cmap=cmap or "cividis", linewidths=0)
        fig.colorbar(sc, ax=ax, label=color_label)
        if points is not None and len(points[0]):
            px, py, psize = points[0], points[1], points[2]
            ax.scatter(px, py, s=psize, color=point_color, edgecolors="#222222",
                       linewidths=0.4, alpha=0.9, label="most-uncertain sources")
            ax.legend(fontsize=8, loc="upper right")
        if log_x:
            ax.set_xscale("log")
        if log_y:
            ax.set_yscale("log")
        ax.set_xlabel(axis_labels[0])
        ax.set_ylabel(axis_labels[1])
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    def sky_hex_panels(self, selections, *, xc, yc, cmap_density=None, cmap_freq=None,
                       point_color="#ff2a2a", title="", active=0, hexrows=110):
        """Two static sky panels (density, mean f0) drawn as a true hexagonal
        tessellation via PolyCollection. The fine streaming grid is *aggregated*
        (counts summed) into a coarser pointy-top hex lattice of ``hexrows``
        rows -- so millions of streamed sources render as crisp,
        visibly-hexagonal cells without ever holding the point cloud. Static (no
        dropdown): pick the band with ``active``."""
        plt = self._plt
        from matplotlib.collections import PolyCollection
        sel = selections[max(0, min(active, len(selections) - 1))]
        x0, x1, y0, y1 = float(xc[0]), float(xc[-1]), float(yc[0]), float(yc[-1])
        xext, yext = x1 - x0, y1 - y0
        # Pointy-top hex lattice: vertical pitch 1.5R, horizontal pitch sqrt(3)R,
        # odd rows offset half a column. Coarser than the grid so hexagons read.
        R = (yext / hexrows) / 1.5
        hpitch, vpitch = np.sqrt(3) * R, 1.5 * R
        ncol, nrow = int(np.ceil(xext / hpitch)) + 1, int(np.ceil(yext / vpitch)) + 1
        ang = np.radians([90, 150, 210, 270, 330, 30])
        unit = np.column_stack([np.cos(ang), np.sin(ang)]) * R  # pointy-top hexagon

        # Aggregate every populated grid cell into its hex cell (sum, not sample).
        GX, GY = np.meshgrid(xc, yc)
        cnt = sel["counts_grid"]
        occ = cnt > 0
        gx, gy, gc, gf = GX[occ], GY[occ], cnt[occ], sel["freqsum_grid"][occ]
        hrow = np.clip(np.round((gy - y0) / vpitch).astype(int), 0, nrow - 1)
        hcol = np.clip(np.round((gx - x0) / hpitch - (hrow % 2) * 0.5).astype(int), 0, ncol - 1)
        hcount = np.zeros((nrow, ncol))
        hfreq = np.zeros((nrow, ncol))
        np.add.at(hcount, (hrow, hcol), gc)
        np.add.at(hfreq, (hrow, hcol), gf)
        # Centres of every hex cell (for the dark filled sky inside the ellipse).
        RR, CC = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
        hcx = (x0 + (CC + (RR % 2) * 0.5) * hpitch).ravel()
        hcy = (y0 + RR * vpitch).ravel()
        ex, ey = 0.5 * xext, 0.5 * yext
        inside = ((hcx - (x0 + ex)) / ex) ** 2 + ((hcy - (y0 + ey)) / ey) ** 2 <= 1.0
        hc, hf = hcount.ravel(), hfreq.ravel()
        with np.errstate(divide="ignore", invalid="ignore"):
            logc = np.where(hc > 0, np.log10(np.maximum(hc, 1.0)),
                            np.where(inside, 0.0, np.nan))
            meanf = np.where(hc > 0, hf / np.maximum(hc, 1) * 1e3, np.nan)

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        fig.patch.set_facecolor("white")

        base_verts = np.column_stack([hcx[inside], hcy[inside]])[:, None, :] + unit[None, :, :]

        def panel(ax, val, cmap, label, clip_lo=None):
            # Dark base over the whole projected sky so empty cells read as sky,
            # not white page (keeps both panels filled).
            ax.add_collection(PolyCollection(base_verts, facecolors="#0a0f24", linewidths=0.0))
            keep = np.isfinite(val)
            verts = np.column_stack([hcx[keep], hcy[keep]])[:, None, :] + unit[None, :, :]
            pc = PolyCollection(verts, array=val[keep], cmap=cmap, linewidths=0.0)
            if clip_lo is not None:
                pc.set_clim(clip_lo, float(np.nanmax(val[keep])) if keep.any() else 1.0)
            ax.add_collection(pc)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect("equal")
            ax.set_facecolor("#0a0f24")
            ax.set_axis_off()
            fig.colorbar(pc, ax=ax, fraction=0.035, pad=0.02).set_label(label)

        panel(axes[0], logc, cmap_density or "magma", "log₁₀ source count", clip_lo=0.0)
        panel(axes[1], meanf, cmap_freq or "cividis", "mean f0 [mHz]")
        if len(sel["mx"]):
            # Outliers (least-confident sources, kept by the reservoir): a vivid
            # accent that is the opposite end of the sequential map, sized by
            # 1 - confidence so the most uncertain stand out most.
            axes[0].scatter(sel["mx"], sel["my"], s=(sel["msize"] * 1.6) ** 2,
                            c=point_color, edgecolors="white", linewidths=0.3,
                            zorder=5, label="outlier marks kept (1 − confidence)")
            axes[0].legend(fontsize=8, loc="upper right", framealpha=0.85)
        axes[0].set_title("Source density (log₁₀ count)")
        axes[1].set_title("Mean f0 [mHz]")
        if title:
            fig.suptitle(f"{title} — {sel['label']}")
        fig.tight_layout()
        return fig
