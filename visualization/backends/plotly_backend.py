"""The interactive Plotly strategy (the dashboard default).

Implements every chart kind of :class:`~lisaviz.visualization.backends.base.
RenderBackend` except the Matplotlib-only static hex panels. Supports a dark
variant so figures blend into a dark dashboard theme.
"""

from __future__ import annotations

import numpy as np

from .base import RenderBackend


class PlotlyBackend(RenderBackend):
    name = "plotly"

    def __init__(self, dark: bool = False):
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        self._go = go
        self._make_subplots = make_subplots
        self._dark = dark
        # reference-line / axis foreground colour that stays readable on either
        # background (dark dashboard card vs. white notebook/export page).
        self._fg = "#cccccc" if dark else "black"

    def save(self, fig, path: str) -> None:
        if path.endswith(".html"):
            fig.write_html(path)
        else:
            fig.write_image(path)

    def _layout(self, fig, title=""):
        if self._dark:
            # Transparent backgrounds so the figure blends into the Streamlit
            # dark theme card, with a light font so labels stay readable.
            fig.update_layout(
                template="plotly_dark", title=title,
                font=dict(color="#fafafa", size=11), margin=dict(l=60, r=30, t=50, b=50),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
        else:
            fig.update_layout(
                template="simple_white", title=title,
                font=dict(color="#333333", size=11), margin=dict(l=60, r=30, t=50, b=50),
            )
        return fig

    def trace_panels(self, panels, burn_in=0, title=""):
        go = self._go
        fig = self._make_subplots(rows=len(panels), cols=1, shared_xaxes=True,
                                  subplot_titles=[p.name for p in panels], vertical_spacing=0.03)
        for i, panel in enumerate(panels, start=1):
            for s in panel.series:
                fig.add_trace(go.Scatter(x=s.x, y=s.y, mode="lines", name=s.label,
                                         line=dict(color=s.color, width=s.width), opacity=s.opacity,
                                         legendgroup=s.label, showlegend=(i == 1)), row=i, col=1)
            if burn_in > 0:
                fig.add_vrect(x0=0, x1=burn_in, fillcolor="red", opacity=0.08,
                              line_width=0, row=i, col=1)
            fig.update_yaxes(title_text=panel.name, row=i, col=1)
        fig.update_xaxes(title_text="draw", row=len(panels), col=1)
        # Taller panels relieve the cramped stack. The legend is styled as a
        # bordered box pinned to the side so it is obvious that walkers can be
        # toggled (click to show/hide, double-click to isolate) -- the title acts
        # as the affordance hint.
        legend_bg = "rgba(28,31,41,0.75)" if self._dark else "rgba(255,255,255,0.85)"
        fig.update_layout(
            height=230 * len(panels),
            legend=dict(
                title=dict(text="Walkers — click to show/hide"),
                bgcolor=legend_bg, bordercolor=self._fg, borderwidth=1,
                x=1.02, xanchor="left", y=1.0, yanchor="top",
            ),
        )
        return self._layout(fig, title)

    def autocorr_panels(self, curves, tau, title=""):
        go = self._go
        names = list(curves.keys())
        fig = self._make_subplots(rows=len(names), cols=1, shared_xaxes=True,
                                  subplot_titles=[f"{n}  (tau~{tau.get(n, float('nan')):.1f})" for n in names],
                                  vertical_spacing=0.03)
        for i, n in enumerate(names, start=1):
            rho = curves[n]
            fig.add_trace(go.Bar(x=np.arange(rho.size), y=rho, marker_color="#888888",
                                 showlegend=False), row=i, col=1)
            fig.add_hline(y=0.0, line_color=self._fg, line_width=1, row=i, col=1)
            fig.update_yaxes(title_text="autocorr", row=i, col=1)
        fig.update_xaxes(title_text="lag", row=len(names), col=1)
        fig.update_layout(height=150 * len(names))
        return self._layout(fig, title)

    def scatter(self, x, y, color_values, sizes, *, color_label="", axis_labels=("", ""),
                log_x=False, log_y=False, hovertext=None, title="", cmap=None, customdata=None):
        go = self._go
        fig = go.Figure(go.Scatter(
            x=x, y=y, mode="markers", text=hovertext, customdata=customdata,
            marker=dict(size=sizes, color=color_values, colorscale=cmap or "Cividis",
                        showscale=True, colorbar=dict(title=color_label), opacity=0.75,
                        line=dict(width=0)),
        ))
        fig.update_xaxes(title_text=axis_labels[0], type="log" if log_x else "linear")
        fig.update_yaxes(title_text=axis_labels[1], type="log" if log_y else "linear")
        return self._layout(fig, title)

    def sky_mollweide(self, x, y, color_values, sizes, *, graticule=None, color_label="",
                      hovertext=None, highlight=None, title="", cmap=None, customdata=None):
        go = self._go
        fig = go.Figure()
        if graticule is not None:
            for gx, gy in graticule:
                fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines",
                                         line=dict(color="#dddddd", width=0.6),
                                         hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="markers", text=hovertext, name="sources",
            showlegend=False,  # single series -> a legend entry only obscures the map
            customdata=customdata,  # carries f0 [Hz] so a click can resolve the band
            marker=dict(size=sizes, color=color_values, colorscale=cmap or "Cividis",
                        showscale=True,
                        colorbar=dict(title=dict(text=color_label, side="right")),
                        opacity=0.8, line=dict(width=0)),
        ))
        if highlight is not None and len(highlight[0]):
            fig.add_trace(go.Scatter(x=highlight[0], y=highlight[1], mode="markers",
                                     marker=dict(size=14, color="rgba(0,0,0,0)",
                                                 line=dict(color="#d62728", width=2)),
                                     name="flagged", hoverinfo="skip"))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        return self._layout(fig, title)

    def spectral_overlay(self, series, *, residual=None, axis_labels=("Frequency [Hz]", "ASD"),
                         log_x=True, log_y=True, title=""):
        go = self._go
        rows = 2 if residual is not None else 1
        fig = self._make_subplots(rows=rows, cols=1, shared_xaxes=True,
                                  row_heights=[0.7, 0.3] if rows == 2 else [1.0],
                                  vertical_spacing=0.05)
        for s in series:
            fig.add_trace(go.Scatter(x=s.x, y=s.y, mode="lines", name=s.label,
                                     line=dict(color=s.color, width=s.width, dash=s.dash)), row=1, col=1)
        fig.update_yaxes(title_text=axis_labels[1], type="log" if log_y else "linear", row=1, col=1)
        if residual is not None:
            fig.add_trace(go.Scatter(x=residual[0], y=residual[1], mode="lines",
                                     line=dict(color="#888888", width=1), name="residual"), row=2, col=1)
            fig.add_hline(y=1.0, line_color=self._fg, line_width=1, row=2, col=1)
            fig.update_yaxes(title_text="resid", row=2, col=1)
        fig.update_xaxes(title_text=axis_labels[0], type="log" if log_x else "linear", row=rows, col=1)
        return self._layout(fig, title)

    def fstat_contour(self, x, y, z, *, axis_labels=("", ""), color_label="F-statistic",
                      title="", cmap=None, peak=None):
        go = self._go
        fig = go.Figure(go.Contour(
            x=x, y=y, z=z, colorscale=cmap or "Cividis", connectgaps=True,
            colorbar=dict(title=dict(text=color_label, side="right")),
            contours=dict(coloring="heatmap"),
        ))
        if peak is not None:
            fig.add_trace(go.Scatter(x=[peak[0]], y=[peak[1]], mode="markers",
                                     marker=dict(symbol="x", size=11, color="#d62728"),
                                     name="max F", hoverinfo="name"))
        fig.update_xaxes(title_text=axis_labels[0])
        fig.update_yaxes(title_text=axis_labels[1])
        return self._layout(fig, title)

    def sky_density(self, hx, hy, hval, *, points=None, graticule=None,
                    color_label="", title="", cmap=None, hex_size=9, point_color="#ff2a2a"):
        go = self._go
        fig = go.Figure()
        if graticule is not None:
            for gx, gy in graticule:
                fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines",
                                         line=dict(color="#dddddd", width=0.6),
                                         hoverinfo="skip", showlegend=False))
        # The colorbar already names this trace's meaning, so it stays out of the
        # legend (a "density" legend entry would collide with the colorbar).
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers", name="density", hoverinfo="skip",
            showlegend=False,
            marker=dict(symbol="hexagon", size=hex_size, color=hval,
                        colorscale=cmap or "Cividis", showscale=True,
                        colorbar=dict(title=dict(text=color_label, side="right")),
                        line=dict(width=0))))
        if points is not None and len(points[0]):
            px, py, psize = points[0], points[1], points[2]
            fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="outlier marks kept",
                                     marker=dict(size=psize, color=point_color,
                                                 line=dict(width=0.8, color="white"), opacity=0.95)))
            # Anchor the legend in the empty top-left corner of the Mollweide
            # ellipse, away from the colorbar on the right.
            legend_bg = "rgba(28,31,41,0.75)" if self._dark else "rgba(255,255,255,0.85)"
            fig.update_layout(legend=dict(x=0.01, y=0.98, bgcolor=legend_bg,
                                          bordercolor="#cccccc", borderwidth=1))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1)
        return self._layout(fig, title)

    def sky_band_panels(self, selections, *, xc, yc, graticule=None, title="",
                        cmap_density=None, cmap_freq=None, point_color="#ff2a2a",
                        active=0):
        """Two linked sky panels -- left: source density (log10 count), right:
        mean f0 -- with a dropdown that filters to a single f0 sub-band or 'All
        bands'. The density is a *filled* heatmap of the streaming count grid (no
        inter-marker gaps); empty cells are NaN and render transparent. Each
        selection carries its own density/frequency/outlier traces; the dropdown
        toggles which group is visible, and the right panel's axes match the left
        so panning either pans both."""
        go = self._go
        fig = self._make_subplots(
            rows=1, cols=2, horizontal_spacing=0.13,
            subplot_titles=("Source density (log₁₀ count)", "Mean f0 [mHz]"))

        hover = ("count=%{customdata[0]:,.0f}<br>"
                 "mean f0=%{customdata[1]:.3f} mHz<extra></extra>")
        groups = []
        for sel in selections:
            i0 = len(fig.data)
            cdata = np.dstack([sel["count"], sel["meanfreq"]])  # (ny, nx, 2)
            # left: filled log-count density raster
            fig.add_trace(go.Heatmap(
                z=sel["logcount"], x=xc, y=yc, customdata=cdata, visible=False,
                colorscale=cmap_density or "Magma", hoverongaps=False, zmin=0.0,
                colorbar=dict(title=dict(text="log₁₀ count", side="right"), x=0.43, len=0.85),
                hovertemplate=hover), row=1, col=1)
            # right: filled mean-frequency raster
            fig.add_trace(go.Heatmap(
                z=sel["meanfreq"], x=xc, y=yc, customdata=cdata, visible=False,
                colorscale=cmap_freq or "Cividis", hoverongaps=False,
                colorbar=dict(title=dict(text="f0 [mHz]", side="right"), x=1.0, len=0.85),
                hovertemplate=hover), row=1, col=2)
            # outlier marks on the density panel (sparse-region sources)
            fig.add_trace(go.Scatter(
                x=sel["mx"], y=sel["my"], mode="markers", visible=False, showlegend=False,
                name="outlier marks kept", hoverinfo="skip",
                marker=dict(size=sel["msize"], color=point_color,
                            line=dict(width=0.3, color="#333333"), opacity=0.9)), row=1, col=1)
            groups.append((i0, i0 + 1, i0 + 2))

        # Graticule drawn last so the meridians/parallels sit on top of the rasters.
        grat_idx = []
        if graticule is not None:
            for gx, gy in graticule:
                for col in (1, 2):
                    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", hoverinfo="skip",
                                             line=dict(color="#888888", width=0.5),
                                             showlegend=False), row=1, col=col)
                    grat_idx.append(len(fig.data) - 1)

        total = len(fig.data)
        buttons = []
        for k, sel in enumerate(selections):
            vis = [False] * total
            for gi in grat_idx:
                vis[gi] = True
            for t in groups[k]:
                vis[t] = True
            buttons.append(dict(label=sel["label"], method="update",
                                args=[{"visible": vis},
                                      {"title": f"{title} — {sel['label']}"}]))
        active = max(0, min(active, len(selections) - 1))
        for gi in grat_idx:
            fig.data[gi].visible = True
        for t in groups[active]:
            fig.data[t].visible = True  # default selection shown on load

        # Link pan/zoom across the two panels and keep each on an equal aspect.
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False, scaleanchor="x", scaleratio=1, row=1, col=1)
        fig.update_yaxes(visible=False, scaleanchor="x2", scaleratio=1, row=1, col=2)
        fig.update_xaxes(matches="x", row=1, col=2)
        fig.update_yaxes(matches="y", row=1, col=2)
        fig.add_annotation(text="f0 sub-band:", showarrow=False, xref="paper", yref="paper",
                           x=0.62, y=1.22, xanchor="right", yanchor="middle",
                           font=dict(size=11))
        fig.update_layout(
            updatemenus=[dict(active=active, x=0.64, y=1.22, xanchor="left", yanchor="middle",
                              buttons=buttons, showactive=True)],
            margin=dict(l=20, r=70, t=110, b=20))
        return self._layout(fig, f"{title} — {selections[active]['label']}")

    def density_panel(self, hx, hy, hval, *, points=None, axis_labels=("", ""),
                      log_x=False, log_y=True, color_label="source count", title="",
                      cmap=None, hex_size=9, point_color="#ff7f0e"):
        go = self._go
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hx, y=hy, mode="markers", name="density", hoverinfo="skip", showlegend=False,
            marker=dict(symbol="hexagon", size=hex_size, color=hval,
                        colorscale=cmap or "Cividis", showscale=True,
                        colorbar=dict(title=dict(text=color_label, side="right")),
                        line=dict(width=0))))
        if points is not None and len(points[0]):
            px, py, psize = points[0], points[1], points[2]
            fig.add_trace(go.Scatter(x=px, y=py, mode="markers", name="most-uncertain sources",
                                     marker=dict(size=psize, color=point_color,
                                                 line=dict(width=0.4, color="#222222"), opacity=0.9)))
        fig.update_xaxes(title_text=axis_labels[0], type="log" if log_x else "linear")
        fig.update_yaxes(title_text=axis_labels[1], type="log" if log_y else "linear")
        return self._layout(fig, title)
