"""Module 4 -- Population: catalog waterfall + interactive sky map (FR-02).

* Sky map colour = GW frequency (perceptually-uniform sequential colormap);
  mark size = ``1 - confidence`` (BIG = MORE uncertain).
* Equatorial (alpha, delta) -> Mollweide projection is delegated to
  ``astropy.coordinates`` / ``astropy.wcs`` (no hand-rolled transforms).
* At 15.5M-source scale a Splatterplots-style density reduction collapses dense
  regions into a hexagonally-binned density layer (Carr et al. 1987) while keeping
  isolated / most-uncertain / flagged sources as discrete points. The bounded-
  streaming aggregators bin into a fixed-size offset-row hexagonal lattice, so peak
  memory stays O(1) in the source count.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..domain.models import GBCatalog
from . import encoding
from .backends import PlotlyBackend, RenderBackend

# Above this source count the sky map switches on density reduction.
SPLATTER_THRESHOLD = 5000


def _mollweide_wcs():
    from astropy.wcs import WCS

    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---MOL", "DEC--MOL"]
    wcs.wcs.crval = [180.0, 0.0]
    wcs.wcs.crpix = [0.0, 0.0]
    wcs.wcs.cdelt = [-1.0, 1.0]  # degrees per pixel; only relative scale matters
    return wcs


def _project(ra_rad: np.ndarray, dec_rad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project equatorial coordinates to Mollweide plane pixels via astropy."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords = SkyCoord(ra=np.asarray(ra_rad) * u.rad, dec=np.asarray(dec_rad) * u.rad, frame="icrs")
    wcs = _mollweide_wcs()
    x, y = wcs.world_to_pixel(coords)
    return np.asarray(x), np.asarray(y)


def _graticule(n_lines: int = 7) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Meridian/parallel lines projected to the Mollweide plane."""
    lines: List[Tuple[np.ndarray, np.ndarray]] = []
    lat = np.linspace(-89.9, 89.9, 180)
    for lon0 in np.linspace(0, 360, n_lines, endpoint=False):
        x, y = _project(np.radians(np.full_like(lat, lon0)), np.radians(lat))
        lines.append((x, y))
    lon = np.linspace(0.01, 359.99, 360)
    for lat0 in np.linspace(-60, 60, 5):
        x, y = _project(np.radians(lon), np.radians(np.full_like(lon, lat0)))
        lines.append((x, y))
    return lines


def _hexbin(x: np.ndarray, y: np.ndarray, gridsize: int = 80):
    """Assign points to a **hexagonal** lattice (Carr et al. 1987).

    Each point is binned to the nearer of two interleaved rectangular lattices,
    which tiles the plane with hexagons. Returns ``(centers_x, centers_y, counts,
    per_point_count)`` where ``per_point_count`` is the occupancy of the hex each
    point landed in.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    dx = (xmax - xmin) / max(gridsize, 1) or 1.0
    dy = dx * np.sqrt(3.0)  # hexagon row spacing
    sx = (x - xmin) / dx
    sy = (y - ymin) / dy
    # Two candidate centres: the integer lattice and the half-offset lattice.
    i1, j1 = np.round(sx), np.round(sy)
    i2, j2 = np.floor(sx) + 0.5, np.floor(sy) + 0.5
    use2 = ((sx - i2) ** 2 + (sy - j2) ** 2) < ((sx - i1) ** 2 + (sy - j1) ** 2)
    ci = np.where(use2, i2, i1)
    cj = np.where(use2, j2, j1)
    keys = np.stack([ci, cj], axis=1)
    uniq, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    centers_x = xmin + uniq[:, 0] * dx
    centers_y = ymin + uniq[:, 1] * dy
    return centers_x, centers_y, counts, counts[inv]


def _density_reduction(x: np.ndarray, y: np.ndarray, gridsize: int = 80):
    """Splatterplots-style split: return a boolean mask of *isolated* points
    (kept as discrete marks) plus a **hexagonal** density field (centres +
    counts) for the dense regions."""
    cx, cy, counts, per_point = _hexbin(x, y, gridsize)
    # Isolated = its hexagon holds few neighbours.
    isolated = per_point <= max(2, np.percentile(per_point, 40))
    return isolated, (cx, cy, counts)


# --- Streaming hexagonal binning (fixed-size, offset-row lattice) ----------- #
def _data_to_index(v, edges, g, log):
    """Continuous lattice index in [0, g] for a value, linear or log axis."""
    lo, hi = float(edges[0]), float(edges[-1])
    if log:
        v = np.where(np.asarray(v, dtype=float) > 0, v, lo)
        return (np.log(v) - np.log(lo)) / (np.log(hi) - np.log(lo)) * g
    return (np.asarray(v, dtype=float) - lo) / (hi - lo) * g


def _index_to_data(idx, edges, g, log):
    """Inverse of :func:`_data_to_index`: lattice index back to data coordinate."""
    lo, hi = float(edges[0]), float(edges[-1])
    frac = np.asarray(idx, dtype=float) / g
    if log:
        return np.exp(np.log(lo) + frac * (np.log(hi) - np.log(lo)))
    return lo + frac * (hi - lo)


def _offset_hex_bin(cix, ciy, g):
    """Assign continuous indices to an offset-row hex cell ``(row, col)``. Odd
    rows are shifted half a column, so each point lands in one cell of a fixed
    ``(g, g)`` lattice -- bounded memory, every source counted exactly once."""
    row = np.clip(np.round(np.asarray(ciy, dtype=float)).astype(int), 0, g - 1)
    col = np.clip(np.round(np.asarray(cix, dtype=float) - (row % 2) * 0.5).astype(int), 0, g - 1)
    return row, col


def _hex_cells(counts, value_grid, xedges, yedges, log_x, log_y):
    """Centres (data space, odd rows shifted) and values of the non-empty hex
    cells, ready to draw as discrete hexagon marks."""
    rows, cols = np.nonzero(counts)
    g = counts.shape[0]
    hx = _index_to_data(cols + (rows % 2) * 0.5, xedges, g, log_x)
    hy = _index_to_data(rows.astype(float), yedges, g, log_y)
    return hx, hy, value_grid[rows, cols]


def _hex_px(gridsize):
    """A hexagon marker size (px) that roughly tiles a ~640px plot at gridsize."""
    return float(np.clip(640.0 / gridsize, 4.0, 16.0))


def _sky_plane_grid(gridsize: int, margin: float = 0.02):
    """Fixed Mollweide-plane bin edges covering the whole projected sky plus a
    small margin. Computed once from a coarse lon/lat probe grid (O(1)), so a
    streaming aggregator never needs a first pass over the data to find its
    extent. Returns ``(xedges, yedges, extent)`` with ``extent = (xmin, xmax,
    ymin, ymax)`` of the projected sky itself (margin excluded)."""
    lon, lat = np.meshgrid(np.linspace(0, 360, 73), np.linspace(-89.0, 89.0, 37))
    gx, gy = _project(np.radians(lon.ravel()), np.radians(lat.ravel()))
    mx = margin * (gx.max() - gx.min())
    my = margin * (gy.max() - gy.min())
    xedges = np.linspace(gx.min() - mx, gx.max() + mx, gridsize + 1)
    yedges = np.linspace(gy.min() - my, gy.max() + my, gridsize + 1)
    return xedges, yedges, (float(gx.min()), float(gx.max()), float(gy.min()), float(gy.max()))


def _keep_most_uncertain(x, y, conf, freq, max_points):
    """Truncate a reservoir to its ``max_points`` smallest-confidence entries
    (smallest confidence = most uncertain = most worth keeping visible)."""
    if conf.size > max_points:
        keep = np.argsort(conf)[:max_points]
        return x[keep], y[keep], conf[keep], freq[keep]
    return x, y, conf, freq


def _sparse_reservoir(rx, ry, counts, xedges, yedges, log_x, log_y):
    """Mask of reservoir points sitting in *low-density* hex cells. The dense
    interior is already shown by the hex field, so only the sparse-region
    outliers are worth drawing as discrete marks (the Splatterplot rule)."""
    g = counts.shape[0]
    rrow, rcol = _offset_hex_bin(_data_to_index(rx, xedges, g, log_x),
                                 _data_to_index(ry, yedges, g, log_y), g)
    local = counts[rrow, rcol]
    occupied = counts[counts > 0]
    thr = max(2.0, float(np.percentile(occupied, 40))) if occupied.size else np.inf
    return local <= thr


def plot_sky_map(
    catalog: GBCatalog,
    backend: Optional[RenderBackend] = None,
    flagged_ids: Optional[Sequence[str]] = None,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
):
    """Interactive Mollweide sky map of the catalog."""
    backend = backend or PlotlyBackend()
    src = catalog.sources
    if not src:
        return backend.sky_mollweide(np.array([]), np.array([]), np.array([]), np.array([]),
                                     graticule=_graticule(), color_label="f0 [mHz]")

    ra = np.array([s.binary.intrinsic.right_ascension for s in src])
    dec = np.array([s.binary.intrinsic.declination for s in src])
    freq = np.array([s.binary.intrinsic.f0 for s in src])
    snr = np.array([s.snr for s in src])
    conf = encoding.resolve_confidence([s.confidence for s in src], snr)
    sizes = encoding.confidence_sizes(conf)
    x, y = _project(ra, dec)

    ids = np.array([s.binary.source_id for s in src])
    flag_set = set(flagged_ids or [])
    hl_mask = np.array([sid in flag_set for sid in ids])

    # Shared colour scale in physical units (mHz for readability).
    color_values = freq * 1e3
    hovertext = [
        f"{ids[i]}<br>f0={freq[i]*1e3:.4f} mHz<br>SNR={snr[i]:.1f}<br>conf={conf[i]:.2f}"
        for i in range(len(src))
    ]

    keep = np.ones(len(src), dtype=bool)
    if len(src) > SPLATTER_THRESHOLD:
        isolated, _ = _density_reduction(x, y)
        # Keep isolated points, high-confidence-uncertain (large) points and flagged.
        keep = isolated | (sizes > np.percentile(sizes, 90)) | hl_mask

    highlight = (x[hl_mask], y[hl_mask])
    return backend.sky_mollweide(
        x[keep], y[keep], color_values[keep], sizes[keep],
        graticule=_graticule(), color_label="f0 [mHz]",
        hovertext=[hovertext[i] for i in np.where(keep)[0]],
        highlight=highlight, title=f"Sky map ({keep.sum()}/{len(src)} shown)",
        cmap=encoding.FREQUENCY_CMAP_PLOTLY,
        customdata=freq[keep],  # f0 [Hz] per point -> a click resolves the band
    )


class StreamingSkyMap:
    """Bounded-memory sky-map aggregator. Consumes the catalog one row-block at a
    time and accumulates into a *fixed-size* Mollweide density grid plus a
    *bounded* reservoir of the most-uncertain sources (kept as discrete points).
    Peak memory is O(block + grid + reservoir), independent of catalog size --
    the streaming form of the Splatterplots idea.
    """

    def __init__(self, gridsize: int = 150, max_points: int = 800):
        self.gridsize = gridsize
        self.max_points = max_points
        self.counts = np.zeros((gridsize, gridsize))
        self.freqsum = np.zeros((gridsize, gridsize))
        self._rx = np.empty(0)
        self._ry = np.empty(0)
        self._rconf = np.empty(0)
        self._rfreq = np.empty(0)
        self.total_seen = 0
        self._fmin = np.inf
        self._fmax = -np.inf
        self._xedges, self._yedges, _ = _sky_plane_grid(gridsize)
        self._xc = 0.5 * (self._xedges[:-1] + self._xedges[1:])
        self._yc = 0.5 * (self._yedges[:-1] + self._yedges[1:])

    def add_block(self, arrays: dict) -> None:
        n = arrays["frequency"].size
        if n == 0:
            return
        self.total_seen += n
        freq = arrays["frequency"]
        snr = arrays.get("snr", np.full(n, np.nan))
        conf = encoding.resolve_confidence(arrays.get("confidence", np.full(n, np.nan)), snr)
        x, y = _project(arrays["right_ascension"], arrays["declination"])

        cix = _data_to_index(x, self._xedges, self.gridsize, False)
        ciy = _data_to_index(y, self._yedges, self.gridsize, False)
        row, col = _offset_hex_bin(cix, ciy, self.gridsize)
        np.add.at(self.counts, (row, col), 1.0)
        np.add.at(self.freqsum, (row, col), freq)
        self._fmin = min(self._fmin, float(np.nanmin(freq)))
        self._fmax = max(self._fmax, float(np.nanmax(freq)))

        # Keep only the most-uncertain (largest 1-confidence) points, bounded.
        self._rx, self._ry, self._rconf, self._rfreq = _keep_most_uncertain(
            np.concatenate([self._rx, x]), np.concatenate([self._ry, y]),
            np.concatenate([self._rconf, conf]), np.concatenate([self._rfreq, freq]),
            self.max_points)

    def figure(self, backend: Optional[RenderBackend] = None, log_counts: bool = True):
        backend = backend or PlotlyBackend()
        # Colour the hexes by SOURCE COUNT -- that is the whole point of the hex
        # binning: it shows where sources pile up (the Galactic structure), which
        # overdraw in a naive scatter would hide. Frequency is a per-source
        # attribute that does not reduce to a single hex value without losing the
        # density signal, so it is not the colour channel here (see freqsum, kept
        # for an optional second frequency view).
        cmap = encoding.DENSITY_CMAP if backend.name == "matplotlib" else encoding.DENSITY_CMAP_PLOTLY
        hx, hy, hval = _hex_cells(self.counts, self.counts, self._xedges, self._yedges, False, False)
        # Only sparse-region outliers are kept as discrete marks (the dense
        # interior is already the hex field), sized small so they sit on top.
        m = _sparse_reservoir(self._rx, self._ry, self.counts, self._xedges, self._yedges, False, False)
        sizes = encoding.confidence_sizes(self._rconf[m], min_px=3.0, max_px=10.0)
        color_label = "source count"
        if log_counts:
            # Counts span orders of magnitude between a sparse cell and the
            # Galactic bulge, which a linear colour scale flattens to black.
            hval = np.log10(np.maximum(np.asarray(hval, dtype=float), 1.0))
            color_label = "log₁₀ source count"
        return backend.sky_density(
            hx, hy, hval,
            points=(self._rx[m], self._ry[m], sizes),
            graticule=_graticule(), color_label=color_label, hex_size=_hex_px(self.gridsize),
            title=f"Sky map (streaming, {self.total_seen} sources, grid {self.gridsize})",
            cmap=cmap,
        )


def default_band_edges_mhz() -> "np.ndarray":
    """Log-spaced f0 sub-band edges (mHz) spanning the GB population."""
    return np.logspace(np.log10(0.1), np.log10(30.0), 7)  # 6 bands


class StreamingSkyMapBands:
    """Bounded-memory sky map split into frequency sub-bands. Maintains, *per
    sub-band*, a fixed Mollweide source-count grid and a frequency-sum grid (plus
    a bounded reservoir of the most-uncertain sources). From those it can render,
    for any single sub-band or for all bands combined, BOTH a source-density panel
    (the hex count -- where sources pile up) and a mean-frequency panel (the per-
    hex average f0). Memory is O(nbands * grid + nbands * reservoir), independent
    of catalog size: the whole point is that splitting by band does not reintroduce
    an O(N) cost.
    """

    def __init__(self, gridsize: int = 150, band_edges_mhz=None, max_points: int = 800):
        self.gridsize = gridsize
        self.edges = np.asarray(band_edges_mhz if band_edges_mhz is not None
                                else default_band_edges_mhz(), dtype=float)
        self.nbands = len(self.edges) - 1
        self.max_points = max_points
        self.counts = np.zeros((self.nbands, gridsize, gridsize))
        self.freqsum = np.zeros((self.nbands, gridsize, gridsize))
        # Per-band bounded reservoirs (lists of arrays, each <= max_points long).
        self._rx = [np.empty(0) for _ in range(self.nbands)]
        self._ry = [np.empty(0) for _ in range(self.nbands)]
        self._rconf = [np.empty(0) for _ in range(self.nbands)]
        self._rfreq = [np.empty(0) for _ in range(self.nbands)]
        self.total_seen = 0
        self._xedges, self._yedges, extent = _sky_plane_grid(gridsize)
        # Regular (non-offset) cell centres: a heatmap is a regular matrix, so the
        # density renders as a filled raster (no inter-marker gaps), the cell whose
        # centre is _xc[col]/_yc[row].
        self._xc = 0.5 * (self._xedges[:-1] + self._xedges[1:])
        self._yc = 0.5 * (self._yedges[:-1] + self._yedges[1:])
        # Mask of grid cells inside the valid Mollweide ellipse (the projected sky).
        # A cell inside the ellipse but with zero sources is genuine empty sky and
        # renders dark (zero density); a cell outside is off-projection and stays
        # transparent. Without this, low-density high-latitude sky shows as white
        # speckle (Poisson holes) rather than a filled dark background.
        xmin, xmax, ymin, ymax = extent
        XX, YY = np.meshgrid(self._xc, self._yc)
        ex, ey = 0.5 * (xmax - xmin), 0.5 * (ymax - ymin)
        cx0, cy0 = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        self._inside = ((XX - cx0) / ex) ** 2 + ((YY - cy0) / ey) ** 2 <= 1.0

    def add_block(self, arrays: dict) -> None:
        n = arrays["frequency"].size
        if n == 0:
            return
        self.total_seen += n
        freq = arrays["frequency"]
        fmhz = freq * 1e3
        snr = arrays.get("snr", np.full(n, np.nan))
        conf = encoding.resolve_confidence(arrays.get("confidence", np.full(n, np.nan)), snr)
        x, y = _project(arrays["right_ascension"], arrays["declination"])
        g = self.gridsize
        # Rectangular binning (no odd-row offset): point with continuous index in
        # [j, j+1) falls in cell j, whose centre is _xc[j]/_yc[j] -- so the count
        # matrix maps exactly onto a filled heatmap.
        col = np.clip(np.floor(_data_to_index(x, self._xedges, g, False)).astype(int), 0, g - 1)
        row = np.clip(np.floor(_data_to_index(y, self._yedges, g, False)).astype(int), 0, g - 1)
        # Sub-band index per source (clamped so out-of-range f0 lands in an edge band).
        b = np.clip(np.searchsorted(self.edges, fmhz, side="right") - 1, 0, self.nbands - 1)
        for bi in range(self.nbands):
            m = b == bi
            if not m.any():
                continue
            np.add.at(self.counts[bi], (row[m], col[m]), 1.0)
            np.add.at(self.freqsum[bi], (row[m], col[m]), freq[m])
            self._rx[bi], self._ry[bi], self._rconf[bi], self._rfreq[bi] = _keep_most_uncertain(
                np.concatenate([self._rx[bi], x[m]]), np.concatenate([self._ry[bi], y[m]]),
                np.concatenate([self._rconf[bi], conf[m]]), np.concatenate([self._rfreq[bi], freq[m]]),
                self.max_points)

    def _selection(self, label: str, counts2d, freqsum2d, rx, ry, rconf, rfreq) -> dict:
        g = self.gridsize
        # Empty cells -> NaN so the heatmap leaves them transparent (the sky
        # outside the populated region, and genuine holes in a sparse sub-band,
        # read as background rather than a colour).
        with np.errstate(invalid="ignore", divide="ignore"):
            # Inside-but-empty cells -> 0 (dark, filled sky); outside ellipse -> NaN.
            logcount = np.where(counts2d > 0, np.log10(np.maximum(counts2d, 1.0)),
                                np.where(self._inside, 0.0, np.nan))
            mean_freq = np.where(counts2d > 0, freqsum2d / np.maximum(counts2d, 1), np.nan) * 1e3
        count_disp = np.where(counts2d > 0, counts2d, np.where(self._inside, 0.0, np.nan))
        # Outlier marks: reservoir sources sitting in low-density cells (rectangular
        # lookup, matching the grid). The dense interior is the heatmap's job.
        if rx.size:
            rcol = np.clip(np.floor(_data_to_index(rx, self._xedges, g, False)).astype(int), 0, g - 1)
            rrow = np.clip(np.floor(_data_to_index(ry, self._yedges, g, False)).astype(int), 0, g - 1)
            local = counts2d[rrow, rcol]
        else:
            local = np.empty(0)
        occ = counts2d[counts2d > 0]
        thr = max(2.0, float(np.percentile(occ, 40))) if occ.size else np.inf
        m = local <= thr
        sizes = encoding.confidence_sizes(rconf[m], min_px=3.0, max_px=10.0)
        return {
            "label": label,
            "count": count_disp, "logcount": logcount, "meanfreq": mean_freq,
            # Raw grids (0 where empty) for the matplotlib hex renderer, which
            # aggregates them into a coarser, visibly-hexagonal lattice.
            "counts_grid": counts2d, "freqsum_grid": freqsum2d,
            "mx": rx[m], "my": ry[m], "msize": sizes,
        }

    def selections(self) -> list:
        """Build the 'All bands' selection plus one per sub-band (each with both
        the density and mean-frequency channels precomputed)."""
        sels = []
        # All bands combined (per-band reservoirs merged, re-bounded).
        all_rx, all_ry, all_rc, all_rf = _keep_most_uncertain(
            np.concatenate(self._rx), np.concatenate(self._ry),
            np.concatenate(self._rconf), np.concatenate(self._rfreq), self.max_points)
        sels.append(self._selection("All bands", self.counts.sum(0), self.freqsum.sum(0),
                                    all_rx, all_ry, all_rc, all_rf))
        for bi in range(self.nbands):
            lab = f"{self.edges[bi]:.2g}–{self.edges[bi + 1]:.2g} mHz"
            sels.append(self._selection(lab, self.counts[bi], self.freqsum[bi],
                                        self._rx[bi], self._ry[bi], self._rconf[bi], self._rfreq[bi]))
        return sels

    def figure(self, backend: Optional[RenderBackend] = None, active: int = 0):
        backend = backend or PlotlyBackend()
        title = f"Sky map (streaming, {self.total_seen} sources, grid {self.gridsize})"
        if backend.name == "matplotlib":
            # Static, publication-quality: a true hexagonal tessellation.
            return backend.sky_hex_panels(
                self.selections(), xc=self._xc, yc=self._yc,
                cmap_density=encoding.DENSITY_CMAP, cmap_freq=encoding.FREQUENCY_CMAP,
                title=title, active=active)
        # Interactive: filled heatmap with the sub-band dropdown.
        return backend.sky_band_panels(
            self.selections(), xc=self._xc, yc=self._yc, graticule=_graticule(),
            cmap_density=encoding.DENSITY_CMAP_PLOTLY, cmap_freq=encoding.FREQUENCY_CMAP_PLOTLY,
            title=title, active=active)


def plot_sky_map_bands(block_iter, gridsize: int = 150, band_edges_mhz=None,
                       max_points: int = 800, backend: Optional[RenderBackend] = None,
                       active: int = 0):
    """Bounded-memory, sub-band-split sky map from an iterator of column blocks.
    Renders both a density panel and a mean-frequency panel, filterable by f0
    sub-band (or 'All bands'). ``active`` picks which selection is shown on load
    (0 = All bands, 1..nbands = each sub-band). See :class:`StreamingSkyMapBands`."""
    agg = StreamingSkyMapBands(gridsize=gridsize, band_edges_mhz=band_edges_mhz,
                               max_points=max_points)
    for arrays, _start, _total in block_iter:
        agg.add_block(arrays)
    return agg.figure(backend=backend, active=active)


def plot_sky_map_streaming(block_iter, gridsize: int = 150, max_points: int = 800,
                           backend: Optional[RenderBackend] = None, log_counts: bool = True):
    """Build a bounded-memory sky map from an iterator of catalog column blocks
    (e.g. ``HDF5RepositoryAdapter.iter_catalog_arrays``). Each block is consumed
    then discarded, so peak memory does not grow with catalog size."""
    agg = StreamingSkyMap(gridsize=gridsize, max_points=max_points)
    for arrays, _start, _total in block_iter:
        agg.add_block(arrays)
    return agg.figure(backend=backend, log_counts=log_counts)


class StreamingWaterfall:
    """Bounded-memory waterfall aggregator -- the streaming counterpart of
    :func:`plot_waterfall`. Consumes the catalog one row-block at a time and bins
    each block into a *fixed-size* frequency x amplitude/SNR density grid (both
    axes log) plus a *bounded* reservoir of the most-uncertain sources kept as
    discrete marks. Peak memory is O(block + grid + reservoir), independent of
    catalog size.

    The axis extents are fixed once from physically-motivated GB ranges (rather
    than discovered from a first pass), so the aggregation is genuinely
    single-pass and bounded; out-of-range sources fall into the edge bins.
    """

    def __init__(self, y_axis: str = "Amplitude", gridsize: int = 90, max_points: int = 800,
                 freq_range_mhz: Tuple[float, float] = (0.1, 100.0),
                 amp_range: Tuple[float, float] = (1e-24, 1e-19),
                 snr_range: Tuple[float, float] = (1.0, 1000.0)):
        self.y_axis = y_axis
        self._is_snr = y_axis.lower().startswith("snr")
        self.gridsize = gridsize
        self.max_points = max_points
        self._xedges = np.logspace(np.log10(freq_range_mhz[0]), np.log10(freq_range_mhz[1]), gridsize + 1)
        yr = snr_range if self._is_snr else amp_range
        self._yedges = np.logspace(np.log10(yr[0]), np.log10(yr[1]), gridsize + 1)
        self._xc = np.sqrt(self._xedges[:-1] * self._xedges[1:])  # geometric bin centres
        self._yc = np.sqrt(self._yedges[:-1] * self._yedges[1:])
        self.counts = np.zeros((gridsize, gridsize))
        self._rx = np.empty(0)
        self._ry = np.empty(0)
        self._rconf = np.empty(0)
        self._rfreq = np.empty(0)
        self.total_seen = 0

    def add_block(self, arrays: dict) -> None:
        n = arrays["frequency"].size
        if n == 0:
            return
        self.total_seen += n
        fmhz = arrays["frequency"] * 1e3
        snr = arrays.get("snr", np.full(n, np.nan))
        y = snr if self._is_snr else arrays.get("amplitude", np.full(n, np.nan))
        conf = encoding.resolve_confidence(arrays.get("confidence", np.full(n, np.nan)), snr)

        cix = _data_to_index(fmhz, self._xedges, self.gridsize, True)
        ciy = _data_to_index(y, self._yedges, self.gridsize, True)
        row, col = _offset_hex_bin(cix, ciy, self.gridsize)
        np.add.at(self.counts, (row, col), 1.0)

        # Bounded reservoir of the most-uncertain sources (smallest confidence).
        self._rx, self._ry, self._rconf, self._rfreq = _keep_most_uncertain(
            np.concatenate([self._rx, fmhz]), np.concatenate([self._ry, y]),
            np.concatenate([self._rconf, conf]), np.concatenate([self._rfreq, fmhz]),
            self.max_points)

    def figure(self, backend: Optional[RenderBackend] = None):
        backend = backend or PlotlyBackend()
        cmap = encoding.FREQUENCY_CMAP if backend.name == "matplotlib" else encoding.FREQUENCY_CMAP_PLOTLY
        ylab = "SNR" if self._is_snr else "Amplitude"
        hx, hy, hval = _hex_cells(self.counts, self.counts, self._xedges, self._yedges, True, True)
        # Keep only the sparse-region outliers as marks; shrink them so they sit
        # on top of the density rather than blanket it. The density hexes carry
        # the source count, the marks keep their own accent colour (not frequency,
        # which is already the x axis) to avoid two meanings on one colour scale.
        m = _sparse_reservoir(self._rx, self._ry, self.counts, self._xedges, self._yedges, True, True)
        sizes = encoding.confidence_sizes(self._rconf[m], min_px=3.0, max_px=9.0)
        return backend.density_panel(
            hx, hy, hval,
            points=(self._rx[m], self._ry[m], sizes),
            axis_labels=("f0 [mHz]", ylab), log_x=True, log_y=True,
            color_label="source count", hex_size=_hex_px(self.gridsize),
            title=f"Catalog waterfall (streaming, {self.total_seen} sources)",
            cmap=cmap,
        )


def plot_waterfall_streaming(block_iter, y_axis: str = "Amplitude", gridsize: int = 90,
                             max_points: int = 800, backend: Optional[RenderBackend] = None):
    """Build a bounded-memory waterfall from an iterator of catalog column blocks
    (e.g. ``HDF5RepositoryAdapter.iter_catalog_arrays``). Each block is consumed
    then discarded, so peak memory does not grow with catalog size."""
    agg = StreamingWaterfall(y_axis=y_axis, gridsize=gridsize, max_points=max_points)
    for arrays, _start, _total in block_iter:
        agg.add_block(arrays)
    return agg.figure(backend=backend)


def plot_waterfall(
    catalog: GBCatalog,
    y_axis: str = "Amplitude",
    backend: Optional[RenderBackend] = None,
):
    """Frequency vs amplitude/SNR scatter, coloured by frequency, sized by
    ``1 - confidence``."""
    backend = backend or PlotlyBackend()
    src = catalog.sources
    freq = np.array([s.binary.intrinsic.f0 for s in src])
    snr = np.array([s.snr for s in src])
    conf = encoding.resolve_confidence([s.confidence for s in src], snr)
    sizes = encoding.confidence_sizes(conf)
    if y_axis.lower().startswith("snr"):
        y = snr
        ylab = "SNR"
    else:
        y = np.array([s.binary.extrinsic.amplitude if s.binary.extrinsic else np.nan for s in src])
        ylab = "Amplitude"
    hovertext = [f"{s.binary.source_id}<br>f0={f*1e3:.4f} mHz" for s, f in zip(src, freq)]
    return backend.scatter(
        freq * 1e3, y, freq * 1e3, sizes,
        color_label="f0 [mHz]", axis_labels=("f0 [mHz]", ylab),
        log_x=False, log_y=True, hovertext=hovertext, title="Catalog waterfall",
        cmap=encoding.FREQUENCY_CMAP_PLOTLY,
        customdata=freq,  # f0 [Hz] per point -> a click resolves the band
    )
