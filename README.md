# lisaviz

An interactive diagnostic library that turns the LISA Global Fit pipeline's raw
MCMC / HDF5 outputs into Galactic-Binary diagnostic plots. GB-only for now, but
the layering is designed so MBHB / EMRI / Noise source types slot in behind the
same interfaces.

## Install

```bash
pip install .          # Python >= 3.10; light, pip-installable deps only
pip install .[dev]     # + pytest, emcee
```

## Architecture (3 layers, dependencies point inward — not MVC)

```
visualization ─┐                 ┌─ ingestion
               ├──▶ domain ◀─────┤
   (hides       │  (core model,   │  (hides HDF5;
   rendering)   │  stdlib+numpy)  │   Repository/Adapter)
```

* **`domain/`** — Core model. Imports only the standard library and `numpy`
  (NFR-01, enforced by `tests/test_architecture.py`).
  * Value Objects: `IntrinsicParameters` (f0, fdot, δ, α — sky position is kept
    intrinsic because of Doppler coupling), `ExtrinsicParameters` (A, ι, ψ, φ0),
    `FrequencyBand`.
  * Entity (identity, not value): `GalacticBinary`.
  * Aggregate Root: `MCMCDraw` — a **variable-length** source collection; the
    trans-dimensional source count is a first-class fact, never padded. The
    model is sampler-agnostic: reversible-jump (Eryn) and product-space
    (jexplore) chains both reduce to the same per-draw source list.
  * IO models: `GBCatalog`/`CatalogSource`, `GBChain`, `MCMCChain`,
    `NoiseModel`, `MBHBCatalog`.
  * Services: `sampling` (hypercube↔physical transforms, mirrored from the
    pipeline's GB likelihood), `confidence` (sigmoid→overlap→confusion),
    `convergence` (FR-05 candidate flag).
* **`ingestion/`** — `IGalacticBinaryRepository` implementations.
  `HDF5RepositoryAdapter` reads the real gfrun schema (PyTables catalogs;
  chains via mmap slicing, preferring the `new_source/chain_*.npy` 8-param PE
  posterior and falling back to the legacy `fstats/chain_*.npy` 4-param search
  chain), is **read-only**, and fails fast on incompatible chunk layouts.
  `InMemoryMockRepository` returns synthetic domain objects — no filesystem,
  no HDF5.
* **`visualization/`** — Plotting behind a `RenderBackend` Strategy, one
  concrete strategy per module in `visualization/backends/`
  (`PlotlyBackend` interactive / `MatplotlibBackend` publication export). MCMC
  statistics go through the `arviz-stats` **array** interface only (no xarray /
  DataTree). Sky projection is delegated to `astropy`.

## GB view modules

| Module | Function(s) | FR |
|---|---|---|
| MCMC Health | `plot_traces`, `plot_autocorrelation`, `diagnostics_table` | FR-04 |
| Posterior | `plot_corner` (sampling→physical), publication export | FR-03, FR-07 |
| Reconstruction | `plot_waveform_overlay` (+ match), `plot_psd_residual` | FR-06 |
| Population | `plot_sky_map` (+ streaming variant), `plot_waterfall` | FR-02 |
| F-stat search | `plot_fstat_contour` (2-D degeneracy contours, profile-max) | FR-09 |

## Quick start

```python
from lisaviz import LISAViz

viz = LISAViz.mock()                 # or LISAViz(run_dir="/path/to/gfrun")
band = viz.subbands()[0].label
viz.traces(band, burn_in=200)        # plotly figure
viz.corner(band, burn_in=200)        # matplotlib figure (physical units)
viz.sky_map()                        # frequency colour, size = 1 - confidence
viz.sky_map_streaming()              # bounded-memory density + outlier marks
viz.diagnostics(band, burn_in=200)   # ESS / R-hat / tau via arviz-stats
```

Every plotting function is also callable directly with plain domain objects
(no h5py handles, no chunk layouts) — ideal for notebooks.

## Examples

`examples/` holds runnable scripts (install the package first):

* `demo.py` — end-to-end walkthrough on mock data, writes HTML/PNG to
  `demo_output/`.
* `make_figures.py` — regenerates the full thesis figure set into
  `examples/figures/`, every panel through a public lisaviz function.
* `export_autocorr.py` — autocorrelation diagnostic from a real run
  (`--run-dir`) or mock (`--mock`).
* `render_skymap_real.py` — the bounded streaming sky map on a real run
  (`--run-dir`), the exact path behind the thesis sky map figure.

## Visual-encoding rules

* Sky map colour = a **perceptually-uniform sequential** colormap
  (rainbow/jet rejected); the streaming density view declares its scale on the
  colorbar (log₁₀ source count).
* Mark size = **1 − confidence** (error-ellipse convention: BIG = MORE uncertain).
* `confidence` starts at `sigmoid(SNR)`, is replaced by cross-iteration overlap
  for carried-over sources, then penalised by neighbour confusion (floored at 0).

## Develop / validate

```bash
python -m pytest tests/    # domain, ingestion, viz, architecture, validation
```

`validation/straight_line.py` runs an end-to-end y = mx + c recovery through
the full stack (`from lisaviz.validation import run_straight_line`).

## License

MIT — see [LICENSE](LICENSE).

## Citation

Developed for the master's thesis *"Visualizing the LISA Global Fit:
Architecture for High-Dimensional Bayesian Spaces"* (FEUP, 2026).
See [CITATION.cff](CITATION.cff).
