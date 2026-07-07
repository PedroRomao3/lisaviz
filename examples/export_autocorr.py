#!/usr/bin/env python3
"""Export the autocorrelation diagnostic to figures/autocorr.png.

Same publication path as trace.png / corner.png: a real lisaviz figure rendered
through the Matplotlib backend (static, dpi=200) and written via backend.save().

Usage:
    python3 export_autocorr.py --run-dir /path/to/run --band <subband-label>
    python3 export_autocorr.py --mock          # quick smoke test, no run needed
"""

from __future__ import annotations

import argparse
import os

from lisaviz import LISAViz
from lisaviz.visualization.backends import MatplotlibBackend


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", help="Global-fit run directory (HDF5).")
    ap.add_argument("--band", help="Sub-band label to plot (defaults to first sampled band).")
    ap.add_argument("--source-index", type=int, default=0)
    ap.add_argument("--max-lag", type=int, default=200)
    ap.add_argument("--iteration", type=int, default=None)
    ap.add_argument("--mock", action="store_true", help="Use the in-memory mock repository.")
    ap.add_argument("--out", default="figures/autocorr.png")
    args = ap.parse_args()

    viz = LISAViz.mock() if args.mock else LISAViz(run_dir=args.run_dir, iteration=args.iteration)

    band = args.band or viz.sampled_bands()[0]
    print(f"Rendering autocorrelation for band {band!r} (source {args.source_index})")

    fig = viz.autocorrelation(
        band,
        source_index=args.source_index,
        max_lag=args.max_lag,
        backend=MatplotlibBackend(),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    MatplotlibBackend().save(fig, args.out)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
