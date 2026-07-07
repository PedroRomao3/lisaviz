#!/usr/bin/env python3
"""Render the streaming (bounded-memory) sky map from a real Global Fit run.

This is the exact path behind the thesis sky map figure: the resolved catalog
is streamed in fixed row-blocks into a fixed Mollweide density grid plus a
bounded reservoir of the most-uncertain sources.

Usage:
    python examples/render_skymap_real.py --run-dir /path/to/gfrun
    python examples/render_skymap_real.py --run-dir /path/to/gfrun --linear
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")

from lisaviz import LISAViz
from lisaviz.visualization.backends import MatplotlibBackend


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True, help="Global Fit run directory (HDF5).")
    ap.add_argument("--out", default="sky_map_streaming.png", help="Output PNG path.")
    ap.add_argument("--block-size", type=int, default=50_000)
    ap.add_argument("--linear", action="store_true",
                    help="Linear count colouring instead of the log10 default.")
    args = ap.parse_args()

    viz = LISAViz(run_dir=args.run_dir)
    fig = viz.sky_map_streaming(block_size=args.block_size,
                                backend=MatplotlibBackend(),
                                log_counts=not args.linear)
    fig.savefig(args.out, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    print("wrote", args.out)


if __name__ == "__main__":
    main()
