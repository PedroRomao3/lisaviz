"""Computer-assisted heuristic evaluation of the library's figures.

A scoped, faithful implementation of the method of Zhu and Gumieniak (2022),
"Computer-Assisted Heuristic Evaluation of Data Visualization". The method reads
a declarative (Plotly) figure's JSON through one call (their ``heustic_eval(
fig.to_json())``), labels the plot by seven attributes (visual frames, visual
structures, visual unities, visual primitives, labeling, interaction, data
attributes), then matches heuristic rules to the plot. Rules fall into three
categories: *difficult-to-check* (too abstract, excluded), *advice* (specific but
needing a human, surfaced as reminders) and *automatic-check* (mechanically
verifiable, checked by the program). Per that method this applies ONLY to
declarative figures: a Plotly ``Figure`` (or its JSON), never an imperative
Matplotlib export.

This module implements the *automatic-check* rules relevant to the library's own
figures plus the *advice* reminders that fire on its encodings. It is the
automated half of a two-pronged evaluation method; the expert
inspection (the advice and difficult-to-check rules) owns scientific utility and
is not replaced here. The checker reports warnings and advice. It does not
certify usability.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .population import SPLATTER_THRESHOLD

# Perceptually-uniform sequential colormaps (Plotly / matplotlib names, lowercased).
# Anything outside this set that is a named rainbow is rejected outright (Borland
# and Taylor, rainbow map still considered harmful); an unrecognised name is
# flagged for a human to confirm rather than passed silently.
_PERCEPTUALLY_UNIFORM = {
    "viridis", "cividis", "plasma", "inferno", "magma", "mako", "rocket",
    "greys", "grays", "blues", "greens", "oranges", "reds", "purples",
}
_RAINBOW = {
    "jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral", "gist_ncar",
}

_AXIS_KEY = re.compile(r"^[xy]axis\d*$")
_CARTESIAN = {"scatter", "scattergl", "bar", "histogram", "contour", "heatmap", "box", "violin"}


# --------------------------------------------------------------------------- #
#  Report model                                                               #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HeuristicFinding:
    rule_id: str
    category: str   # "automatic-check" | "advice"
    attribute: str  # one of the seven attributes the rule attaches to
    status: str     # "pass" | "warn" (automatic-check) | "info" (advice)
    message: str


@dataclass(frozen=True)
class HeuristicReport:
    """Outcome of evaluating one figure: the seven-attribute labels detected and
    the findings, split into automatic-check results and advice reminders."""

    labels: Dict[str, Any]
    findings: List[HeuristicFinding] = field(default_factory=list)

    @property
    def warnings(self) -> List[HeuristicFinding]:
        return [f for f in self.findings if f.status == "warn"]

    @property
    def checks(self) -> List[HeuristicFinding]:
        return [f for f in self.findings if f.category == "automatic-check"]

    @property
    def advice(self) -> List[HeuristicFinding]:
        return [f for f in self.findings if f.category == "advice"]

    @property
    def passed(self) -> bool:
        """True when no automatic-check rule raised a warning."""
        return not self.warnings

    def summary(self) -> str:
        n_pass = sum(1 for f in self.checks if f.status == "pass")
        head = f"{'PASS' if self.passed else 'WARN'}  ({n_pass}/{len(self.checks)} automatic checks pass)"
        lines = [head]
        for f in self.warnings:
            lines.append(f"  [warn] {f.rule_id} ({f.attribute}): {f.message}")
        for f in self.advice:
            lines.append(f"  [advice] {f.rule_id} ({f.attribute}): {f.message}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
#  JSON helpers                                                               #
# --------------------------------------------------------------------------- #
def _to_json(fig: Any) -> Dict[str, Any]:
    """Coerce a Plotly figure, JSON string or dict into a plain dict. Reject an
    imperative (Matplotlib) figure, which the method does not support."""
    if hasattr(fig, "to_plotly_json"):
        return fig.to_plotly_json()
    if isinstance(fig, str):
        return json.loads(fig)
    if isinstance(fig, dict):
        return fig
    if hasattr(fig, "savefig"):
        raise ValueError(
            "heuristic_eval works on declarative (Plotly) figures only, following "
            "Zhu and Gumieniak. A Matplotlib figure exposes no plot JSON to "
            "introspect. Render the interactive view through the Plotly backend, "
            "or inspect the static export by hand."
        )
    raise TypeError(f"Cannot read a Plotly figure from {type(fig)!r}.")


def _text(obj: Any) -> str:
    """Pull display text out of a Plotly title field, which may be a bare string
    or a ``{'text': ...}`` dict."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        return _text(obj.get("text"))
    return ""


def _arr_len(v: Any) -> int:
    """Length of a Plotly data field, allowing for its typed-array JSON encoding
    (a ``{'dtype', 'bdata'}`` base64 dict, produced by ``fig.to_json()``)."""
    if v is None:
        return 0
    if isinstance(v, dict):
        if "bdata" in v and "dtype" in v:
            try:
                raw = base64.b64decode(v["bdata"])
                return len(raw) // max(np.dtype(v["dtype"]).itemsize, 1)
            except Exception:  # noqa: BLE001 - introspection must not crash on odd input
                return 0
        return 0
    if isinstance(v, str):
        return 0
    arr = np.asarray(v)
    return int(arr.shape[0]) if arr.ndim >= 1 else 1


def _is_array(v: Any) -> bool:
    """True when a field carries a data variable (an array), not a scalar."""
    if isinstance(v, dict):
        return "bdata" in v
    if v is None or isinstance(v, str):
        return False
    return hasattr(v, "__len__") and len(v) > 1


def _trace_len(trace: Dict[str, Any]) -> int:
    for key in ("x", "lat", "r", "values"):
        if trace.get(key) is not None:
            return _arr_len(trace.get(key))
    return 0


def _colorscale_name(trace: Dict[str, Any]) -> Optional[Any]:
    marker = trace.get("marker") or {}
    for holder in (trace, marker):
        cs = holder.get("colorscale")
        if cs is not None:
            return cs
    return None


def _parse_color(c: Any) -> Optional[tuple]:
    """Parse a ``#rrggbb`` or ``rgb()/rgba()`` colour into an (r, g, b) triple in
    [0, 1]."""
    if not isinstance(c, str):
        return None
    c = c.strip()
    if c.startswith("#"):
        h = c[1:]
        if len(h) == 3:
            h = "".join(ch * 2 for ch in h)
        if len(h) >= 6:
            return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    if c.startswith(("rgb(", "rgba(")):
        nums = c[c.index("(") + 1:c.index(")")].split(",")
        try:
            return tuple(float(x) / 255.0 for x in nums[:3])
        except ValueError:
            return None
    return None


def _luminance(rgb: tuple) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


def _classify_colorscale(cs: Any) -> Optional[str]:
    """Classify a Plotly colorscale as ``uniform``, ``rainbow`` or ``unknown``.

    Plotly stores a named scale either as its name or, after a round trip, as an
    explicit list of ``[position, colour]`` stops. A named scale is matched
    directly. A stop list is judged by its lightness profile: a perceptually
    uniform sequential map climbs in lightness monotonically, whereas a rainbow
    map (jet and kin) rises then falls, which is the defect Borland and Taylor
    identify. A non-monotonic lightness profile is therefore read as rainbow."""
    if cs is None:
        return None
    if isinstance(cs, str):
        n = cs.lower()
        if n in _RAINBOW:
            return "rainbow"
        if n in _PERCEPTUALLY_UNIFORM:
            return "uniform"
        return "unknown"
    try:
        lums = [_luminance(c) for c in (_parse_color(stop[1]) for stop in cs) if c is not None]
    except (TypeError, IndexError):
        return "unknown"
    if len(lums) < 3:
        return "unknown"
    diffs = [b - a for a, b in zip(lums, lums[1:]) if abs(b - a) > 1e-3]
    reversals = sum(1 for a, b in zip(diffs, diffs[1:]) if a * b < 0)
    return "uniform" if reversals == 0 else "rainbow"


def _marker_shows_scale(trace: Dict[str, Any]) -> bool:
    marker = trace.get("marker") or {}
    return bool(marker.get("showscale")) or trace.get("type") == "contour"


def _colorbar_title(trace: Dict[str, Any]) -> str:
    marker = trace.get("marker") or {}
    for holder in (marker, trace):
        cb = holder.get("colorbar")
        if cb:
            return _text(cb.get("title"))
    return ""


def _size_is_encoded(trace: Dict[str, Any]) -> bool:
    """True when marker size carries a data variable (an array) rather than a
    single scalar."""
    return _is_array((trace.get("marker") or {}).get("size"))


# --------------------------------------------------------------------------- #
#  Seven-attribute labelling                                                  #
# --------------------------------------------------------------------------- #
def _label_plot(spec: Dict[str, Any]) -> Dict[str, Any]:
    data = spec.get("data") or []
    layout = spec.get("layout") or {}

    axes = {k: v for k, v in layout.items() if _AXIS_KEY.match(k)}
    visible_axes = {k: v for k, v in axes.items() if v.get("visible", True) is not False}

    types = sorted({t.get("type", "scatter") for t in data})
    updatemenus = layout.get("updatemenus") or []
    has_animation = any(
        (b or {}).get("method") == "animate"
        for um in updatemenus for b in (um.get("buttons") or [])
    )

    primitives = set()
    for t in data:
        marker = t.get("marker") or {}
        if marker.get("color") is not None or _colorscale_name(t) is not None:
            primitives.add("color")
        if _size_is_encoded(t):
            primitives.add("size")
        sym = marker.get("symbol")
        if sym is not None and not isinstance(sym, str):
            primitives.add("shape")

    return {
        "visual_frames": "multiple" if len(axes) > 2 else "single",
        "visual_structures": types,
        "visual_primitives": sorted(primitives),
        "labeling": {
            "title": _text(layout.get("title")),
            "visible_axes": list(visible_axes.keys()),
            "has_colorbar": any(_marker_shows_scale(t) for t in data),
        },
        "interaction": {"animation": has_animation},
        "data_attributes": {"max_trace_points": max((_trace_len(t) for t in data), default=0)},
        "_data": data,
        "_axes": axes,
        "_visible_axes": visible_axes,
    }


# --------------------------------------------------------------------------- #
#  Rules                                                                       #
# --------------------------------------------------------------------------- #
def _automatic_checks(labels: Dict[str, Any], overdraw_threshold: int) -> List[HeuristicFinding]:
    out: List[HeuristicFinding] = []
    data = labels["_data"]
    lab = labels["labeling"]

    # AC-TITLE -- a plot must carry a title (paper: missing-title warning).
    out.append(HeuristicFinding(
        "AC-TITLE", "automatic-check", "labeling",
        "pass" if lab["title"] else "warn",
        "title present" if lab["title"] else "no figure title; a reader cannot tell what the plot shows",
    ))

    # AC-AXES -- a cartesian plot must label both data dimensions: at least one
    # visible x-axis and one visible y-axis carry a title. Axes hidden on purpose
    # (the sky map suppresses them, the colorbar carries the meaning) are exempt
    # and checked by AC-COLORBAR instead, as are non-cartesian plots.
    axes = labels["_axes"]
    hidden = [k for k, v in axes.items() if v.get("visible") is False]
    cartesian = any(t.get("type", "scatter") in _CARTESIAN for t in data)
    is_map = bool(axes) and len(hidden) == len(axes)  # every axis deliberately hidden
    if cartesian and not is_map:
        x_titled = any(_text(v.get("title")) for k, v in axes.items()
                       if k.startswith("xaxis") and v.get("visible") is not False)
        y_titled = any(_text(v.get("title")) for k, v in axes.items()
                       if k.startswith("yaxis") and v.get("visible") is not False)
        ok = x_titled and y_titled
        out.append(HeuristicFinding(
            "AC-AXES", "automatic-check", "labeling",
            "pass" if ok else "warn",
            "both axes labelled" if ok
            else f"missing axis title ({'x' if not x_titled else 'y'} axis is unlabelled)",
        ))

    # AC-COLORBAR -- a colour encoding must name the quantity it encodes.
    scaled = [t for t in data if _marker_shows_scale(t)]
    if scaled:
        missing = [t for t in scaled if not _colorbar_title(t)]
        out.append(HeuristicFinding(
            "AC-COLORBAR", "automatic-check", "labeling",
            "pass" if not missing else "warn",
            "colour scale labelled" if not missing
            else "a colour scale carries no title; the encoded quantity is unstated",
        ))

    # AC-COLORMAP -- colour scales must be perceptually uniform; reject rainbow
    # (Borland and Taylor) and flag anything unrecognised for a human to confirm.
    for t in data:
        cs = _colorscale_name(t)
        kind = _classify_colorscale(cs)
        if kind is None:
            continue
        name = cs if isinstance(cs, str) else "the colour scale"
        if kind == "rainbow":
            out.append(HeuristicFinding(
                "AC-COLORMAP", "automatic-check", "visual-primitives", "warn",
                f"{name} is not perceptually uniform (non-monotonic lightness); use a uniform map",
            ))
        elif kind == "uniform":
            out.append(HeuristicFinding(
                "AC-COLORMAP", "automatic-check", "visual-primitives", "pass",
                "perceptually-uniform colour scale (monotonic lightness)",
            ))
        else:
            out.append(HeuristicFinding(
                "AC-COLORMAP", "automatic-check", "visual-primitives", "warn",
                f"{name} is not a recognised perceptually-uniform map; confirm or replace",
            ))
        break  # one verdict per figure is enough for the report

    # AC-OVERDRAW -- a raw marker scatter above the density-reduction threshold
    # will overdraw; recommend the streaming density view.
    for t in data:
        if t.get("type") in ("scatter", "scattergl") and "markers" in (t.get("mode") or "") \
                and _trace_len(t) > overdraw_threshold:
            out.append(HeuristicFinding(
                "AC-OVERDRAW", "automatic-check", "visual-structures", "warn",
                f"{_trace_len(t)} marker points exceed the density-reduction threshold "
                f"({overdraw_threshold}); render through the streaming density view",
            ))
            break
    else:
        if any(t.get("type") in ("scatter", "scattergl") for t in data):
            out.append(HeuristicFinding(
                "AC-OVERDRAW", "automatic-check", "visual-structures", "pass",
                "no raw scatter exceeds the density-reduction threshold",
            ))

    return out


def _advice(labels: Dict[str, Any]) -> List[HeuristicFinding]:
    out: List[HeuristicFinding] = []
    data = labels["_data"]

    # ADV-ANIMATION -- if the plot animates, remind of the pros and cons.
    if labels["interaction"]["animation"]:
        out.append(HeuristicFinding(
            "ADV-ANIMATION", "advice", "interaction", "info",
            "the plot animates; staged transitions aid comparison but can hide change. "
            "Confirm the transition is legible (Heer and Robertson)",
        ))

    # ADV-SIZE -- mark size encodes a quantity; its polarity needs to be stated
    # because size carries no intrinsic direction (here larger = less certain).
    if "size" in labels["visual_primitives"]:
        out.append(HeuristicFinding(
            "ADV-SIZE", "advice", "visual-primitives", "info",
            "mark size encodes a data variable; state its polarity in the legend, "
            "since the library uses size = 1 - confidence (a larger mark is less certain)",
        ))

    # ADV-COLOR-COUNT -- more than four discrete series colours can be hard to
    # tell apart (paper's too-many-colours rule, Szafir on small marks).
    discrete = [t for t in data if isinstance((t.get("line") or {}).get("color"), str)
                and _colorscale_name(t) is None]
    n_colors = len({(t.get("line") or {}).get("color") for t in discrete})
    if n_colors > 4:
        out.append(HeuristicFinding(
            "ADV-COLOR-COUNT", "advice", "visual-primitives", "info",
            f"{n_colors} discrete series colours are in use; confirm they stay "
            "discriminable on thin marks (Szafir)",
        ))

    return out


# --------------------------------------------------------------------------- #
#  Public API                                                                  #
# --------------------------------------------------------------------------- #
def heuristic_eval(fig: Any, overdraw_threshold: int = SPLATTER_THRESHOLD) -> HeuristicReport:
    """Evaluate one declarative (Plotly) figure against the automatic-check and
    advice heuristics of Zhu and Gumieniak.

    ``fig`` is a Plotly ``Figure``, its ``to_plotly_json()`` dict or a JSON
    string. A Matplotlib figure is rejected: the method only reads declarative
    plot JSON. Returns a :class:`HeuristicReport`. The report ``passed`` when no
    automatic-check rule raised a warning. The advice items are reminders for the
    expert inspection, not pass/fail results.
    """
    spec = _to_json(fig)
    labels = _label_plot(spec)
    findings = _automatic_checks(labels, overdraw_threshold) + _advice(labels)
    public = {k: v for k, v in labels.items() if not k.startswith("_")}
    return HeuristicReport(labels=public, findings=findings)


def evaluate_figures(named_figures: Dict[str, Any], overdraw_threshold: int = SPLATTER_THRESHOLD) -> Dict[str, HeuristicReport]:
    """Run :func:`heuristic_eval` over a mapping of label -> figure (e.g. one per
    visualization module) and return label -> report. Used to build the
    heuristics-by-module results table."""
    return {name: heuristic_eval(fig, overdraw_threshold) for name, fig in named_figures.items()}
