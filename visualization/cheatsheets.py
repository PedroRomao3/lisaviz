"""Scoped reading cheat sheets for the diagnostic views (Visualization layer).

Following Wang et al.'s cheat-sheet families, this carries only the two that
matter for an expert audience: **pitfalls** (a spot on a chart where a misread is
easy) and **false-friends** (a feature that looks like the familiar thing but
means something else). The anatomy / construction / well-known-relative families
are deliberately omitted: a physicist already reads a corner, trace or sky map
fluently, so those would only restate what they know.

Every entry is tied to one of the tool's *own* rendering choices, so it warns
about misreadings the visualization itself can introduce rather than teaching the
chart type. The content lives in the library (pure data, no IO or rendering) so
the dashboard, a notebook or a docstring can all surface the same text on demand.
NaN-padding artifacts are intentionally NOT listed: the domain model strips
padding by construction, so they are not a rendering hazard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

PITFALL = "pitfall"
FALSE_FRIEND = "false-friend"

# View keys match the dashboard's diagnostic surfaces.
VIEWS = ("sky_map", "waterfall", "corner", "trace", "fstat", "psd_residual", "waveform")


@dataclass(frozen=True)
class CheatSheet:
    """One reading watch-out: its kind, a short title, a one or two sentence body
    and the views it applies to."""

    kind: str          # PITFALL | FALSE_FRIEND
    title: str
    body: str
    views: Tuple[str, ...]


CHEATSHEETS: Tuple[CheatSheet, ...] = (
    CheatSheet(
        FALSE_FRIEND,
        "A bigger mark is less certain",
        "Mark size is 1 minus confidence, so the largest marks are the least "
        "reliable sources, not the strongest. This is the opposite of the usual "
        "'bigger is more important' reflex.",
        ("sky_map", "waterfall"),
    ),
    CheatSheet(
        FALSE_FRIEND,
        "A second mode may be label switching",
        "A second peak in the corner can be label switching rather than real "
        "bimodality: the sources are exchangeable, so the sampler can swap which "
        "one is labelled 'source 0' between draws and blend two sources into one "
        "slot. Check the nsource trace before reading two peaks as two physical "
        "modes.",
        ("corner", "trace"),
    ),
    CheatSheet(
        PITFALL,
        "Density is a count, not a posterior",
        "Above the density-reduction threshold the smooth field is a binned source "
        "count, not a posterior surface. A dense blob marks where many sources sit, "
        "not the probable location of one source.",
        ("sky_map", "waterfall"),
    ),
    CheatSheet(
        PITFALL,
        "A flag is a candidate, not a verdict",
        "The convergence flag (FR-05) and the red walker highlights mark candidates "
        "for inspection. A flagged band is a prompt to look, not proof that the "
        "chain failed to converge.",
        ("sky_map", "trace"),
    ),
)


def cheatsheets_for(view: str) -> List[CheatSheet]:
    """Return the cheat sheets that apply to a given view (e.g. ``"sky_map"``).
    Unknown or unscoped views return an empty list."""
    return [c for c in CHEATSHEETS if view in c.views]


def cheatsheets_text(view: str) -> str:
    """Readable plain-text rendering of a view's cheat sheets, for a notebook,
    a script or a docstring (the dashboard-free path to the same guidance).
    Returns an empty string when the view has none."""
    lines = [f"[{c.kind}] {c.title}\n    {c.body}" for c in cheatsheets_for(view)]
    return "\n".join(lines)
