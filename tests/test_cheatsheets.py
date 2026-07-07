"""Tests for the scoped reading cheat sheets. Pure data, so no IO/rendering: we
check the entries are well formed and scoped to the right views."""

from lisaviz.visualization.cheatsheets import (
    CHEATSHEETS,
    FALSE_FRIEND,
    PITFALL,
    VIEWS,
    cheatsheets_for,
    cheatsheets_text,
)


def test_text_helper_is_the_dashboard_free_path():
    """A notebook user reaches the same guidance through cheatsheets_text:
    every entry's title and body appears, and an unscoped view is empty."""
    txt = cheatsheets_text("corner")
    for c in cheatsheets_for("corner"):
        assert c.title in txt and c.body in txt
    assert cheatsheets_text("fstat") == ""


def test_entries_are_well_formed():
    assert CHEATSHEETS, "expected at least one cheat sheet"
    for c in CHEATSHEETS:
        assert c.kind in (PITFALL, FALSE_FRIEND)
        assert c.title.strip()
        assert c.body.strip()
        assert c.views, f"{c.title!r} has no views"
        assert set(c.views) <= set(VIEWS), f"{c.title!r} references an unknown view"


def test_only_expert_relevant_families():
    """Only pitfalls and false-friends are carried (anatomy/construction/etc. are
    deliberately omitted for the expert audience)."""
    kinds = {c.kind for c in CHEATSHEETS}
    assert kinds <= {PITFALL, FALSE_FRIEND}


def test_scoping_per_view():
    sky = {c.title for c in cheatsheets_for("sky_map")}
    corner = {c.title for c in cheatsheets_for("corner")}
    trace = {c.title for c in cheatsheets_for("trace")}

    # The inverted-size and density caveats belong to the population views.
    assert any("less certain" in t for t in sky)
    assert any("count, not a posterior" in t for t in sky)
    # Label switching belongs to the per-source posterior views.
    assert any("label switching" in t for t in corner)
    assert corner == {c.title for c in cheatsheets_for("corner")}
    # The convergence-flag caveat reaches the trace too.
    assert any("candidate, not a verdict" in t for t in trace)


def test_unknown_view_is_empty():
    assert cheatsheets_for("does_not_exist") == []
    # A view with no scoped cards (e.g. the F-stat surface) is simply empty.
    assert cheatsheets_for("fstat") == []
