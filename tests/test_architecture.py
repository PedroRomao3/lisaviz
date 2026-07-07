"""NFR-01: strict layered decoupling. The Core Domain layer must not import any
IO / rendering / statistics infrastructure."""

import ast
import os

import pytest

_DOMAIN_DIR = os.path.join(os.path.dirname(__file__), "..", "domain")
_FORBIDDEN = {"h5py", "plotly", "arviz", "arviz_stats", "astropy", "xarray", "pandas", "matplotlib", "streamlit"}


def _imports(path):
    with open(path) as fh:
        tree = ast.parse(fh.read(), filename=path)
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("fname", sorted(f for f in os.listdir(_DOMAIN_DIR) if f.endswith(".py")))
def test_domain_imports_no_infrastructure(fname):
    used = _imports(os.path.join(_DOMAIN_DIR, fname))
    leaked = used & _FORBIDDEN
    assert not leaked, f"{fname} imports forbidden infrastructure: {leaked}"


_PYPROJECT = os.path.join(os.path.dirname(__file__), "..", "pyproject.toml")


def test_lisaviz_is_standalone_and_lightweight():
    """NFR-03: lisaviz ships its own pyproject with only light, pip-installable,
    open-source runtime deps -- no HPC-specific or git-only packages, and
    nothing from the surrounding globalfit monorepo."""
    assert os.path.isfile(_PYPROJECT), "lisaviz needs its own pyproject.toml (NFR-03)"
    text = open(_PYPROJECT).read()
    assert 'name = "lisaviz"' in text
    deps_block = text.split("dependencies = [", 1)[1].split("]", 1)[0]
    assert "git+" not in deps_block, "runtime deps must not be git-only packages"
    for heavy in ("jaxgb", "lisabeta", "ptemcee", "mojito", "globalfit"):
        assert heavy not in deps_block, f"lisaviz must not depend on {heavy}"
    for needed in ("numpy", "pandas", "h5py", "arviz-stats", "astropy", "plotly", "matplotlib"):
        assert needed in deps_block, f"missing expected runtime dependency: {needed}"
