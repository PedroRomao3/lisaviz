"""MCMC diagnostics via the arviz-stats ARRAY interface (Visualization layer).

ArviZ is used *exclusively* through ``arviz_stats.base.array_stats`` -- the
low-level NumPy array interface. No xarray, no ``InferenceData``, no
``DataTree``. Domain objects are converted to clean, padding-free arrays here;
ArviZ therefore remains a replaceable implementation detail behind this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from arviz_stats.base import array_stats

from ..domain.models import MCMCChain

# Convention used throughout: arrays are shaped (n_chains, n_draws).


def chain_matrix(chain: MCMCChain, param_name: str) -> np.ndarray:
    """Stack the per-walker samples of one sampling-space parameter into a
    ``(n_walkers, n_draws)`` array. Walkers are truncated to the shortest length
    so the result is rectangular *for the diagnostic call only* -- the domain
    model itself never pads."""
    series = [np.asarray(w.samples[param_name], dtype=float) for w in chain.walkers]
    n = min(s.size for s in series)
    return np.vstack([s[:n] for s in series])


def loglik_matrix(chain: MCMCChain) -> np.ndarray:
    series = [np.asarray(w.log_likelihood, dtype=float) for w in chain.walkers]
    n = min(s.size for s in series)
    return np.vstack([s[:n] for s in series])


def integrated_autocorr_time(ary: np.ndarray, c: float = 5.0) -> float:
    """Integrated autocorrelation time tau with automatic (Sokal) windowing.

    ``ary`` is ``(n_chains, n_draws)`` or 1-D. Autocorrelation is computed per
    chain via arviz-stats and averaged; tau = 1 + 2*sum rho_k truncated at the
    first window M with M >= c*tau.
    """
    ary = np.atleast_2d(ary)
    rho = np.mean([np.asarray(array_stats.autocorr(row)) for row in ary], axis=0)
    taus = 1.0 + 2.0 * np.cumsum(rho[1:])
    window = np.arange(1, taus.size + 1)
    candidates = np.where(window >= c * taus)[0]
    m = candidates[0] if candidates.size else taus.size - 1
    return float(taus[m]) if taus.size else 1.0


def acceptance_fraction(chain: MCMCChain) -> float:
    """Proxy acceptance fraction: the mean fraction of consecutive draws in
    which at least one sampling parameter changed (a Metropolis reject repeats
    the previous state)."""
    fracs: List[float] = []
    for w in chain.walkers:
        changed = np.zeros(w.n_draws - 1, dtype=bool)
        for v in w.samples.values():
            v = np.asarray(v, dtype=float)
            changed |= np.diff(v) != 0.0
        if changed.size:
            fracs.append(float(np.mean(changed)))
    return float(np.mean(fracs)) if fracs else float("nan")


@dataclass(frozen=True)
class ParamDiagnostics:
    name: str
    ess: float
    rhat: float
    mcse: float
    tau: float
    rhat_is_split: bool = False  # True when R-hat came from splitting one walker


def _rhat(mat: np.ndarray) -> tuple:
    """R-hat for a ``(n_chains, n_draws)`` array.

    With >=2 walkers this is the usual (rank-normalized, split) between-walker
    R-hat. With a *single* walker -- the case for the on-disk GB F-statistic
    search chains -- there is no between-chain variance, so we fall back to a
    **split-R-hat**: the chain is split into its two halves and treated as two
    sub-chains (the standard single-chain Gelman-Rubin diagnostic). Returns
    ``(value, is_split)``.
    """
    if mat.shape[0] >= 2:
        return float(array_stats.rhat(mat)), False
    n = mat.shape[1]
    if n < 4:
        return float("nan"), True
    half = n // 2
    split = np.vstack([mat[0, :half], mat[0, half : 2 * half]])
    return float(array_stats.rhat(split)), True


def diagnostics(chain: MCMCChain, burn_in: int = 0) -> Dict[str, ParamDiagnostics]:
    """Per-parameter ESS / R-hat / MCSE / tau. With >=2 walkers R-hat is the
    standard between-walker statistic; with a single walker it is a split-R-hat
    (flagged via ``rhat_is_split``) -- never silently presented as a multi-chain
    R-hat."""
    out: Dict[str, ParamDiagnostics] = {}
    for name in chain.param_names:
        mat = chain_matrix(chain, name)[:, burn_in:]
        rhat, is_split = _rhat(mat)
        out[name] = ParamDiagnostics(
            name=name,
            ess=float(array_stats.ess(mat)),
            rhat=rhat,
            mcse=float(array_stats.mcse(mat)),
            tau=integrated_autocorr_time(mat),
            rhat_is_split=is_split,
        )
    return out


def autocorr_curves(chain: MCMCChain, param_name: str, max_lag: Optional[int] = None) -> np.ndarray:
    """Walker-averaged autocorrelation curve rho_k for one parameter."""
    mat = chain_matrix(chain, param_name)
    rho = np.mean([np.asarray(array_stats.autocorr(row)) for row in mat], axis=0)
    return rho if max_lag is None else rho[:max_lag]
