"""Straight-line (y = m x + c) validation: a known-answer test of the full stack.

A linear-regression posterior is Gaussian with a known mean, so it is the ideal
ground truth to confirm the Trace and Corner modules: the traces must look
stationary after burn-in and the corner must recover the injected (m, c).

Uses ``emcee`` (32 walkers, 5000 iters, 1000 burn-in) when available; otherwise
falls back to an equivalent per-walker Gaussian random-walk Metropolis sampler
so the harness runs with no extra dependency. The sampler output is wrapped in a
domain :class:`MCMCChain` (sampling-space parameter names ``"m"``, ``"c"``) so
the *same* visualization code path is validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from ..domain.models import GBChain, MCMCChain
from ..visualization import plot_traces
from ..visualization.backends import MatplotlibBackend, RenderBackend


@dataclass(frozen=True)
class StraightLineResult:
    chain: MCMCChain
    truth: Dict[str, float]
    burn_in: int
    # Analytic OLS posterior (the exact known answer the MCMC must recover).
    ols_mean: Dict[str, float]
    ols_std: Dict[str, float]


def _ols_posterior(x: np.ndarray, y: np.ndarray, sigma: float) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Closed-form Gaussian posterior for (m, c) under a flat prior."""
    design = np.column_stack([x, np.ones_like(x)])
    cov = np.linalg.inv(design.T @ design) * sigma ** 2
    mean = cov / sigma ** 2 @ design.T @ y
    return ({"m": float(mean[0]), "c": float(mean[1])},
            {"m": float(np.sqrt(cov[0, 0])), "c": float(np.sqrt(cov[1, 1]))})


def _synthetic_data(m_true: float, c_true: float, sigma: float, n: int, rng) -> Tuple[np.ndarray, np.ndarray, float]:
    x = np.linspace(0.0, 10.0, n)
    y = m_true * x + c_true + rng.normal(0.0, sigma, n)
    return x, y, sigma


def _log_prob(theta: np.ndarray, x: np.ndarray, y: np.ndarray, sigma: float) -> float:
    m, c = theta
    if not (-10 < m < 10 and -50 < c < 50):
        return -np.inf
    resid = y - (m * x + c)
    return -0.5 * np.sum((resid / sigma) ** 2)


def _metropolis(log_prob, p0: np.ndarray, n_iter: int, step: np.ndarray, rng):
    """Per-walker Gaussian random-walk Metropolis. ``p0`` is (n_walkers, ndim)."""
    n_walkers, ndim = p0.shape
    chain = np.empty((n_walkers, n_iter, ndim))
    logp = np.empty((n_walkers, n_iter))
    cur = p0.copy()
    cur_lp = np.array([log_prob(p) for p in cur])
    for it in range(n_iter):
        prop = cur + rng.normal(0.0, step, size=(n_walkers, ndim))
        prop_lp = np.array([log_prob(p) for p in prop])
        accept = np.log(rng.uniform(size=n_walkers)) < (prop_lp - cur_lp)
        cur[accept] = prop[accept]
        cur_lp[accept] = prop_lp[accept]
        chain[:, it, :] = cur
        logp[:, it] = cur_lp
    return chain, logp


def run_straight_line(
    m_true: float = 2.0,
    c_true: float = -3.0,
    sigma: float = 1.0,
    n_data: int = 50,
    n_walkers: int = 32,
    n_iter: int = 5000,
    burn_in: int = 1000,
    seed: int = 0,
) -> StraightLineResult:
    rng = np.random.default_rng(seed)
    x, y, sigma = _synthetic_data(m_true, c_true, sigma, n_data, rng)

    def log_prob(theta):
        return _log_prob(theta, x, y, sigma)

    p0 = np.array([m_true, c_true]) + 0.5 * rng.standard_normal((n_walkers, 2))

    try:
        import emcee

        sampler = emcee.EnsembleSampler(n_walkers, 2, log_prob)
        sampler.run_mcmc(p0, n_iter, progress=False)
        chain = sampler.get_chain()  # (n_iter, n_walkers, ndim)
        chain = np.transpose(chain, (1, 0, 2))  # -> (n_walkers, n_iter, ndim)
        logp = np.transpose(sampler.get_log_prob(), (1, 0))
    except ImportError:
        chain, logp = _metropolis(log_prob, p0, n_iter, np.array([0.05, 0.15]), rng)

    walkers = [
        GBChain(
            walker_id=w,
            samples={"m": chain[w, :, 0], "c": chain[w, :, 1]},
            log_likelihood=logp[w],
            band=None,
        )
        for w in range(n_walkers)
    ]
    ols_mean, ols_std = _ols_posterior(x, y, sigma)
    return StraightLineResult(chain=MCMCChain(walkers=walkers, band=None),
                              truth={"m": m_true, "c": c_true}, burn_in=burn_in,
                              ols_mean=ols_mean, ols_std=ols_std)


def plot_validation(result: StraightLineResult, backend: Optional[RenderBackend] = None):
    """Trace + corner for the straight-line validation. Corner is rendered
    directly via the backend (these are generic m/c parameters, not GB physical
    parameters, so the GB physical-transform corner is bypassed)."""
    backend = backend or MatplotlibBackend()
    trace_fig = plot_traces(result.chain, burn_in=result.burn_in, backend=backend)
    pooled = {
        "m": np.concatenate([w.samples["m"][result.burn_in:] for w in result.chain.walkers]),
        "c": np.concatenate([w.samples["c"][result.burn_in:] for w in result.chain.walkers]),
    }
    corner_fig = MatplotlibBackend().corner(pooled, ["m", "c"], truths=result.truth, title="Straight-line validation")
    return trace_fig, corner_fig
