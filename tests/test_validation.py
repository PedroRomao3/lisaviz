"""The straight-line MCMC validation must recover the analytic OLS posterior."""

import numpy as np

from lisaviz.validation import run_straight_line


def test_recovers_analytic_ols_posterior():
    res = run_straight_line(n_iter=5000, burn_in=1000, seed=0)
    for p in ("m", "c"):
        post = np.concatenate([w.samples[p][res.burn_in:] for w in res.chain.walkers])
        assert abs(post.mean() - res.ols_mean[p]) < 0.05
        assert abs(post.std() - res.ols_std[p]) < 0.05


def test_chain_shape():
    res = run_straight_line(n_walkers=32, n_iter=500, burn_in=100)
    assert res.chain.n_walkers == 32
    assert res.chain.walkers[0].n_draws == 500
    assert set(res.chain.param_names) == {"m", "c"}
