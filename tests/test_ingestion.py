"""Ingestion-layer tests: mock repository, fail-fast chunk inspection, and the
HDF5 adapter against real gfrun data when present."""

import os

import h5py
import numpy as np
import pytest

from lisaviz.domain.repository import IGalacticBinaryRepository
from lisaviz.ingestion.hdf5_adapter import HDF5RepositoryAdapter
from lisaviz.ingestion.mock import InMemoryMockRepository
from lisaviz.ingestion.pytables import ChunkLayoutError, validate_streamable

_GFRUN = os.path.join(os.path.dirname(__file__), "..", "..", "gfrun_old")


def test_mock_implements_interface():
    repo = InMemoryMockRepository(n_catalog=20, n_draws=50)
    assert isinstance(repo, IGalacticBinaryRepository)
    bands = repo.list_subbands()
    assert bands
    cat = repo.get_catalog()
    assert len(cat) == 20
    chain = repo.get_chain(bands[0].label, max_draws=30)
    assert chain.n_walkers >= 1 and chain.walkers[0].n_draws == 30
    draw = repo.get_draw(bands[0].label, 10)
    assert draw.nsource >= 1
    assert repo.get_noise().parameters["fknee"] > 0


def test_max_draws_is_bounded():
    repo = InMemoryMockRepository(n_catalog=5, n_draws=200)
    label = repo.list_subbands()[0].label
    assert repo.get_chain(label, max_draws=25).walkers[0].n_draws == 25


def test_validate_streamable_rejects_single_row_chunks(tmp_path):
    bad = tmp_path / "bad.h5"
    with h5py.File(bad, "w") as f:
        f.create_dataset("d", data=np.zeros((10, 3)), chunks=(1, 3))
    with pytest.raises(ChunkLayoutError):
        validate_streamable(str(bad))


def test_validate_streamable_accepts_contiguous(tmp_path):
    good = tmp_path / "good.h5"
    with h5py.File(good, "w") as f:
        f.create_dataset("d", data=np.zeros((10, 3)))  # contiguous
    validate_streamable(str(good))  # must not raise


def test_catalog_absorbs_snake_case_schema(tmp_path):
    """NFR-04 / Sec 4.4: the L2-to-L3 migration renames catalog columns to
    snake_case (e.g. ``right_ascension``). The adapter must absorb that drift
    without any change to the domain or visualization layers."""
    pd = pytest.importorskip("pandas")
    df = pd.DataFrame(
        {
            "frequency": [3.0e-3],
            "frequency_derivative": [1.0e-16],
            "right_ascension": [1.0],
            "declination": [0.1],
            "amplitude": [1.0e-22],
            "inclination": [1.0],
            "polarization": [0.5],
            "initial_phase": [0.3],
            "snr": [10.0],
            "bayes_factor": [5.0],
            "confidence": [0.9],
            "ID": [0],
        }
    )
    (tmp_path / "data").mkdir()
    (tmp_path / "1" / "gb-3.049-3.053").mkdir(parents=True)
    df.to_hdf(tmp_path / "data" / "central_catalog.h5", key="cat", mode="w")
    df.to_hdf(tmp_path / "1" / "gb-3.049-3.053" / "central_catalog.h5", key="cat", mode="w")

    cat = HDF5RepositoryAdapter(str(tmp_path)).get_catalog()
    assert len(cat) == 1
    src = cat.sources[0]
    assert src.binary.intrinsic.f0 == 3.0e-3
    assert src.binary.intrinsic.right_ascension == 1.0
    assert src.confidence == 0.9


# The 8 sampling-space parameters of one GB parameter-estimation source slot
# (globalfit/gb/likelihood.py PARS_NAMES).
_PE_PARS = ("amp_log10", "fr", "fdot", "dec_sin", "alpha", "iota_cos", "phiL", "phiR")


def _write_chain(path, names, n_draws=300, seed=0):
    """A synthetic chain in the exact on-disk format of the pipeline's
    ``Sampler._dump_chain``: a structured 1-D ``.npy`` with one float64 field
    per sampling parameter (``param:slot``) plus the ``logL, logP`` extras,
    walkers already flattened."""
    rng = np.random.default_rng(seed)
    dtype = np.dtype([(nm, "<f8") for nm in [*names, "logL", "logP"]])
    chn = np.zeros(n_draws, dtype=dtype)
    for nm in names:
        chn[nm] = rng.random(n_draws)  # sampling-space values in [0, 1)
    chn["logL"] = rng.normal(100.0, 1.0, n_draws)
    chn["logP"] = chn["logL"] + rng.normal(0.0, 1.0, n_draws)
    np.save(path, chn)


class TestPEChainIngestion:
    """Tier-1 chain ingestion: the adapter prefers the flattened
    parameter-estimation posterior (``new_source/``) and falls back to the
    legacy F-statistic search chain (``fstats/``)."""

    _BAND = "gb-3.049-3.053"

    def _make_run(self, tmp_path, with_fstats=True, nsrc=2, n_draws=300):
        band = tmp_path / "1" / self._BAND
        (band / "new_source").mkdir(parents=True)
        pe_names = [f"{p}:{s}" for s in range(nsrc) for p in _PE_PARS]
        _write_chain(band / "new_source" / "chain_0.npy", pe_names, n_draws)
        if with_fstats:
            (band / "fstats").mkdir()
            fs_names = [f"{p}:0" for p in ("fr", "fdot", "dec_sin", "alpha")]
            _write_chain(band / "fstats" / "chain_0.npy", fs_names, n_draws)
        return HDF5RepositoryAdapter(str(tmp_path))

    def test_pe_posterior_preferred_over_fstats(self, tmp_path):
        repo = self._make_run(tmp_path, with_fstats=True)
        chain = repo.get_chain(self._BAND, max_draws=100)
        assert set(chain.param_names) == set(_PE_PARS)  # 8 params, not the 4 intrinsic
        assert chain.n_walkers == 1  # one flattened file, walkers merged upstream
        assert chain.walkers[0].n_draws == 100
        assert np.isfinite(chain.walkers[0].log_likelihood).all()

    def test_source_slots_read_from_pe_dtype(self, tmp_path):
        repo = self._make_run(tmp_path, nsrc=2)
        assert repo.source_indices(self._BAND) == [0, 1]
        chain1 = repo.get_chain(self._BAND, source_index=1)
        assert set(chain1.param_names) == set(_PE_PARS)

    def test_get_draw_populates_extrinsic_from_pe_chain(self, tmp_path):
        repo = self._make_run(tmp_path, nsrc=2)
        draw = repo.get_draw(self._BAND, 10)
        assert draw.nsource == 2
        gb = draw.sources[0]
        assert gb.extrinsic is not None
        assert gb.extrinsic.amplitude > 0
        assert 0.0 <= gb.extrinsic.inclination <= np.pi
        assert 0.0 <= gb.extrinsic.polarization <= np.pi
        assert 3.049e-3 <= gb.intrinsic.f0 <= 3.053e-3

    def test_per_walker_files_upgrade_diagnostics(self, tmp_path):
        """If a future pipeline update writes one chain file per walker,
        the adapter loads them as a multi-walker chain and the diagnostics
        upgrade to the real between-walker R-hat with no code change."""
        band = tmp_path / "1" / self._BAND
        (band / "new_source").mkdir(parents=True)
        pe_names = [f"{p}:0" for p in _PE_PARS]
        _write_chain(band / "new_source" / "chain_0.npy", pe_names, seed=0)
        _write_chain(band / "new_source" / "chain_1.npy", pe_names, seed=1)
        repo = HDF5RepositoryAdapter(str(tmp_path))
        chain = repo.get_chain(self._BAND)
        assert chain.n_walkers == 2
        assert [w.walker_id for w in chain.walkers] == [0, 1]
        from lisaviz.visualization.stats import diagnostics
        diag = diagnostics(chain, burn_in=10)
        assert all(not d.rhat_is_split for d in diag.values())  # true between-walker R-hat

    def test_fstats_fallback_when_pe_absent(self, tmp_path):
        # new_source/ exists but is empty, as in the local mid-migration runs.
        band = tmp_path / "1" / self._BAND
        (band / "new_source").mkdir(parents=True)
        (band / "fstats").mkdir()
        _write_chain(band / "fstats" / "chain_0.npy",
                     [f"{p}:0" for p in ("fr", "fdot", "dec_sin", "alpha")])
        repo = HDF5RepositoryAdapter(str(tmp_path))
        chain = repo.get_chain(self._BAND)
        assert set(chain.param_names) == {"fr", "fdot", "dec_sin", "alpha"}
        assert repo.get_draw(self._BAND, 5).sources[0].extrinsic is None


@pytest.mark.skipif(not os.path.isdir(_GFRUN), reason="gfrun_old data not available")
class TestRealAdapter:
    def setup_method(self):
        self.repo = HDF5RepositoryAdapter(_GFRUN)

    def test_lists_gb_bands(self):
        bands = self.repo.list_subbands()
        assert bands and all(b.label.startswith("gb-") for b in bands)
        assert bands[0].f_min < bands[0].f_max

    def test_catalog_real_schema(self):
        cat = self.repo.get_catalog()
        assert len(cat) > 0
        s = cat.sources[0]
        assert s.binary.intrinsic.f0 > 0
        assert s.binary.source_id  # id resolved (ID or index)

    def test_chain_is_four_intrinsic_params(self):
        label = self.repo.list_subbands()[0].label
        chain = self.repo.get_chain(label, max_draws=100)
        assert set(chain.param_names) == {"fr", "fdot", "dec_sin", "alpha"}
        assert chain.walkers[0].n_draws == 100
        assert np.isfinite(chain.walkers[0].log_likelihood).all()

    def test_get_draw_returns_physical(self):
        label = self.repo.list_subbands()[0].label
        band = self.repo.list_subbands()[0]
        draw = self.repo.get_draw(label, 50)
        gb = draw.sources[0]
        assert band.f_min <= gb.intrinsic.f0 <= band.f_max
        assert -np.pi / 2 <= gb.intrinsic.declination <= np.pi / 2

    def test_iter_catalog_arrays_streams_correctly(self):
        # Streaming in small blocks must visit every source exactly once and
        # agree with the whole-catalog read.
        full = self.repo.get_catalog()
        seen = 0
        fsum_stream = 0.0
        for arrays, start, total in self.repo.iter_catalog_arrays(block_size=512):
            seen += arrays["frequency"].size
            fsum_stream += float(arrays["frequency"].sum())
            assert total == len(full)
        assert seen == len(full)
        fsum_full = sum(s.binary.intrinsic.f0 for s in full.sources)
        assert abs(fsum_stream - fsum_full) < 1e-6 * abs(fsum_full)

    def test_fstat_grid_real(self):
        label = self.repo.list_subbands()[0].label
        grid = self.repo.get_fstat_grid(label)
        assert grid.grid.shape[1] == 4 and grid.grid.shape[0] == grid.fstat.size
        assert "fr" in grid.variable_axes()
        assert grid.fstat.max() > 1.0  # a real detection surface has peaks
