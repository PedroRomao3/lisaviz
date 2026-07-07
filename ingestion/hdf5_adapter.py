"""Concrete HDF5 repository adapter (Ingestion layer).

Implements :class:`IGalacticBinaryRepository` against the *real* gfrun on-disk
schema confirmed from ``lisa_data_dictionary.txt`` and the HDF5 files:

* ``central_catalog.h5``  -> pandas/PyTables ``cat`` table (8 params + snr +
  Bayes factor + confidence + origin + id). Column names drift between runs
  (``BF``/``bayes_factor``, ``index``/``ID``, optional ``e_*`` errors); a small
  explicit alias map -- not fuzzy matching -- absorbs that (NFR-04).
* ``<band>/new_source/chain_*.npy`` -> flattened parameter-estimation posterior
  (8 sampling params per source slot as ``param:S`` fields plus ``logL, logP``;
  the parallel-tempered walkers are already merged by the pipeline's
  ``Sampler._dump_chain``). Preferred when the PE stage has written it.
* ``<band>/fstats/chain_*.npy`` -> structured F-statistic search chain over the
  4 intrinsic params in hypercube space (``fr:s, fdot:s, dec_sin:s, alpha:s,
  logL, logP``). Legacy fallback. Both kinds are read via mmap and sliced so
  peak RAM is bounded to one chunk.
* ``noise_pars.h5`` / ``mbhb.h5`` -> pandas tables.

READ-ONLY: this adapter never writes back to the run.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Dict, List, Optional

import numpy as np

from ..domain.models import (
    CatalogSource,
    ExtrinsicParameters,
    FrequencyBand,
    FstatGrid,
    GalacticBinary,
    GBCatalog,
    GBChain,
    IntrinsicParameters,
    MBHBCatalog,
    MCMCChain,
    MCMCDraw,
    NoiseModel,
    NOISE_PARAM_NAMES,
)
from ..domain.repository import IGalacticBinaryRepository
from ..domain.sampling import intrinsic_chain_to_physical
from . import pytables

# Physical-name -> ordered list of accepted on-disk aliases.
_CATALOG_ALIASES: Dict[str, List[str]] = {
    "Amplitude": ["Amplitude", "amplitude"],
    "Frequency": ["Frequency", "frequency"],
    "FrequencyDerivative": ["FrequencyDerivative", "frequency_derivative"],
    "RightAscension": ["RightAscension", "right_ascension"],
    "Declination": ["Declination", "declination"],
    "Inclination": ["Inclination", "inclination"],
    "Polarization": ["Polarization", "polarization"],
    "InitialPhase": ["InitialPhase", "initial_phase"],
    "snr": ["snr", "SNR"],
    "bayes_factor": ["bayes_factor", "BF"],
    "confidence": ["confidence"],
    "origin": ["origin"],
    "id": ["ID", "index", "id"],
}

_BAND_RE = re.compile(r"gb-([0-9.]+)-([0-9.]+)")


class HDF5RepositoryAdapter(IGalacticBinaryRepository):
    def __init__(self, run_dir: str, iteration: Optional[int] = None):
        if not os.path.isdir(run_dir):
            raise FileNotFoundError(f"Run directory '{run_dir}' does not exist.")
        self.run_dir = run_dir
        self.iteration = iteration if iteration is not None else self._latest_iteration()
        # Fail fast: any global catalog we will stream must be chunk-compatible.
        cat = self._global_catalog_path()
        if cat is not None:
            pytables.validate_streamable(cat)

    # -- run layout helpers ---------------------------------------------- #
    def list_iterations(self) -> List[int]:
        """All global-fit iteration directories present under the run, sorted."""
        return sorted(
            int(n)
            for n in os.listdir(self.run_dir)
            if n.isdigit() and os.path.isdir(os.path.join(self.run_dir, n))
        )

    def _latest_iteration(self) -> int:
        iters = self.list_iterations()
        if not iters:
            return 1
        # Prefer the latest iteration that actually contains GB sub-bands;
        # later iterations may hold only MBHB/noise blocks.
        with_gb = [
            it
            for it in iters
            if any(
                name.startswith("gb-")
                for name in os.listdir(os.path.join(self.run_dir, str(it)))
            )
        ]
        return max(with_gb) if with_gb else max(iters)

    def _iter_dir(self) -> str:
        return os.path.join(self.run_dir, str(self.iteration))

    def _global_catalog_path(self) -> Optional[str]:
        for cand in (
            os.path.join(self.run_dir, "data", "central_catalog.h5"),
            os.path.join(self.run_dir, "central_catalog.h5"),
        ):
            if os.path.isfile(cand):
                return cand
        return None

    def _catalog_path(self, band: Optional[FrequencyBand]) -> str:
        """The per-band catalog when a band is requested and one exists, else
        the run-level catalog."""
        if band is not None and band.label:
            cand = os.path.join(self._iter_dir(), band.label, "central_catalog.h5")
            if os.path.isfile(cand):
                return cand
        path = self._global_catalog_path()
        if path is None:
            raise FileNotFoundError(f"No central_catalog.h5 found under '{self.run_dir}'.")
        return path

    def _band_dir(self, subband_id: str) -> str:
        path = os.path.join(self._iter_dir(), subband_id)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Sub-band '{subband_id}' not found under '{self._iter_dir()}'.")
        return path

    @staticmethod
    def _parse_band(label: str) -> FrequencyBand:
        m = _BAND_RE.search(label)
        if not m:
            return FrequencyBand(f_min=float("nan"), f_max=float("nan"), label=label)
        # band edges are quoted in mHz in the directory name.
        return FrequencyBand(
            f_min=float(m.group(1)) * 1e-3,
            f_max=float(m.group(2)) * 1e-3,
            label=label,
        )

    # -- interface -------------------------------------------------------- #
    def list_subbands(self) -> List[FrequencyBand]:
        idir = self._iter_dir()
        if not os.path.isdir(idir):
            return []
        labels = sorted(
            n for n in os.listdir(idir)
            if n.startswith("gb-") and os.path.isdir(os.path.join(idir, n))
        )
        return [self._parse_band(lab) for lab in labels]

    def _resolve_column(self, columns, physical_name: str) -> Optional[str]:
        for alias in _CATALOG_ALIASES[physical_name]:
            if alias in columns:
                return alias
        return None

    def get_catalog(self, band: Optional[FrequencyBand] = None) -> GBCatalog:
        df = pytables.read_dataframe(self._catalog_path(band), key="cat")
        cols = set(df.columns)
        col = {name: self._resolve_column(cols, name) for name in _CATALOG_ALIASES}

        sources: List[CatalogSource] = []
        for i, (_, row) in enumerate(df.iterrows()):
            f0 = float(row[col["Frequency"]])
            if band is not None and not (np.isnan(band.f_min) or band.contains(f0)):
                continue
            binary = GalacticBinary(
                intrinsic=IntrinsicParameters(
                    f0=f0,
                    fdot=float(row[col["FrequencyDerivative"]]),
                    declination=float(row[col["Declination"]]),
                    right_ascension=float(row[col["RightAscension"]]),
                ),
                extrinsic=ExtrinsicParameters(
                    amplitude=float(row[col["Amplitude"]]),
                    inclination=float(row[col["Inclination"]]),
                    polarization=float(row[col["Polarization"]]),
                    initial_phase=float(row[col["InitialPhase"]]),
                ),
                snr=float(row[col["snr"]]) if col["snr"] else float("nan"),
                source_id=str(row[col["id"]]) if col["id"] else f"src_{i}",
            )
            sources.append(
                CatalogSource(
                    binary=binary,
                    confidence=float(row[col["confidence"]]) if col["confidence"] else float("nan"),
                    bayes_factor=float(row[col["bayes_factor"]]) if col["bayes_factor"] else float("nan"),
                    origin=str(row[col["origin"]]) if col["origin"] else "",
                )
            )
        return GBCatalog(sources=sources)

    # -- bounded streaming of catalog numeric columns -------------------- #
    def iter_catalog_arrays(self, block_size: int = 50_000, band: Optional[FrequencyBand] = None):
        """Stream the catalog's render-relevant numeric columns in fixed row
        blocks (bounded memory). Yields ``(arrays, start, total)`` where
        ``arrays`` has logical keys (``frequency``, ``right_ascension``,
        ``declination``, ``snr``, ``confidence``, ``amplitude``) for the columns
        present. Used by the bounded sky-map aggregator."""
        from .streaming import iter_catalog_blocks

        path = self._catalog_path(band)
        logical = {
            "frequency": _CATALOG_ALIASES["Frequency"],
            "right_ascension": _CATALOG_ALIASES["RightAscension"],
            "declination": _CATALOG_ALIASES["Declination"],
            "snr": _CATALOG_ALIASES["snr"],
            "confidence": _CATALOG_ALIASES["confidence"],
            "amplitude": _CATALOG_ALIASES["Amplitude"],
        }
        wanted = [name for cands in logical.values() for name in cands]
        for block, start, total in iter_catalog_blocks(path, "cat", wanted, block_size):
            arrays = {}
            for field, cands in logical.items():
                for name in cands:
                    if name in block:
                        arrays[field] = block[name]
                        break
            yield arrays, start, total

    # -- chains ----------------------------------------------------------- #
    # Chain sources in preference order: the flattened parameter-estimation
    # posterior when the PE stage has written it, else the legacy F-statistic
    # search chain. Both share the structured ``param:S`` + ``logL/logP``
    # format, so everything downstream of the glob is identical.
    _CHAIN_DIRS = ("new_source", "fstats")

    def _chain_files(self, subband_id: str) -> List[str]:
        band_dir = self._band_dir(subband_id)
        for sub in self._CHAIN_DIRS:
            files = glob.glob(os.path.join(band_dir, sub, "chain_*.npy"))
            if files:
                files.sort(key=lambda p: int(re.search(r"chain_(\d+)", os.path.basename(p)).group(1)))
                return files
        return []

    def _require_chain_files(self, subband_id: str) -> List[str]:
        files = self._chain_files(subband_id)
        if not files:
            raise FileNotFoundError(
                f"No new_source/ or fstats/ chain_*.npy in sub-band '{subband_id}'.")
        return files

    def has_chain(self, subband_id: str) -> bool:
        """True when this sub-band has at least one ``new_source/chain_*.npy``
        (PE posterior) or ``fstats/chain_*.npy`` (legacy search chain)."""
        return bool(self._chain_files(subband_id))

    def source_indices(self, subband_id: str) -> List[int]:
        """Source-group indices present in this sub-band's chains (e.g. ``[0, 1,
        2]``). Read from the first chain file's structured dtype."""
        files = self._chain_files(subband_id)
        if not files:
            return [0]
        dtype = pytables.open_npy_chain(files[0]).dtype
        groups = pytables.structured_field_groups(dtype)
        return sorted(s for s in groups if s >= 0) or [0]

    def get_chain(
        self, subband_id: str, max_draws: Optional[int] = None, source_index: int = 0
    ) -> MCMCChain:
        band = self._parse_band(subband_id)
        files = self._require_chain_files(subband_id)
        walkers: List[GBChain] = []
        for path in files:
            w_id = int(re.search(r"chain_(\d+)", os.path.basename(path)).group(1))
            data = pytables.open_npy_chain(path)
            stop = data.shape[0] if max_draws is None else min(max_draws, data.shape[0])
            block = np.asarray(data[:stop])  # materialise only the requested slice
            groups = pytables.structured_field_groups(block.dtype)
            # The requested source slot, falling back to the primary (lowest) one.
            chosen = groups.get(
                source_index, groups.get(0, next((g for s, g in groups.items() if s >= 0), {}))
            )
            samples = {base: block[field].astype(float) for base, field in chosen.items()}
            extra = groups.get(-1, {})
            loglik = block[extra["logL"]].astype(float) if "logL" in extra else np.full(stop, np.nan)
            logp = block[extra["logP"]].astype(float) if "logP" in extra else None
            walkers.append(GBChain(walker_id=w_id, samples=samples, log_likelihood=loglik, log_prior=logp, band=band))
        return MCMCChain(walkers=walkers, band=band, nsource_trace=None)

    def get_draw(self, subband_id: str, draw_index: int) -> MCMCDraw:
        band = self._parse_band(subband_id)
        files = self._require_chain_files(subband_id)
        data = pytables.open_npy_chain(files[0])
        row = np.asarray(data[draw_index : draw_index + 1])  # single-row read
        groups = pytables.structured_field_groups(row.dtype)
        loglik = float(row[groups[-1]["logL"]][0]) if -1 in groups and "logL" in groups[-1] else float("nan")
        sources: List[GalacticBinary] = []
        for src, fields in sorted((s, g) for s, g in groups.items() if s >= 0):
            hc = {base: row[field].astype(float) for base, field in fields.items()}
            phys = intrinsic_chain_to_physical(hc, band)
            fdot = phys.get("FrequencyDerivative", phys.get("FrequencyDerivative_hc"))
            # An 8-param PE chain carries the extrinsic parameters too; the
            # 4-param legacy search chain leaves them None.
            extrinsic = None
            if all(k in phys for k in ("Amplitude", "Inclination", "Polarization", "InitialPhase")):
                extrinsic = ExtrinsicParameters(
                    amplitude=float(phys["Amplitude"][0]),
                    inclination=float(phys["Inclination"][0]),
                    polarization=float(phys["Polarization"][0]),
                    initial_phase=float(phys["InitialPhase"][0]),
                )
            sources.append(
                GalacticBinary(
                    intrinsic=IntrinsicParameters(
                        f0=float(phys["Frequency"][0]),
                        fdot=float(fdot[0]) if fdot is not None else float("nan"),
                        declination=float(phys["Declination"][0]),
                        right_ascension=float(phys["RightAscension"][0]),
                    ),
                    extrinsic=extrinsic,
                    snr=float("nan"),
                    source_id=f"{subband_id}_draw{draw_index}_s{src}",
                )
            )
        return MCMCDraw(iteration_index=draw_index, sources=sources, walker_id=0, log_likelihood=loglik)

    # -- F-statistic grid (FR-09) ---------------------------------------- #
    def get_fstat_grid(self, subband_id: str) -> FstatGrid:
        from ..domain.sampling import INTRINSIC_PARS

        path = os.path.join(self._band_dir(subband_id), "fstats.pkl")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"No fstats.pkl in sub-band '{subband_id}'.")
        data = pytables.read_pickle(path)
        grid = np.asarray(data["gridp"], dtype=float)
        fstat = np.asarray(data["fstats"], dtype=float)
        # gridp columns follow the intrinsic sampling order (fr, fdot, dec_sin, alpha).
        names = tuple(INTRINSIC_PARS[: grid.shape[1]])
        return FstatGrid(param_names=names, grid=grid, fstat=fstat, band=self._parse_band(subband_id))

    # -- signal reconstruction (FR-06) ----------------------------------- #
    def get_reconstruction(self, subband_id: str):
        """Read the band's reconstruction products (observed TDI spectrum,
        reconstructed model, noise curve). The on-disk signal-product schema is
        not yet finalised in this adapter, so this raises ``FileNotFoundError``
        when the products are absent; the noise curve itself is available via
        :meth:`get_noise` + ``domain.noise.evaluate_noise_psd``."""
        raise FileNotFoundError(
            f"Signal-reconstruction products for '{subband_id}' are not wired in the "
            "HDF5 adapter yet (the TDI/signal-product schema is pending). The mock "
            "repository supplies a synthetic equivalent for FR-06."
        )

    # -- noise + mbhb ----------------------------------------------------- #
    def _noise_path(self, subband_id: Optional[str]) -> Optional[str]:
        cands = []
        if subband_id:
            cands.append(os.path.join(self._iter_dir(), subband_id, "noise_pars.h5"))
        cands += [
            os.path.join(self.run_dir, "data", "noise_pars.h5"),
            os.path.join(self.run_dir, "noise_pars.h5"),
        ]
        return next((c for c in cands if os.path.isfile(c)), None)

    def get_noise(self, subband_id: Optional[str] = None) -> NoiseModel:
        path = self._noise_path(subband_id)
        if path is None:
            raise FileNotFoundError(f"No noise_pars.h5 found under '{self.run_dir}'.")
        df = pytables.read_dataframe(path, key="cat")
        row = df.iloc[0]
        params = {name: float(row[name]) for name in NOISE_PARAM_NAMES if name in df.columns}
        return NoiseModel(
            parameters=params,
            iteration=self.iteration,
            origin=str(row["__origin__"]) if "__origin__" in df.columns else "",
            model=str(row["__model__"]) if "__model__" in df.columns else "",
        )

    def get_mbhb(self) -> MBHBCatalog:
        for cand in (
            os.path.join(self.run_dir, "data", "mbhb.h5"),
            os.path.join(self.run_dir, "mbhb.h5"),
        ):
            if os.path.isfile(cand):
                keys = pytables.hdf5_keys(cand)
                final = pytables.read_dataframe(cand, "cat").to_dict("records") if "cat" in keys else []
                inj = pytables.read_dataframe(cand, "preobs").to_dict("records") if "preobs" in keys else []
                return MBHBCatalog(final=final, injected=inj)
        return MBHBCatalog()
