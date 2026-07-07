"""In-memory mock repository (Ingestion layer).

Returns synthetic Domain objects so the Domain and Visualization layers are
testable with no filesystem and no real HDF5. The synthetic data deliberately
exercises the hard cases: a trans-dimensional (variable) source count per draw,
multiple walkers, and catalog confidence/Bayes-factor/origin fields.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from ..domain.confidence import sigmoid
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
    ReconstructionData,
)
from ..domain.noise import evaluate_noise_psd
from ..domain.repository import IGalacticBinaryRepository
from ..domain.sampling import PARS_NAMES


class InMemoryMockRepository(IGalacticBinaryRepository):
    def __init__(
        self,
        n_catalog: int = 500,
        n_walkers: int = 4,
        n_draws: int = 5000,
        seed: int = 42,
        n_sources: int = 3,
    ):
        self.rng = np.random.default_rng(seed)
        self._n_draws = n_draws
        self._n_walkers = n_walkers
        self._n_sources = max(1, n_sources)
        # A small set of synthetic sub-bands around the gfrun example band.
        self._bands: List[FrequencyBand] = [
            FrequencyBand(
                f_min=3.049e-3 + i * 2e-6,
                f_max=3.053e-3 + i * 2e-6,
                label=f"gb-{3.049 + i * 0.002:.3f}-{3.053 + i * 0.002:.3f}",
                fdot_min=1e-18,
                fdot_max=1e-15,
            )
            for i in range(3)
        ]
        self._catalog = self._generate_catalog(n_catalog)
        # Each band holds several source slots (trans-dimensional reality): slot 0
        # is a well-measured source; later slots are progressively lower-SNR, with
        # the last one wide / poorly-constrained -- the case a source selector
        # exists to inspect.
        self._chains: Dict[str, List[MCMCChain]] = {
            b.label: self._generate_chains(b) for b in self._bands
        }

    # -- catalog ---------------------------------------------------------- #
    def _generate_catalog(self, n: int) -> GBCatalog:
        sources: List[CatalogSource] = []
        # Reserve a few slots for sources anchored inside the chain bands (added
        # below) so the catalog total stays exactly ``n``.
        n_anchor = min(len(self._bands) * 2, n)
        n_random = n - n_anchor
        for i in range(n_random):
            snr = math.exp(self.rng.uniform(math.log(5.0), math.log(50.0)))
            dec = float(np.clip(self.rng.normal(0.0, 0.3), -math.pi / 2, math.pi / 2))
            ra = float(self.rng.uniform(0.0, 2 * math.pi))
            f0 = float(math.exp(self.rng.uniform(math.log(1e-4), math.log(1e-2))))
            fdot = 1e-16 * (f0 / 1e-3) ** (11.0 / 3.0) * float(self.rng.uniform(0.5, 2.0))
            amp = float(math.exp(self.rng.uniform(math.log(1e-23), math.log(1e-21))))
            binary = GalacticBinary(
                intrinsic=IntrinsicParameters(f0=f0, fdot=fdot, declination=dec, right_ascension=ra),
                extrinsic=ExtrinsicParameters(
                    amplitude=amp,
                    inclination=float(self.rng.uniform(0.0, math.pi)),
                    polarization=float(self.rng.uniform(0.0, math.pi)),
                    initial_phase=float(self.rng.uniform(0.0, 2 * math.pi)),
                ),
                snr=snr,
                source_id=f"mock_{i:04d}",
            )
            confidence = float(np.clip(sigmoid(snr) - self.rng.uniform(0.0, 0.2), 0.0, 1.0))
            sources.append(
                CatalogSource(
                    binary=binary,
                    confidence=confidence,
                    bayes_factor=float(math.exp(self.rng.uniform(0.0, 6.0))),
                    origin="prev" if self.rng.random() < 0.4 else "new",
                )
            )
        # Anchor a couple of clearly-clickable catalog balls *inside* each chain
        # band, so clicking one in the sky map / waterfall lands on a band that
        # actually has MCMC chains to show (the random sources above almost never
        # fall in these narrow bands). Capped to the reserved budget.
        anchored = 0
        for band in self._bands:
            for k in range(2):
                if anchored >= n_anchor:
                    break
                anchored += 1
                f0 = float(band.f_min + (k + 1) / 3.0 * (band.f_max - band.f_min))
                snr = 30.0 - 8.0 * k
                bin_ = GalacticBinary(
                    intrinsic=IntrinsicParameters(
                        f0=f0,
                        fdot=1e-16 * (f0 / 1e-3) ** (11.0 / 3.0),
                        declination=float(np.clip(self.rng.normal(0.0, 0.3), -math.pi / 2, math.pi / 2)),
                        right_ascension=float(self.rng.uniform(0.0, 2 * math.pi)),
                    ),
                    extrinsic=ExtrinsicParameters(
                        amplitude=float(math.exp(self.rng.uniform(math.log(1e-22), math.log(1e-21)))),
                        inclination=float(self.rng.uniform(0.0, math.pi)),
                        polarization=float(self.rng.uniform(0.0, math.pi)),
                        initial_phase=float(self.rng.uniform(0.0, 2 * math.pi)),
                    ),
                    snr=snr,
                    source_id=f"{band.label}_src{k}",
                )
                sources.append(
                    CatalogSource(binary=bin_, confidence=float(sigmoid(snr)),
                                  bayes_factor=float(math.exp(self.rng.uniform(2.0, 6.0))),
                                  origin="new")
                )
        return GBCatalog(sources=sources)

    # -- chains ----------------------------------------------------------- #
    def _generate_chains(self, band: FrequencyBand) -> List[MCMCChain]:
        """The source slots of one sub-band: a list of multi-walker chains, one
        per source. The band-level trans-dimensional ``nsource`` trajectory is
        shared across slots (it describes the band, not a single source)."""
        # Band-level trans-dimensional source count: integer that jumps over time.
        nsource_rows = []
        for _ in range(self._n_walkers):
            nsrc = 1 + (np.cumsum(self.rng.random(self._n_draws) < 0.0008)).astype(int)
            nsource_rows.append(np.clip(nsrc, 1, 5))
        nsource_trace = np.vstack(nsource_rows)
        return [
            self._generate_source_chain(band, src, nsource_trace)
            for src in range(self._n_sources)
        ]

    def _generate_source_chain(
        self, band: FrequencyBand, source_index: int, nsource_trace: np.ndarray
    ) -> MCMCChain:
        """One source slot: a multi-walker chain in sampling (hypercube) space.

        ``source_index`` sets the measurement quality: slot 0 is high-SNR (tight
        posterior, high log L, crisp mixing); the last slot is low-SNR (wide,
        possibly bimodal posterior, lower log L) -- the poorly-constrained source
        you would select to inspect why it 'didn't differentiate well'."""
        # Quality ramps from well-measured (slot 0) to poorly-measured (last slot).
        frac = source_index / max(1, self._n_sources - 1)  # 0 -> 1
        width = 0.02 + 0.13 * frac          # posterior/walker spread
        loglik_level = -925.0 - 70.0 * frac  # plateau height (lower = lower SNR)
        decay_rate = 3.0 - 1.6 * frac        # slower burn-in for low-SNR slots
        bimodal = frac > 0.66               # the worst slot splits into two modes

        targets = {p: float(self.rng.uniform(0.35, 0.65)) for p in PARS_NAMES}
        # An alternate mode the low-SNR walkers may settle into (label-switching).
        alt = {p: float(np.clip(targets[p] + self.rng.uniform(-0.25, 0.25), 0.05, 0.95))
               for p in PARS_NAMES}

        walkers: List[GBChain] = []
        t = np.linspace(0.0, 1.0, self._n_draws)
        decay = np.exp(-decay_rate * t)
        for w in range(self._n_walkers):
            tgt = alt if (bimodal and w % 2 == 0) else targets
            samples: Dict[str, np.ndarray] = {}
            for p in PARS_NAMES:
                start = float(self.rng.uniform(0.0, 1.0))
                noise = self.rng.normal(0.0, width, self._n_draws).cumsum() * 0.01
                series = tgt[p] + (start - tgt[p]) * decay + noise
                samples[p] = np.clip(series, 0.0, 1.0)
            loglik = loglik_level + 25.0 * (1.0 - decay) + self.rng.normal(0.0, 1.0 + 2.0 * frac, self._n_draws)
            logprior = np.full(self._n_draws, loglik_level - 10.0) + self.rng.normal(0.0, 0.5, self._n_draws)
            walkers.append(
                GBChain(walker_id=w, samples=samples, log_likelihood=loglik, log_prior=logprior, band=band)
            )
        return MCMCChain(walkers=walkers, band=band, nsource_trace=nsource_trace)

    # -- interface -------------------------------------------------------- #
    def list_subbands(self) -> List[FrequencyBand]:
        return list(self._bands)

    def _band_for(self, subband_id: str) -> FrequencyBand:
        for b in self._bands:
            if b.label == subband_id:
                return b
        raise KeyError(f"Unknown sub-band '{subband_id}'.")

    def get_catalog(self, band: Optional[FrequencyBand] = None) -> GBCatalog:
        if band is None:
            return self._catalog
        sel = [s for s in self._catalog.sources if band.contains(s.binary.intrinsic.f0)]
        return GBCatalog(sources=sel)

    def source_indices(self, subband_id: str) -> List[int]:
        return list(range(len(self._chains[subband_id])))

    def get_chain(
        self, subband_id: str, max_draws: Optional[int] = None, source_index: int = 0
    ) -> MCMCChain:
        slots = self._chains[subband_id]
        chain = slots[source_index if 0 <= source_index < len(slots) else 0]
        if max_draws is None or max_draws >= chain.walkers[0].n_draws:
            return chain
        walkers = [
            GBChain(
                walker_id=w.walker_id,
                samples={k: v[:max_draws] for k, v in w.samples.items()},
                log_likelihood=w.log_likelihood[:max_draws],
                log_prior=None if w.log_prior is None else w.log_prior[:max_draws],
                band=w.band,
            )
            for w in chain.walkers
        ]
        nst = None if chain.nsource_trace is None else chain.nsource_trace[:, :max_draws]
        return MCMCChain(walkers=walkers, band=chain.band, nsource_trace=nst)

    def get_draw(self, subband_id: str, draw_index: int) -> MCMCDraw:
        from ..domain.sampling import intrinsic_chain_to_physical

        chain = self._chains[subband_id][0]
        band = self._band_for(subband_id)
        w0 = chain.walkers[0]
        nsrc = 1
        if chain.nsource_trace is not None:
            nsrc = int(chain.nsource_trace[0, draw_index])
        sources: List[GalacticBinary] = []
        for s in range(nsrc):
            hc = {p: np.array([w0.samples[p][draw_index]]) for p in PARS_NAMES}
            phys = intrinsic_chain_to_physical(hc, band)
            sources.append(
                GalacticBinary(
                    intrinsic=IntrinsicParameters(
                        f0=float(phys["Frequency"][0]),
                        fdot=float(phys.get("FrequencyDerivative", phys.get("FrequencyDerivative_hc"))[0]),
                        declination=float(phys["Declination"][0]),
                        right_ascension=float(phys["RightAscension"][0]),
                    ),
                    extrinsic=ExtrinsicParameters(
                        amplitude=float(phys["Amplitude"][0]),
                        inclination=float(phys["Inclination"][0]),
                        polarization=float(phys["Polarization"][0]),
                        initial_phase=float(phys["InitialPhase"][0]),
                    ),
                    snr=float("nan"),
                    source_id=f"{subband_id}_draw{draw_index}_s{s}",
                )
            )
        return MCMCDraw(
            iteration_index=draw_index,
            sources=sources,
            walker_id=0,
            log_likelihood=float(w0.log_likelihood[draw_index]),
        )

    def get_fstat_grid(self, subband_id: str) -> FstatGrid:
        """Synthesize a regular intrinsic-parameter grid whose F-statistic has a
        diagonal ``fr``-``dec_sin`` degeneracy ridge plus a sharper peak, so the
        contour view (FR-09) shows a realistic degeneracy."""
        band = self._band_for(subband_id)
        n_fr, n_ds, n_al = 80, 40, 12
        fr = np.linspace(0.05, 0.95, n_fr)
        ds = np.linspace(0.05, 0.95, n_ds)
        al = np.linspace(0.05, 0.95, n_al)
        FR, DS, AL = np.meshgrid(fr, ds, al, indexing="ij")
        ridge = np.exp(-((FR - DS) ** 2) / (2 * 0.05 ** 2))  # fr ~ dec_sin degeneracy
        peak = 80.0 * np.exp(-(((FR - 0.6) ** 2 + (DS - 0.55) ** 2) / (2 * 0.02 ** 2)))
        fstat = 5.0 + 30.0 * ridge + peak * (0.6 + 0.4 * np.cos(2 * np.pi * AL))
        grid = np.column_stack([FR.ravel(), np.full(FR.size, 0.5), DS.ravel(), AL.ravel()])
        return FstatGrid(param_names=("fr", "fdot", "dec_sin", "alpha"),
                         grid=grid, fstat=fstat.ravel() + self.rng.normal(0, 0.3, FR.size), band=band)

    def get_noise(self, subband_id: Optional[str] = None) -> NoiseModel:
        params = {
            "Sacc_log10": -44.0,
            "Soms_log10": -40.0,
            "A_log10": -7.5,
            "f1": 4.0e-4,
            "f2": 1.0e-3,
            "alpha": 1.6,
            "fknee": 2.0e-3,
        }
        assert set(NOISE_PARAM_NAMES).issubset(params)
        return NoiseModel(parameters=params, iteration=1, origin="mock", model="instrument+confusion")

    def get_reconstruction(self, subband_id: str) -> ReconstructionData:
        """Synthesize band-aware signal-reconstruction inputs (FR-06): a wide PSD
        residual (noise model + a GB-line forest, with periodogram scatter) and a
        narrow-band waveform overlay with recovered + injected lines."""
        band = self._band_for(subband_id)
        rng = np.random.default_rng(int(band.f_min * 1e9) % (2 ** 32))

        # PSD residual over a wide window so the noise shape is visible.
        freq = np.geomspace(3e-4, 1e-2, 4000)
        noise_psd = evaluate_noise_psd(self.get_noise(subband_id), freq)
        forest = np.zeros_like(freq)
        for _ in range(60):
            lf = float(np.exp(rng.uniform(np.log(5e-4), np.log(8e-3))))
            amp = noise_psd[np.argmin(np.abs(freq - lf))] * rng.uniform(3, 40)
            forest += amp * np.exp(-0.5 * ((freq - lf) / (lf * 6e-4)) ** 2)
        model_psd = noise_psd + forest
        observed_psd = model_psd * rng.exponential(1.0, freq.size)  # ~chi-square periodogram
        noise_evo = [(f"noise iter {k}", noise_psd * (1.0 + 0.3 * (3 - k))) for k in range(1, 4)]

        # Waveform overlay over a narrow window centred on the band.
        fc = band.f_center
        w = max((band.f_max - band.f_min) * 0.25, 1e-7)
        of = np.linspace(fc - 6 * w, fc + 6 * w, 2000)

        def _line(f0, width, amp, phase=0.0):
            env = amp * np.exp(-0.5 * ((of - f0) / width) ** 2)
            return env * np.exp(1j * (phase + 2 * np.pi * (of - f0) / (4 * width)))

        namp = 1.2e-23
        injection = _line(fc, w, 2.2e-22)
        noise = namp * (rng.standard_normal(of.size) + 1j * rng.standard_normal(of.size))
        observed_spectrum = injection + noise
        recovered = [_line(fc, w, 2.18e-22, 0.03), _line(fc + 0.8 * w, w * 1.1, 1.95e-22, 0.4)]
        return ReconstructionData(
            freq=freq, observed_psd=observed_psd, model_psd=model_psd, noise_psd=noise_psd,
            noise_evolution=noise_evo, overlay_freq=of, observed_spectrum=observed_spectrum,
            recovered=recovered, injection=injection, overlay_psd=np.full_like(of, namp ** 2),
            labels=["recovered (best)", "recovered (offset)"], band=band)

    def get_mbhb(self) -> MBHBCatalog:
        def row(seed: float) -> Dict[str, float]:
            return {
                "Mchirp": 1e6 * (1 + seed), "q": 0.5 + 0.1 * seed, "chi1": 0.3, "chi2": 0.2,
                "Deltat": 1e6 * seed, "dist": 1e4, "inc": 1.0, "phi": 2.0,
                "lambda": 3.0, "beta": 0.2, "psi": 1.2,
            }
        return MBHBCatalog(final=[row(0.0), row(0.5)], injected=[row(0.01), row(0.51)])
