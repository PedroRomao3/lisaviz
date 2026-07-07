"""Domain-layer unit tests (no filesystem, no real HDF5)."""

import math

import numpy as np
import pytest

from lisaviz.domain import confidence, convergence, sampling
from lisaviz.domain.models import (
    ExtrinsicParameters,
    FrequencyBand,
    GalacticBinary,
    IntrinsicParameters,
    MCMCDraw,
    NoiseModel,
)


def _gb(sid="a", f0=1e-3):
    return GalacticBinary(
        intrinsic=IntrinsicParameters(f0=f0, fdot=0.0, declination=0.1, right_ascension=1.0),
        extrinsic=ExtrinsicParameters(amplitude=1e-22, inclination=0.5, polarization=0.5, initial_phase=0.5),
        snr=10.0, source_id=sid,
    )


def test_value_objects_immutable():
    p = IntrinsicParameters(f0=1e-3, fdot=0.0, declination=0.0, right_ascension=0.0)
    with pytest.raises(Exception):
        p.f0 = 2e-3  # frozen


def test_galactic_binary_is_entity_identity_based():
    a1 = _gb("same", f0=1e-3)
    a2 = _gb("same", f0=9e-3)  # different params, same identity
    b = _gb("other")
    assert a1 == a2 and hash(a1) == hash(a2)
    assert a1 != b
    assert len({a1, a2, b}) == 2


def test_mcmcdraw_variable_cardinality():
    assert MCMCDraw(0, [_gb("a"), _gb("b")]).nsource == 2
    assert MCMCDraw(1, []).nsource == 0


def test_noise_model_requires_all_params():
    with pytest.raises(ValueError):
        NoiseModel(parameters={"Sacc_log10": -29.0})


def test_rescale_scale_roundtrip():
    hc = np.array([0.0, 0.25, 0.5, 1.0])
    phys = sampling.rescale(hc, 3.0, 7.0)
    np.testing.assert_allclose(sampling.scale(phys, 3.0, 7.0), hc, atol=1e-12)
    # endpoints map to the prior interval
    np.testing.assert_allclose(phys[[0, -1]], [3.0, 7.0])


def test_declination_transform_matches_arcsin():
    band = FrequencyBand(f_min=3.049e-3, f_max=3.053e-3, label="gb")
    dec_sin_hc = np.array([0.5])  # rescales to 0.0 -> arcsin(0) = 0
    out = sampling.intrinsic_chain_to_physical({"dec_sin": dec_sin_hc, "fr": np.array([0.5]),
                                                "alpha": np.array([0.5]), "fdot": np.array([0.5])}, band)
    assert abs(float(out["Declination"][0])) < 1e-12
    assert band.f_min <= float(out["Frequency"][0]) <= band.f_max


def test_confidence_pipeline():
    # high SNR -> high base confidence
    assert confidence.compute_confidence(40.0) > 0.99
    # previous-overlap replaces the sigmoid start
    assert confidence.compute_confidence(40.0, previous_overlap=0.5) == pytest.approx(0.5)
    # confusion penalty subtracts and floors at 0
    assert confidence.compute_confidence(40.0, neighbour_overlaps=[5.0]) == 0.0
    np.testing.assert_allclose(confidence.mark_size(np.array([0.0, 1.0])), [1.0, 0.0])


def test_convergence_flags_anomalously_low_band():
    scores = {"b1": -900.0, "b2": -901.0, "b3": -902.0, "b4": -1500.0}
    report = convergence.flag_low_bands(scores)
    assert "b4" in report.flagged
    assert "b1" not in report.flagged
