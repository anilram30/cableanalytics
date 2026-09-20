import numpy as np
import pytest

from cableanalytics.lab import temperature_sweep
from cableanalytics.periodic import attribute, period_from_profile
from cableanalytics.physics import ALLOYS, INSULATIONS, hf_properties
from cableanalytics.production import DESIGNS, LINES


@pytest.fixture(scope="module")
def sweep(tmp_path_factory):
    d = DESIGNS["D100-PE-035"]
    line = LINES["L2"]
    return d, line, temperature_sweep(d, 15.0, tmp_path_factory.mktemp("lab"), temperatures=(23, 85, 125), n_points=801,
                                      line={"line_speed_m_min": line.line_speed_m_min, "screw_rpm": line.screw_rpm,
                                            "capstan_d_m": line.capstan_d_m, "ripple_period_m": np.pi * line.capstan_d_m,
                                            "ripple_rel": 0.006})


def test_fits_track_the_model_at_each_temperature(sweep):
    d, line, res = sweep
    for T, ft in zip(res.temperatures, res.fits):
        p = hf_properties(d, T)
        assert abs(ft.a / p["a"] - 1) < 0.01, T
        assert abs(ft.b / p["b"] - 1) < 0.05, T


def test_derating_recovers_material_coefficients(sweep):
    d, line, res = sweep
    der = res.derating
    assert abs(der.beta_d / INSULATIONS[d.insulation][2] - 1) < 0.05
    assert abs(der.alpha_rho / ALLOYS[d.alloy][1] - 1) < 0.15     # effective (includes Z(T)); documented
    assert 1.10 < der.derating(600e6, 105.0) < 1.18


def test_bragg_period_and_capstan_attribution(sweep):
    d, line, res = sweep
    assert res.period is not None
    assert abs(res.period["period_m"] / (np.pi * line.capstan_d_m) - 1) < 0.02
    assert res.attribution.best["element"] == "capstan" and not res.attribution.ambiguous


def test_attribution_screw_and_harmonics():
    a = attribute(50 / 38, 50.0, 38.0, 0.5)
    assert a.best["element"] == "screw" and a.best["harmonic"] == 1
    a2 = attribute(np.pi * 0.5 / 2, 50.0, 38.0, 0.5)
    assert a2.best["element"] == "capstan" and a2.best["harmonic"] == 2
    a3 = attribute(0.123, 50.0, 38.0, 0.5)
    assert a3.best is None


def test_period_from_profile():
    x = np.linspace(0, 15, 1500)
    z = 100 + 1.2 * np.sin(2 * np.pi * x / 0.26) + 0.05 * np.random.default_rng(1).standard_normal(x.size)
    r = period_from_profile(x, z, 0.05)
    assert abs(r["period_m"] - 0.26) < 0.01 and abs(r["amplitude_ohm"] - 1.2) < 0.2
