import numpy as np
import pytest

from cableanalytics.lossfit import fit_loss, loss_model
from cableanalytics.physics import (
    ALLOYS,
    Design,
    hf_properties,
    temperature_scaling,
)


def test_design_properties_are_physical():
    p = hf_properties(Design())
    assert 95 < p["z_diff"] < 115 and 0.6 < p["nvp"] < 0.85
    assert 1e-5 < p["a"] < 3e-5 and 1e-11 < p["b"] < 1e-10
    # resistivity ordering carries into a; foaming lowers b and raises nvp
    a_cu = hf_properties(Design(alloy="Cu-ETP"))["a"]
    a_mg = hf_properties(Design(alloy="CuMg0.2"))["a"]
    assert a_mg / a_cu == pytest.approx(np.sqrt(ALLOYS["CuMg0.2"][0] / ALLOYS["Cu-ETP"][0]), rel=1e-6)
    assert hf_properties(Design(foaming=0.5))["b"] < hf_properties(Design(foaming=0.1))["b"]


def test_temperature_scaling_matches_material_constants():
    d = Design()
    s = temperature_scaling(d, [105.0])
    # a ~ sqrt(rho)/Z: the resistivity factor times the (small) impedance change with eps(T)
    z_ratio = s["z_ratio"][0]
    assert s["a_ratio"][0] == pytest.approx(np.sqrt(1 + ALLOYS[d.alloy][1] * 85) / z_ratio, rel=1e-6)
    assert s["b_ratio"][0] > 1.3


def test_fit_recovers_coefficients_with_ripple_spikes():
    f = np.linspace(5e6, 1e9, 1500)
    a, b, c0 = 1.7e-5, 2.3e-11, 0.009
    il = 15.0 * loss_model(f, a, b, c0, "physical")
    rng = np.random.default_rng(0)
    il = il + 0.01 * rng.standard_normal(f.size)
    il[700:706] += 0.6                         # a Bragg-type spike
    ft = fit_loss(f, il, 15.0)
    assert abs(ft.a / a - 1) < 0.01 and abs(ft.b / b - 1) < 0.05 and abs(ft.c - c0) < 0.003
    assert ft.sigma[0] / ft.a < 0.01 and ft.corr_ab < -0.8
    # standard basis describes the curve but mis-assigns the DC term
    fs = fit_loss(f, il, 15.0, basis="standard")
    assert fs.rms_residual_db < 0.05 and abs(fs.b / b - 1) > 0.3


def test_loss_model_bases():
    f = np.array([1e6, 1e8])
    assert np.allclose(loss_model(f, 1e-5, 1e-11, 0.01, "physical"), 0.01 + 1e-5 * np.sqrt(f) + 1e-11 * f)
    assert np.allclose(loss_model(f, 1e-5, 1e-11, 0.02, "standard"), 1e-5 * np.sqrt(f) + 1e-11 * f + 0.02 / np.sqrt(f))
