"""
Design -> high-frequency properties: the physics that links how a cable is
built to the three loss coefficients, its impedance and its velocity.

Loss shape
----------
Per unit length, in dB/m, the insertion loss of a balanced pair follows

    IL(f) = a sqrt(f) + b f + c / sqrt(f)                      (f in Hz)

* a sqrt(f): conductor (skin-effect) loss.  With the loop resistance
  R'(f) = 2 k_prox sqrt(pi f mu0 rho) / (pi d_c)  and  alpha_c = R'/(2 Z_d),
      a = 8.686 * k_prox * sqrt(mu0 rho / pi) / (d_c Z_d)     [dB/m/sqrt(Hz)]
  so a grows with the square root of the resistivity of the alloy and
  inversely with conductor diameter and impedance.
* b f: dielectric loss, alpha_d = pi f sqrt(eps_eff) tan(delta_eff) / c0,
      b = 8.686 * pi * sqrt(eps_eff) * tan(delta_eff) / c0     [dB/m/Hz]
* c / sqrt(f): the empirical low-frequency term of the cabling standards
  (it absorbs the DC-resistance and the mismatch contributions); small.

Foamed insulation
-----------------
With foaming degree phi (volume fraction of gas) the Lichtenecker
logarithmic mixing rule gives  eps_eff = eps_solid^(1 - phi)  and the loss
tangent scales with the solid fraction and the field concentration,
tan(delta_eff) = tan(delta_solid) (1 - phi) eps_solid^(-phi/2)  (empirical
form used here).

Geometry
--------
For a twisted pair of conductor diameter d_c and centre spacing s equal to
the insulation diameter D (the two insulated wires touch),
    Z_d = (120 / sqrt(eps_eff)) arccosh(D / d_c)          [ohm],
and  v = c0 / sqrt(eps_eff).

Temperature
-----------
rho(T) = rho_20 (1 + alpha_rho (T - 20)),  tan(delta)(T) = tan(delta)_20
(1 + beta_d (T - 20)),  eps(T) = eps_20 (1 + gamma_e (T - 20)), so that
    a(T) = a_20 sqrt(1 + alpha_rho (T - 20)),
    b(T) = b_20 (1 + beta_d (T - 20)) sqrt(eps(T)/eps_20).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

C0 = 299_792_458.0
MU0 = 4e-7 * np.pi

ALLOYS = {
    # resistivity at 20 C (ohm m), temperature coefficient (1/K)
    "Cu-ETP": (1.72e-8, 3.93e-3),
    "CuAg0.1": (1.76e-8, 3.85e-3),
    "CuSn0.3": (2.20e-8, 3.20e-3),
    "CuMg0.2": (2.45e-8, 2.90e-3),
}
INSULATIONS = {
    # eps_solid, tan delta solid at 20 C, beta_d (1/K), gamma_e (1/K)
    "PE": (2.30, 3.0e-4, 5.0e-3, -3.0e-4),
    "PP": (2.25, 4.0e-4, 6.0e-3, -3.5e-4),
    "FEP": (2.05, 5.0e-4, 2.0e-3, -1.5e-4),
}

__all__ = ["ALLOYS", "INSULATIONS", "Design", "hf_properties", "temperature_scaling", "insertion_loss_db_per_m", "eps_eff", "C0"]


@dataclass
class Design:
    alloy: str = "Cu-ETP"
    d_cond_mm: float = 0.50            # conductor diameter
    insulation: str = "PE"
    foaming: float = 0.30              # volume fraction of gas
    d_ins_mm: float = 0.92             # insulation outer diameter (centre spacing of the pair)
    k_prox: float = 1.15               # proximity / stranding factor on the ac resistance
    c_term: float = 0.02               # dB/m*sqrt(Hz) empirical low-frequency term


def eps_eff(insulation: str, foaming: float) -> tuple[float, float]:
    e_s, td_s, _, _ = INSULATIONS[insulation]
    e = e_s ** (1.0 - foaming)
    td = td_s * (1.0 - foaming) * e_s ** (-foaming / 2)
    return e, td


def hf_properties(d: Design, temperature_c: float = 20.0) -> dict:
    rho20, a_rho = ALLOYS[d.alloy]
    e_s, td_s, beta_d, gamma_e = INSULATIONS[d.insulation]
    rho = rho20 * (1 + a_rho * (temperature_c - 20))
    e20, td20 = eps_eff(d.insulation, d.foaming)
    e = e20 * (1 + gamma_e * (temperature_c - 20))
    td = td20 * (1 + beta_d * (temperature_c - 20))
    dc = d.d_cond_mm * 1e-3
    D = d.d_ins_mm * 1e-3
    z_d = 120.0 / np.sqrt(e) * np.arccosh(D / dc)
    a = 8.686 * d.k_prox * np.sqrt(MU0 * rho / np.pi) / (dc * z_d)
    b = 8.686 * np.pi * np.sqrt(e) * td / C0
    return {"a": float(a), "b": float(b), "c": float(d.c_term), "z_diff": float(z_d), "eps_eff": float(e),
            "tan_delta": float(td), "nvp": float(1 / np.sqrt(e)), "rho": float(rho)}


def temperature_scaling(d: Design, temperatures_c) -> dict:
    """a(T)/a20, b(T)/b20 and Z(T)/Z20 from the material coefficients."""
    base = hf_properties(d, 20.0)
    out = {"T": [], "a_ratio": [], "b_ratio": [], "z_ratio": []}
    for T in temperatures_c:
        p = hf_properties(d, T)
        out["T"].append(T)
        out["a_ratio"].append(p["a"] / base["a"])
        out["b_ratio"].append(p["b"] / base["b"])
        out["z_ratio"].append(p["z_diff"] / base["z_diff"])
    return out


def insertion_loss_db_per_m(f_hz: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    f = np.asarray(f_hz, float)
    return a * np.sqrt(f) + b * f + c / np.sqrt(np.maximum(f, 1.0))
