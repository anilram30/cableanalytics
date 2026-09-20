"""
Synthetic production data set: how each sample was built, what the
extrusion-line gauge saw while it was made, and what the laboratory
measured afterwards.

Every relationship in the data set is physical (see :mod:`physics`) plus
the kinds of nuisance a real plant has: per-line stranding factors that
nobody has written down, lot-level dielectric contamination, gauge noise,
and laboratory measurement uncertainty.  The columns are grouped:

production   line, lot, alloy, d_cond_mm, insulation, foaming, d_ins_mm,
             line_speed_m_min, screw_rpm, capstan_d_m
gauge        gauge_mean_mm, gauge_std_mm, gauge_period_m, gauge_ripple_um
             (statistics of the simulated in-line diameter gauge log)
truth        a_true, b_true, c_true, z_true  (never available in practice)
measured     a_meas, b_meas, c_meas, sigma_a, sigma_b, z_meas,
             ripple_period_m, ripple_amp_ohm, il600_db_per_m, rl_min_db

The gauge log itself is  D(z) = D_nom + drift(z) + A_cap sin(2 pi z/p_cap)
+ A_scr sin(2 pi z/p_scr) + noise  with  p_cap = pi d_capstan  (one turn of
the capstan) and  p_scr = v_line / (rpm/60)  (one turn of the extruder
screw); a worn capstan bearing or screw shows up as a larger A_cap or A_scr
on that line.  The impedance ripple that the laboratory sees follows from
dZ/Z = g(D/d_c) dD/D with g = (D/d)/(sqrt((D/d)^2 - 1) arccosh(D/d)) ~ 1,
and its Bragg resonance in the return loss sits at f_B = v/(2 p).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .physics import Design, hf_properties

__all__ = ["Line", "LINES", "DESIGNS", "generate_dataset", "gauge_log", "gauge_statistics", "ripple_gain"]


@dataclass
class Line:
    line_id: str
    site: str
    capstan_d_m: float
    line_speed_m_min: float
    screw_rpm: float
    k_prox: float                 # stranding/proximity factor of this line's twinning process (unknown to the lab)
    capstan_ecc_um: float         # eccentricity of the capstan -> diameter ripple amplitude
    screw_pulsation_um: float     # extruder screw pulsation amplitude
    gauge_noise_um: float = 1.5


LINES = {
    "L1": Line("L1", "Roth", 0.400, 60.0, 42.0, 1.12, 1.0, 0.8),
    "L2": Line("L2", "Roth", 0.315, 80.0, 55.0, 1.18, 6.0, 1.0),      # worn capstan bearing
    "L3": Line("L3", "Nitra", 0.500, 50.0, 38.0, 1.15, 1.2, 5.5),     # worn screw
}

DESIGNS = {
    "D100-PE-035": Design("Cu-ETP", 0.50, "PE", 0.30, 0.845),
    "D100-PP-022": Design("CuSn0.3", 0.40, "PP", 0.35, 0.659),
    "D100-FEP-035": Design("CuAg0.1", 0.50, "FEP", 0.20, 0.843),
    "D100-PE-050": Design("Cu-ETP", 0.60, "PE", 0.40, 0.976),
}


def ripple_gain(d_over_dc: float) -> float:
    r = d_over_dc
    return r / (np.sqrt(r * r - 1) * np.arccosh(r))


def gauge_log(line: Line, d_nom_mm: float, length_m: float = 500.0, dz: float = 0.01, rng=None, drift_um: float = 3.0):
    """Simulated in-line diameter gauge: (z [m], D [mm])."""
    rng = rng or np.random.default_rng(0)
    z = np.arange(0, length_m, dz)
    p_cap = np.pi * line.capstan_d_m
    p_scr = line.line_speed_m_min / line.screw_rpm  # metres per screw revolution (v/60 / (rpm/60))
    drift = drift_um * np.sin(2 * np.pi * z / (0.7 * length_m) + rng.uniform(0, 2 * np.pi))
    d = (d_nom_mm * 1e3 + drift + line.capstan_ecc_um * np.sin(2 * np.pi * z / p_cap + rng.uniform(0, 2 * np.pi))
         + line.screw_pulsation_um * np.sin(2 * np.pi * z / p_scr + rng.uniform(0, 2 * np.pi))
         + line.gauge_noise_um * rng.standard_normal(z.size))
    return z, d / 1e3


def gauge_statistics(z: np.ndarray, d_mm: np.ndarray) -> dict:
    """Mean, std, dominant spatial period and its amplitude (Hann periodogram)."""
    dz = float(z[1] - z[0])
    x = d_mm * 1e3
    x = x - np.polyval(np.polyfit(z, x, 2), z)
    w = np.hanning(x.size)
    sp = np.abs(np.fft.rfft(x * w))
    fr = np.fft.rfftfreq(x.size, dz)
    valid = (fr > 1 / (z[-1] - z[0]) * 4) & (fr < 1 / (4 * dz))
    i = int(np.argmax(np.where(valid, sp, 0)))
    return {"gauge_mean_mm": float(np.mean(d_mm)), "gauge_std_mm": float(np.std(d_mm)),
            "gauge_period_m": float(1 / fr[i]), "gauge_ripple_um": float(2 * sp[i] / np.sum(w))}


def generate_dataset(n: int = 240, seed: int = 7, gauge_dir: str | Path | None = None, n_gauge_files: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    line_ids = list(LINES)
    design_ids = list(DESIGNS)
    for i in range(n):
        line = LINES[line_ids[i % 3]]
        dsg = DESIGNS[design_ids[(i // 3) % 4]]
        lot = f"LOT{2026000 + i // 12:07d}"
        # production variation
        d_cond = dsg.d_cond_mm * (1 + rng.normal(0, 0.008))
        foaming = float(np.clip(dsg.foaming + rng.normal(0, 0.03), 0.05, 0.6))
        d_ins = dsg.d_ins_mm * (1 + rng.normal(0, 0.012))
        speed = line.line_speed_m_min * (1 + rng.normal(0, 0.05))
        rpm = line.screw_rpm * (1 + rng.normal(0, 0.03))
        k_prox = line.k_prox * (1 + rng.normal(0, 0.02))
        lot_td = 1 + 0.15 * (rng.random() < 0.15) * rng.random()       # occasional dielectric contamination
        design = Design(dsg.alloy, d_cond, dsg.insulation, foaming, d_ins, k_prox, 0.02 * (1 + rng.normal(0, 0.2)))
        p = hf_properties(design, 23.0)
        b_true = p["b"] * lot_td
        line_i = Line(line.line_id, line.site, line.capstan_d_m, speed, rpm, k_prox, line.capstan_ecc_um * (1 + rng.normal(0, 0.15)),
                      line.screw_pulsation_um * (1 + rng.normal(0, 0.15)))
        z, d = gauge_log(line_i, d_ins, rng=rng)
        gs = gauge_statistics(z, d)
        if gauge_dir is not None and i < n_gauge_files:
            Path(gauge_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"z_m": z, "d_mm": d}).to_csv(Path(gauge_dir) / f"gauge_S{i:04d}.csv", index=False)
        # ripple seen by the lab: dominant gauge component -> impedance ripple
        g = ripple_gain(d_ins / d_cond)
        ripple_amp = p["z_diff"] * g * gs["gauge_ripple_um"] * 1e-3 / d_ins
        ripple_period = gs["gauge_period_m"] * (1 + rng.normal(0, 0.02))
        # laboratory measurement of the coefficients (fit uncertainty)
        sa, sb = p["a"] * 0.012, p["b"] * 0.03 + 1e-13
        a_meas = p["a"] * (1 + rng.normal(0, 0.012))
        b_meas = b_true * (1 + rng.normal(0, 0.03)) + rng.normal(0, 1e-13)
        c_meas = p["c"] + rng.normal(0, 0.004)
        z_meas = p["z_diff"] * (1 + rng.normal(0, 0.004))
        f600 = 600e6
        il600 = a_meas * np.sqrt(f600) + b_meas * f600 + c_meas / np.sqrt(f600)
        # return-loss minimum from the Bragg resonance of the ripple (15 m sample), plus a floor
        n_per = 15.0 / ripple_period
        gamma_b = min(0.5 * (ripple_amp / p["z_diff"]) * n_per, 0.8)
        rl_min = -20 * np.log10(max(gamma_b, 10 ** (-32 / 20)))
        rows.append({"sample_id": f"S{i:04d}", "line": line.line_id, "site": line.site, "lot": lot, "design": design_ids[(i // 3) % 4],
                     "alloy": dsg.alloy, "d_cond_mm": d_cond, "insulation": dsg.insulation, "foaming": foaming,
                     "d_ins_mm": d_ins, "line_speed_m_min": speed, "screw_rpm": rpm, "capstan_d_m": line.capstan_d_m,
                     **gs, "a_true": p["a"], "b_true": b_true, "c_true": p["c"], "z_true": p["z_diff"], "k_prox_true": k_prox,
                     "lot_td_factor": lot_td, "a_meas": a_meas, "b_meas": b_meas, "c_meas": c_meas, "sigma_a": sa, "sigma_b": sb,
                     "z_meas": z_meas, "ripple_period_m": ripple_period, "ripple_amp_ohm": ripple_amp,
                     "il600_db_per_m": il600, "rl_min_db": rl_min})
    return pd.DataFrame(rows)
