"""
Predict a cable's high-frequency performance from production data before it
is measured (the stretch goal).

    production row  ->  a, b, Z with 90 % intervals (P+ correlation model)
                    ->  a(T), b(T) with the material temperature coefficients
                    ->  IL(f, T) = L (c0 + a(T) sqrt f + b(T) f)
                    ->  margin against the limit line, probability of passing

The probability is a Monte-Carlo integral over the log-normal prediction
distributions of a and b (sigma from the 90 % conformal half-width /
1.645), with the correlation between a and b ignored (conservative for the
sum).  The limit line comes from cablecheck's limit library, so the same
file that judges a measured cable judges a predicted one.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .correlate import Correlation, predict_coefficients
from .physics import ALLOYS, INSULATIONS

__all__ = ["Prediction", "predict_performance"]


@dataclass
class Prediction:
    coefficients: dict
    temperature_c: float
    length_m: float
    f_hz: np.ndarray
    il_db: np.ndarray            # median prediction
    il_lo_db: np.ndarray
    il_hi_db: np.ndarray
    limit_db: np.ndarray | None
    margin_db: float | None
    f_worst_hz: float | None
    p_pass: float | None
    margin_hf_db: float | None = None      # worst margin above 100 MHz (where temperature matters)
    f_worst_hf_hz: float | None = None

    def to_dict(self) -> dict:
        return {"coefficients": self.coefficients, "temperature_c": self.temperature_c, "length_m": self.length_m,
                "margin_db": self.margin_db, "f_worst_mhz": None if self.f_worst_hz is None else self.f_worst_hz / 1e6,
                "margin_hf_db": self.margin_hf_db, "f_worst_hf_mhz": None if self.f_worst_hf_hz is None else self.f_worst_hf_hz / 1e6,
                "p_pass": self.p_pass}


def predict_performance(corr: Correlation, row: pd.Series | dict, length_m: float, temperature_c: float = 23.0,
                        f_hz: np.ndarray | None = None, cable_type: str | None = "1000base-t1-link-segment",
                        c0_db_per_m: float | None = None, n_mc: int = 4000, seed: int = 0) -> Prediction:
    row = dict(row)
    coef = predict_coefficients(corr, row)
    a_rho = ALLOYS[row["alloy"]][1]
    beta_d = INSULATIONS[row["insulation"]][2]
    dT = temperature_c - 20.0
    fa = np.sqrt(1 + a_rho * dT)
    fb = 1 + beta_d * dT
    f = np.linspace(1e6, 600e6, 600) if f_hz is None else np.asarray(f_hz, float)
    c0 = 0.0 if c0_db_per_m is None else c0_db_per_m
    a, b = coef["a"]["value"] * fa, coef["b"]["value"] * fb
    il = length_m * (c0 + a * np.sqrt(f) + b * f)
    # log-normal MC
    rng = np.random.default_rng(seed)
    sa = np.log(coef["a"]["hi90"] / coef["a"]["value"]) / 1.645
    sb = np.log(coef["b"]["hi90"] / coef["b"]["value"]) / 1.645
    A = a * np.exp(sa * rng.standard_normal(n_mc))
    B = b * np.exp(sb * rng.standard_normal(n_mc))
    IL = length_m * (c0 + A[:, None] * np.sqrt(f)[None, :] + B[:, None] * f[None, :])
    lo, hi = np.percentile(IL, 5, axis=0), np.percentile(IL, 95, axis=0)
    limit = margin = f_worst = p_pass = None
    margin_hf = f_worst_hf = None
    if cable_type:
        from cablecheck.limits.library import load_cable_type
        ct = load_cable_type(cable_type)
        lim, mask = ct.limit_for("insertion_loss").evaluate(f, length_m)
        limit = np.where(mask, lim, np.nan)
        mg = limit - il
        k = int(np.nanargmin(mg))
        margin, f_worst = float(mg[k]), float(f[k])
        ok = np.all((IL[:, mask] <= limit[mask][None, :]), axis=1)
        p_pass = float(np.mean(ok))
        hf = mask & (f >= 100e6)
        if np.any(hf):
            k2 = int(np.argmin(np.where(hf, mg, np.inf)))
            margin_hf, f_worst_hf = float(mg[k2]), float(f[k2])
    return Prediction(coef, temperature_c, length_m, f, il, lo, hi, limit, margin, f_worst, p_pass, margin_hf, f_worst_hf)
