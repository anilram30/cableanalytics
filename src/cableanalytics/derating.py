"""
Temperature derating from climate-chamber sweeps.

Given loss fits (a_i, b_i, c_i) at chamber temperatures T_i, the material
model of :mod:`physics` is fitted:

    a(T) = a_20 sqrt(1 + alpha_rho (T - 20))      -> alpha_rho (conductor)
    b(T) = b_20 (1 + beta_d (T - 20))             -> beta_d  (dielectric)

by weighted least squares on a^2 (linear in T) and b (linear in T), with
each point weighted by its fit uncertainty.  The derating curve at a
temperature T is

    D(f, T) = IL(f, T) / IL(f, 23 C)

and the *hot margin* against a limit line IL_lim(f) (from cablecheck's
limit library) is  m(T) = min_f [IL_lim(f) - L IL(f, T)],  which also
gives the maximum cable length that still meets the limit at T:
    L_max(T) = min_f IL_lim(f) / IL(f, T) per metre.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lossfit import LossFit, loss_model

__all__ = ["DeratingModel", "fit_derating"]


@dataclass
class DeratingModel:
    a20: float
    b20: float
    c: float
    alpha_rho: float
    beta_d: float
    sigma_alpha: float
    sigma_beta: float
    temperatures: list[float]
    fits: list[LossFit]
    basis: str = "physical"

    def coefficients(self, T: float) -> tuple[float, float, float]:
        return (self.a20 * np.sqrt(max(1 + self.alpha_rho * (T - 20), 1e-6)),
                self.b20 * (1 + self.beta_d * (T - 20)), self.c)

    def il_per_m(self, f_hz, T: float) -> np.ndarray:
        a, b, c = self.coefficients(T)
        return loss_model(f_hz, a, b, c, self.basis)

    def derating(self, f_hz, T: float, T_ref: float = 23.0) -> np.ndarray:
        return self.il_per_m(f_hz, T) / self.il_per_m(f_hz, T_ref)

    def hot_margin(self, f_hz, limit_db, length_m: float, T: float) -> tuple[float, float]:
        """(worst margin in dB at T, frequency of the worst margin)."""
        il = self.il_per_m(f_hz, T) * length_m
        mg = np.asarray(limit_db) - il
        k = int(np.argmin(mg))
        return float(mg[k]), float(np.asarray(f_hz)[k])

    def max_length(self, f_hz, limit_db, T: float) -> float:
        il = self.il_per_m(f_hz, T)
        return float(np.min(np.asarray(limit_db) / il))

    def to_dict(self) -> dict:
        return {"a20": self.a20, "b20": self.b20, "c": self.c, "alpha_rho_per_K": self.alpha_rho,
                "beta_d_per_K": self.beta_d, "sigma_alpha": self.sigma_alpha, "sigma_beta": self.sigma_beta,
                "temperatures_c": self.temperatures}


def fit_derating(temperatures_c, fits: list[LossFit]) -> DeratingModel:
    T = np.asarray(temperatures_c, float)
    a = np.array([f.a for f in fits])
    b = np.array([f.b for f in fits])
    sa = np.array([max(f.sigma[0], 1e-12) for f in fits])
    sb = np.array([max(f.sigma[1], 1e-12) for f in fits])
    # a^2 = a20^2 (1 + alpha (T-20)):  linear in T with weights from sigma(a^2) = 2 a sigma_a
    X = np.column_stack([np.ones_like(T), T - 20])
    wa = 1 / (2 * a * sa) ** 2
    beta_a, cov_a = _wls(X, a ** 2, wa)
    a20 = float(np.sqrt(max(beta_a[0], 1e-30)))
    alpha = float(beta_a[1] / beta_a[0])
    sig_alpha = float(np.sqrt(cov_a[1, 1]) / beta_a[0])
    wb = 1 / sb ** 2
    beta_b, cov_b = _wls(X, b, wb)
    b20 = float(beta_b[0])
    beta_d = float(beta_b[1] / beta_b[0]) if beta_b[0] > 0 else 0.0
    sig_beta = float(np.sqrt(cov_b[1, 1]) / beta_b[0]) if beta_b[0] > 0 else float("nan")
    c = float(np.average([f.c for f in fits]))
    return DeratingModel(a20, b20, c, alpha, beta_d, sig_alpha, sig_beta, list(map(float, T)), list(fits), fits[0].basis)


def _wls(X, y, w):
    Xw = X * np.sqrt(w)[:, None]
    beta, *_ = np.linalg.lstsq(Xw, y * np.sqrt(w), rcond=None)
    r = y - X @ beta
    dof = max(y.size - X.shape[1], 1)
    s2 = float(np.sum(w * r ** 2) / dof) if y.size > X.shape[1] else 1.0
    cov = s2 * np.linalg.inv(X.T @ (X * w[:, None]))
    return beta, cov
