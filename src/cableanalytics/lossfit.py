"""
Three-term loss fit with robust weighting and parameter uncertainty.

Two bases:

* ``physical``  IL(f) = L (c0 + a sqrt(f) + b f)          [dB], f in Hz, L in m
  c0 = 8.686 R'_dc / (2 Z) is the DC-resistance term, a sqrt f the skin
  effect, b f the dielectric loss - the exact shape of a lossy line with
  R = R_dc + R_s sqrt f and G = omega C tan(delta).
* ``standard``  IL(f) = L (a sqrt(f) + b f + c / sqrt(f))
  the empirical shape of the cabling standards (IEC 61156, IEEE 802.3
  link segments), whose 1/sqrt f term cannot represent a constant.

The choice matters for *interpretation*, not for describing the curve:
on a foamed-PE automotive pair the standard basis fits the curve to 0.02 dB
but returns a dielectric coefficient b that is 98 % too small and a
conductor coefficient a that is 6 % too large, because the DC term is
absorbed by the other two; the physical basis recovers both to 0.3 %.
Use ``standard`` for limit-line work and ``physical`` (default) whenever
a and b are to be related to materials or temperature.

The fit is linear in the coefficients, so it is a weighted least-squares
problem.  Two refinements matter on real cable data:

* **Robustness.**  Standing-wave ripple and the Bragg resonance of a
  periodic impedance defect put narrow spikes on IL(f) that are not loss.
  The fit is iteratively re-weighted with Huber weights
  w_i = min(1, k s / |r_i|)  (k = 1.5, s = MAD-based residual scale), which
  down-weights the spikes without discarding them by hand.
* **Uncertainty.**  The parameter covariance is
  cov = s^2 (X^T W X)^{-1}  with s^2 the weighted residual variance; the
  standard errors sigma_a, sigma_b, sigma_c and the correlation between a
  and b (which is strong, because sqrt(f) and f are similar over one
  decade) are reported, so that a production correlation can weight each
  sample by its own measurement uncertainty.

The physical interpretation follows from Project E's physics module:
a -> sqrt(rho) k_prox / (d_c Z_d),  b -> sqrt(eps) tan(delta).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["LossFit", "fit_loss", "loss_model"]


@dataclass
class LossFit:
    a: float                  # dB/m/sqrt(Hz)
    b: float                  # dB/m/Hz
    c: float                  # physical: c0 in dB/m (DC term);  standard: c in dB*sqrt(Hz)/m
    sigma: tuple[float, float, float]
    basis: str
    corr_ab: float
    rms_residual_db: float
    r2: float
    n_used: int
    n_total: int
    length_m: float
    f_range: tuple[float, float]

    def il_db(self, f_hz, length_m: float | None = None) -> np.ndarray:
        return loss_model(f_hz, self.a, self.b, self.c, self.basis) * (self.length_m if length_m is None else length_m)

    def to_dict(self) -> dict:
        return {"basis": self.basis, "a": self.a, "b": self.b, "c": self.c, "sigma_a": self.sigma[0], "sigma_b": self.sigma[1],
                "sigma_c": self.sigma[2], "corr_ab": self.corr_ab, "rms_residual_db": self.rms_residual_db,
                "r2": self.r2, "n_used": self.n_used, "n_total": self.n_total, "length_m": self.length_m}


def loss_model(f_hz, a: float, b: float, c: float, basis: str = "physical") -> np.ndarray:
    f = np.asarray(f_hz, float)
    if basis == "physical":
        return c + a * np.sqrt(f) + b * f
    return a * np.sqrt(f) + b * f + c / np.sqrt(np.maximum(f, 1.0))


def fit_loss(f_hz: np.ndarray, il_db: np.ndarray, length_m: float, fmin: float | None = None,
             fmax: float | None = None, min_il_db: float = 0.05, huber_k: float = 1.5, n_iter: int = 8,
             fit_c: bool = True, basis: str = "physical") -> LossFit:
    f = np.asarray(f_hz, float)
    y = np.asarray(il_db, float)
    m = np.isfinite(y) & (y > min_il_db)
    if fmin is not None:
        m &= f >= fmin
    if fmax is not None:
        m &= f <= fmax
    fm, ym = f[m], y[m]
    if basis == "physical":
        cols = [np.sqrt(fm), fm] + ([np.ones_like(fm)] if fit_c else [])
    elif basis == "standard":
        cols = [np.sqrt(fm), fm] + ([1 / np.sqrt(fm)] if fit_c else [])
    else:
        raise ValueError("basis must be 'physical' or 'standard'")
    X = length_m * np.column_stack(cols)
    w = np.ones(fm.size)
    beta = np.zeros(X.shape[1])
    for _ in range(n_iter):
        Xw = X * np.sqrt(w)[:, None]
        beta, *_ = np.linalg.lstsq(Xw, ym * np.sqrt(w), rcond=None)
        r = ym - X @ beta
        s = 1.4826 * np.median(np.abs(r - np.median(r))) + 1e-9
        w_new = np.minimum(1.0, huber_k * s / np.maximum(np.abs(r), 1e-12))
        if np.allclose(w_new, w, atol=1e-4):
            w = w_new
            break
        w = w_new
    r = ym - X @ beta
    s2 = float(np.sum(w * r ** 2) / max(np.sum(w) - X.shape[1], 1))
    cov = s2 * np.linalg.inv(X.T @ (X * w[:, None]))
    sig = np.sqrt(np.diag(cov))
    corr_ab = float(cov[0, 1] / (sig[0] * sig[1]))
    ss_res = float(np.sum(w * r ** 2))
    ss_tot = float(np.sum(w * (ym - np.average(ym, weights=w)) ** 2))
    a, b = float(beta[0]), float(beta[1])
    c = float(beta[2]) if fit_c else 0.0
    sigs = (float(sig[0]), float(sig[1]), float(sig[2]) if fit_c else 0.0)
    return LossFit(a, b, c, sigs, basis, corr_ab, float(np.sqrt(np.mean(r ** 2))), 1 - ss_res / ss_tot,
                   int(np.sum(w > 0.5)), int(f.size), length_m, (float(fm.min()), float(fm.max())))
