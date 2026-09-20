"""
Periodic reflection signatures and their attribution to a machine.

A periodic impedance variation of spatial period p produces a Bragg
resonance in the return loss at

    f_B = v / (2 p)        (and harmonics n f_B),

and a peak at spatial frequency 1/p in the periodogram of the impedance
profile.  Both are detected here; the period estimates are combined and
then matched against the periods that the production line can imprint:

    capstan   p = pi d_capstan            (one revolution)
    screw     p = v_line / (rpm / 60)     (one revolution of the extruder screw)
    take-up   p = pi d_reel               (varies with the fill; given as a range)

including harmonics p/n, n = 1..3.  The attribution score is the relative
period mismatch; the best candidate below ``tol`` is reported together
with the runner-up so that an ambiguous case is visible as such.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

__all__ = ["period_from_return_loss", "period_from_profile", "Attribution", "attribute"]


def period_from_return_loss(f_hz: np.ndarray, rl_db: np.ndarray, velocity: float, min_depth_db: float = 6.0,
                            fmin: float = 20e6) -> dict | None:
    """Dominant Bragg resonance -> period.

    The return loss of a cable with a periodic impedance variation has a *deep*
    dip at f_B (and at its harmonics) that stands out from the ordinary
    standing-wave ripple of the two ends.  The dip must be at least
    ``min_depth_db`` below the 10th percentile of the return loss to count;
    among the qualifying dips the deepest is taken, unless a dip near half its
    frequency is nearly as deep (then that is the fundamental).
    Returns dict(period_m, f_bragg_hz, depth_db, harmonic_hz) or None."""
    f = np.asarray(f_hz, float)
    rl = np.asarray(rl_db, float)
    m = f >= fmin
    fm, rm = f[m], rl[m]
    floor = np.percentile(rm, 10)
    pk, props = find_peaks(-rm, prominence=3.0)
    deep = [j for j in pk if rm[j] <= floor - min_depth_db]
    if not deep:
        return None
    deep.sort(key=lambda j: rm[j])
    j0 = deep[0]
    f_b = float(fm[j0])
    # fundamental check: a dip near f_b/2 that is within 6 dB of the deepest
    for j in deep:
        if abs(fm[j] / f_b - 0.5) < 0.06 and rm[j] <= rm[j0] + 6:
            f_b = float(fm[j])
            j0 = j
            break
    harm = [float(fm[j]) for j in deep if any(abs(fm[j] / f_b - n) < 0.05 for n in (2, 3, 4))]
    return {"period_m": velocity / (2 * f_b), "f_bragg_hz": f_b, "depth_db": float(floor - rm[j0]),
            "rl_at_bragg_db": float(rm[j0]), "harmonic_hz": harm}


def period_from_profile(x_m: np.ndarray, z_ohm: np.ndarray, resolution_m: float) -> dict | None:
    x = np.asarray(x_m, float)
    z = np.asarray(z_ohm, float)
    dx = float(np.median(np.diff(x)))
    xu = np.arange(x[0], x[-1], dx)
    zu = np.interp(xu, x, z)
    zu = zu - np.polyval(np.polyfit(xu, zu, 1), xu)
    if zu.size < 16:
        return None
    w = np.hanning(zu.size)
    sp = np.abs(np.fft.rfft(zu * w))
    fr = np.fft.rfftfreq(zu.size, dx)
    valid = (fr > 2 / (xu[-1] - xu[0])) & (fr < 1 / (2 * resolution_m))
    if not np.any(valid):
        return None
    i = int(np.argmax(np.where(valid, sp, 0)))
    return {"period_m": float(1 / fr[i]), "amplitude_ohm": float(2 * sp[i] / np.sum(w))}


@dataclass
class Attribution:
    period_m: float
    candidates: list[dict]          # sorted by score
    best: dict | None
    runner_up: dict | None
    ambiguous: bool

    def to_dict(self) -> dict:
        return {"period_m": self.period_m, "best": self.best, "runner_up": self.runner_up, "ambiguous": self.ambiguous,
                "candidates": self.candidates}


def attribute(period_m: float, line_speed_m_min: float, screw_rpm: float, capstan_d_m: float,
              reel_d_range_m: tuple[float, float] = (0.4, 1.2), tol: float = 0.06, max_harmonic: int = 3) -> Attribution:
    cands = []
    base = {"capstan": np.pi * capstan_d_m, "screw": line_speed_m_min / screw_rpm}
    for name, p0 in base.items():
        for n in range(1, max_harmonic + 1):
            p = p0 / n
            cands.append({"element": name, "harmonic": n, "expected_period_m": p, "score": abs(period_m / p - 1)})
    # take-up reel: its circumference grows as the reel fills, so it cannot imprint a
    # *stationary* period over a whole sample; it is kept as a weak fallback (base score
    # above tol) that only wins when nothing else matches and the period is inside its range
    lo, hi = np.pi * reel_d_range_m[0], np.pi * reel_d_range_m[1]
    for n in range(1, max_harmonic + 1):
        plo, phi = lo / n, hi / n
        s = 0.0 if plo <= period_m <= phi else min(abs(period_m / plo - 1), abs(period_m / phi - 1))
        cands.append({"element": "take-up reel (non-stationary)", "harmonic": n, "expected_period_m": (plo + phi) / 2,
                      "score": s + 1.5 * tol})
    cands.sort(key=lambda c: c["score"])
    best = cands[0] if cands[0]["score"] <= tol else None
    runner = cands[1] if len(cands) > 1 else None
    ambiguous = best is not None and runner is not None and runner["score"] <= tol
    return Attribution(period_m, cands, best, runner, ambiguous)
