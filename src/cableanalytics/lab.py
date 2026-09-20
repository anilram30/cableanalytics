"""
The laboratory path: Touchstone files (from cablecheck's multiconductor
model, driven by this package's design physics) -> insertion loss ->
three-term fit -> derating, and return loss -> periodic signature ->
machine attribution.

This is the end-to-end chain a production sample follows: the design and
line parameters set the cable physics, the climate chamber sets the
temperature, the VNA produces S-parameters, cablecheck extracts IL/RL, and
this package turns them into three numbers, a derating model and a machine.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from cablecheck.io import read_touchstone, write_touchstone
from cablecheck.mixedmode import PortMap, to_mixed_mode
from cablecheck.network import db
from cablecheck.synth import RHO_CU, PairSpec, make_pair

from .derating import DeratingModel, fit_derating
from .lossfit import LossFit, fit_loss
from .periodic import Attribution, attribute, period_from_return_loss
from .physics import C0, Design, hf_properties

__all__ = ["pair_spec_from_design", "measure", "temperature_sweep", "LabResult", "evaluate_touchstone"]


def pair_spec_from_design(d: Design, temperature_c: float, length_m: float, ripple_period_m: float | None = None,
                          ripple_rel: float = 0.0, seed: int = 0, n_segments: int = 120) -> PairSpec:
    """Map a design at a temperature onto cablecheck's PairSpec.

    cablecheck's conductor model has copper resistivity built in; alloy and
    temperature enter through the proximity factor (R_s ~ sqrt(rho)) and the
    DC resistance (~ rho)."""
    p = hf_properties(d, temperature_c)
    rho_ratio = p["rho"] / RHO_CU
    dc = d.d_cond_mm * 1e-3
    r_dc = 4 * p["rho"] / (np.pi * dc ** 2)          # per conductor
    return PairSpec(length_m=length_m, z_diff=p["z_diff"], z_comm=0.35 * p["z_diff"], nvp=p["nvp"], nvp_even=p["nvp"] * 0.97,
                    d_wire_m=dc, r_dc_ohm_per_m=r_dc, tan_delta=p["tan_delta"], proximity=d.k_prox * np.sqrt(rho_ratio),
                    ripple_amp=2 * ripple_rel, ripple_period_m=ripple_period_m or 0.3, roughness=0.003, seed=seed,
                    n_segments=n_segments)


def measure(spec: PairSpec, f: np.ndarray, path: Path, name: str, noise_db: float = -85.0, seed: int = 1) -> Path:
    from cablecheck.synth import synthesize_measurement
    net = synthesize_measurement(make_pair(spec, f, name=name), None, noise_db=noise_db, seed=seed, name=name)
    return write_touchstone(net, path, comments=[f"synthetic lab measurement {name}"])


def il_rl_from_touchstone(path: Path):
    net = read_touchstone(path)
    mm = to_mixed_mode(net, PortMap.single_pair())
    il = -db(mm.param(("d", "A", "far"), ("d", "A", "near")))
    rl = -db(mm.param(("d", "A", "near"), ("d", "A", "near")))
    s21 = mm.param(("d", "A", "far"), ("d", "A", "near"))
    return net.f, il, rl, s21


@dataclass
class LabResult:
    temperatures: list[float]
    fits: list[LossFit]
    derating: DeratingModel
    period: dict | None
    attribution: Attribution | None
    files: list[str]


def temperature_sweep(d: Design, length_m: float, out_dir: Path, temperatures=(-40, 23, 85, 105, 125),
                      n_points: int = 1201, fmax: float = 1e9, line: dict | None = None, seed: int = 0) -> LabResult:
    """Climate-chamber sweep: one Touchstone file per temperature, fits, derating model;
    the 23 C file also yields the periodic signature and (with the line data) its attribution."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = np.linspace(1e6, fmax, n_points)
    fits, files = [], []
    period = attr = None
    rp = (line or {}).get("ripple_period_m")
    rr = (line or {}).get("ripple_rel", 0.0)
    for k, T in enumerate(temperatures):
        spec = pair_spec_from_design(d, T, length_m, rp, rr, seed=seed)
        p = measure(spec, f, out_dir / f"sample_T{int(T):+04d}C.s4p", f"T={T}C", seed=seed + k)
        files.append(str(p))
        ff, il, rl, s21 = il_rl_from_touchstone(p)
        fits.append(fit_loss(ff, il, length_m, fmin=5e6))
        if T == 23 and line:
            v = hf_properties(d, 23.0)["nvp"] * C0
            period = period_from_return_loss(ff, rl, v)
            if period:
                attr = attribute(period["period_m"], line["line_speed_m_min"], line["screw_rpm"], line["capstan_d_m"])
    der = fit_derating(temperatures, fits)
    return LabResult(list(map(float, temperatures)), fits, der, period, attr, files)


def evaluate_touchstone(path: str | Path, length_m: float, line: dict | None = None) -> dict:
    """One real measurement file -> fit, period, attribution (what the CLI runs for a lab sample)."""
    ff, il, rl, s21 = il_rl_from_touchstone(Path(path))
    fit = fit_loss(ff, il, length_m, fmin=5e6)
    ph = np.unwrap(np.angle(s21))
    n = max(5, ff.size // 20)
    v = float(-2 * np.pi * (ff[-1] - ff[-n]) / (ph[-1] - ph[-n]) * length_m)
    period = period_from_return_loss(ff, rl, v)
    attr = attribute(period["period_m"], line["line_speed_m_min"], line["screw_rpm"], line["capstan_d_m"]) if (period and line) else None
    return {"fit": fit.to_dict(), "velocity": v, "period": period, "attribution": attr.to_dict() if attr else None}
