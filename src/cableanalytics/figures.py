"""Figures for the report and the ``demo`` command (the whole chain on synthetic data)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C1, C2, C3, C4, CK, CG = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#0b0b0b", "#9a9a96"


def _style(ax):
    ax.grid(True, color="#e6e6e3", lw=0.7)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def correlation_figures(df: pd.DataFrame, corr, out: Path):
    from .correlate import physics_prediction
    d = physics_prediction(df)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, (tgt, yc, pc, lab) in zip(axes, [("a", "a_meas", "a_phys", "a  [dB/m/√Hz]"), ("b", "b_meas", "b_phys", "b  [dB/m/Hz]"), ("z", "z_meas", "z_phys", "Z  [Ω]")]):
        for line, col in zip(["L1", "L2", "L3"], [C1, C2, C3]):
            m = d["line"] == line
            ax.scatter(d.loc[m, pc], d.loc[m, yc], s=12, color=col, label=line, alpha=0.8)
        lo, hi = d[pc].min(), d[pc].max()
        ax.plot([lo, hi], [lo, hi], color=CK, lw=1)
        ax.set_xlabel(f"physics prediction {lab}")
        ax.set_ylabel(f"measured {lab}")
        ax.set_title(f"{tgt}: physics vs measured", loc="left", fontsize=10)
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out / "correlation_physics_vs_measured.png", dpi=130)
    plt.close(fig)
    # sensitivities
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, tgt in zip(axes, ["a", "b"]):
        s = corr.sensitivities[tgt]
        names = list(s)
        vals = np.array([s[n] for n in names])
        y = np.arange(len(names))
        ax.barh(y, vals, color=[C1 if v >= 0 else C2 for v in vals], height=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.axvline(0, color=CK, lw=1)
        ax.set_title(f"standardised sensitivity of log {tgt}", loc="left", fontsize=10)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out / "sensitivities.png", dpi=130)
    plt.close(fig)
    # ripple / attribution per line
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for line, col in zip(["L1", "L2", "L3"], [C1, C2, C3]):
        m = df["line"] == line
        ax.scatter(df.loc[m, "ripple_period_m"], df.loc[m, "ripple_amp_ohm"], s=14, color=col, label=line)
    from .production import LINES
    for line, col in zip(["L1", "L2", "L3"], [C1, C2, C3]):
        L = LINES[line]
        ax.axvline(np.pi * L.capstan_d_m, color=col, ls=":", lw=1)
        ax.axvline(L.line_speed_m_min / L.screw_rpm, color=col, ls="--", lw=1)
    ax.set_xlabel("dominant ripple period seen by the lab / m   (dotted: capstan circumference, dashed: screw revolution)")
    ax.set_ylabel("ripple amplitude / Ω")
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "ripple_attribution.png", dpi=130)
    plt.close(fig)


def run_demo(out: Path, data: Path):
    out.mkdir(parents=True, exist_ok=True)
    from .correlate import fit_correlation
    from .lab import il_rl_from_touchstone, temperature_sweep
    from .lossfit import fit_loss
    from .physics import ALLOYS, INSULATIONS, hf_properties
    from .predict import predict_performance
    from .production import DESIGNS, LINES, generate_dataset
    # 1. data set
    csv = data / "production_dataset.csv"
    df = pd.read_csv(csv) if csv.exists() else generate_dataset(240, gauge_dir=data / "gauge")
    if not csv.exists():
        data.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv, index=False)
    # 2. lab path: temperature sweep on line L2 (worn capstan)
    d = DESIGNS["D100-PE-035"]
    line = LINES["L2"]
    res = temperature_sweep(d, 15.0, data / "lab" / "D100-PE-035_L2",
                            line={"line_speed_m_min": line.line_speed_m_min, "screw_rpm": line.screw_rpm,
                                  "capstan_d_m": line.capstan_d_m, "ripple_period_m": np.pi * line.capstan_d_m, "ripple_rel": 0.006})
    summary = {"temperatures": res.temperatures, "fits": [ft.to_dict() for ft in res.fits], "derating": res.derating.to_dict(),
               "material": {"alpha_rho": ALLOYS[d.alloy][1], "beta_d": INSULATIONS[d.insulation][2]},
               "truth": {str(T): hf_properties(d, T) for T in res.temperatures},
               "period": res.period, "attribution": res.attribution.to_dict() if res.attribution else None}
    # basis comparison on the 23 C file
    f, il, rl, s21 = il_rl_from_touchstone(Path(res.files[1]))
    summary["basis_comparison"] = {b: fit_loss(f, il, 15.0, fmin=5e6, basis=b).to_dict() for b in ("physical", "standard")}
    # figure: fits at temperatures + derating
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    ax = axes[0]
    cols = [C1, C3, C4, C2, CK]
    for T, ft, col in zip(res.temperatures, res.fits, cols):
        ff, ill, _, _ = il_rl_from_touchstone(Path(res.files[res.temperatures.index(T)]))
        ax.plot(ff / 1e6, ill, color=col, lw=0.8, alpha=0.6)
        ax.plot(ff / 1e6, ft.il_db(ff), color=col, lw=1.4, ls="--", label=f"{T:.0f} °C fit")
    ax.set_xlabel("frequency / MHz"), ax.set_ylabel("IL / dB (15 m)")
    ax.set_title("measured IL and three-term fits", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[1]
    T = np.linspace(-40, 130, 100)
    der = res.derating
    ax.plot(T, [der.coefficients(t)[0] / der.coefficients(23)[0] for t in T], color=C1, label="a(T)/a(23) fitted")
    ax.plot(T, [der.coefficients(t)[1] / der.coefficients(23)[1] for t in T], color=C2, label="b(T)/b(23) fitted")
    ax.plot(res.temperatures, [ft.a / res.fits[1].a for ft in res.fits], "o", color=C1)
    ax.plot(res.temperatures, [ft.b / res.fits[1].b for ft in res.fits], "s", color=C2)
    tr = [hf_properties(d, t) for t in T]
    ax.plot(T, [p["a"] / hf_properties(d, 23)["a"] for p in tr], color=C1, ls=":", label="material model a")
    ax.plot(T, [p["b"] / hf_properties(d, 23)["b"] for p in tr], color=C2, ls=":", label="material model b")
    ax.set_xlabel("temperature / °C"), ax.set_ylabel("ratio to 23 °C")
    ax.set_title("derating of the coefficients", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    ax = axes[2]
    ff = np.linspace(1e6, 600e6, 500)
    for t, col in zip([-40, 23, 85, 105, 125], cols):
        ax.plot(ff / 1e6, der.derating(ff, t), color=col, label=f"{t} °C")
    ax.set_xlabel("frequency / MHz"), ax.set_ylabel("IL(f,T) / IL(f,23 °C)")
    ax.set_title("derating curves", loc="left", fontsize=10)
    ax.legend(fontsize=7, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "derating.png", dpi=130)
    plt.close(fig)
    # figure: return loss with Bragg dip
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(f / 1e6, rl, color=C1, lw=0.9)
    if res.period:
        ax.axvline(res.period["f_bragg_hz"] / 1e6, color=C2, ls="--", lw=1)
        ax.text(res.period["f_bragg_hz"] / 1e6, rl.max(), f" Bragg {res.period['f_bragg_hz'] / 1e6:.1f} MHz → p = {res.period['period_m']:.3f} m"
                f" → {res.attribution.best['element'] if res.attribution and res.attribution.best else '?'}", fontsize=8, va="top")
    ax.invert_yaxis()
    ax.set_xlabel("frequency / MHz"), ax.set_ylabel("return loss / dB")
    ax.set_title("return loss of the L2 sample: the capstan's Bragg resonance", loc="left", fontsize=10)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "bragg.png", dpi=130)
    plt.close(fig)
    # 3. correlation + prediction
    corr = fit_correlation(df)
    (out / "correlation_table.md").write_text(corr.table())
    correlation_figures(df, corr, out)
    preds = []
    for sid in df["sample_id"].iloc[:12]:
        row = df[df["sample_id"] == sid].iloc[0]
        for T in (23.0, 105.0):
            pr = predict_performance(corr, row, 15.0, T)
            preds.append({"sample": sid, "line": row["line"], "design": row["design"], "T": T, "margin_db": pr.margin_db,
                          "margin_hf_db": pr.margin_hf_db, "p_pass": pr.p_pass, "a_pred": pr.coefficients["a"]["value"], "a_meas": float(row["a_meas"]),
                          "b_pred": pr.coefficients["b"]["value"], "b_meas": float(row["b_meas"])})
    pd.DataFrame(preds).to_csv(out / "predictions.csv", index=False)
    # prediction figure for one sample
    row = df.iloc[4]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for T, col in ((23.0, C1), (105.0, C2)):
        pr = predict_performance(corr, row, 15.0, T)
        ax.plot(pr.f_hz / 1e6, pr.il_db, color=col, label=f"predicted IL at {T:.0f} °C (P(pass) = {pr.p_pass:.2f})")
        ax.fill_between(pr.f_hz / 1e6, pr.il_lo_db, pr.il_hi_db, color=col, alpha=0.2)
        if T == 23.0 and pr.limit_db is not None:
            ax.plot(pr.f_hz / 1e6, pr.limit_db, color=CK, ls="--", label="limit (1000BASE-T1 link segment)")
    ax.set_xlabel("frequency / MHz"), ax.set_ylabel("IL / dB (15 m)")
    ax.set_title(f"{row['sample_id']} ({row['design']}, {row['line']}): performance predicted from production data only", loc="left", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out / "prediction.png", dpi=130)
    plt.close(fig)
    summary["correlation_table"] = corr.table()
    summary["sensitivities"] = corr.sensitivities
    (out / "demo_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    return summary
