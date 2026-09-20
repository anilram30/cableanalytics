"""
Manufacturing -> physics correlation and prediction.

Three model families are compared on the production data set, all
predicting the laboratory's fitted loss coefficients and impedance from
production data alone:

P   *physics only*: a, b, Z computed from the design and materials with
    the nominal stranding factor and nominal loss tangent - no fitting.
P+  *physics + correction*: linear regression, in log space, of the
    measured values on the physics prediction plus production features
    (line, foaming, gauge statistics), which learns what the physics does
    not know (each line's stranding factor, lot contamination).  Fitted
    with leave-lot-out cross-validation so that the score is honest about
    new lots.
ML  *black box*: gradient-boosted trees on the raw production features,
    same cross-validation, as a baseline that ignores the physics.

Prediction intervals come from the cross-validated residuals (split
conformal): the 5-95 % interval half-width is the 90 % quantile of the
absolute CV residual.  Sensitivities are the standardised coefficients of
the P+ model, i.e. how many standard deviations of the target one standard
deviation of each feature moves.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from .physics import Design, hf_properties

__all__ = ["physics_prediction", "Correlation", "fit_correlation", "TARGETS"]

TARGETS = {"a": ("a_meas", "a_phys", True), "b": ("b_meas", "b_phys", True), "z": ("z_meas", "z_phys", False)}


def physics_prediction(df: pd.DataFrame, temperature_c: float = 23.0) -> pd.DataFrame:
    out = df.copy()
    a, b, z = [], [], []
    for _, r in df.iterrows():
        p = hf_properties(Design(r["alloy"], r["d_cond_mm"], r["insulation"], r["foaming"], r["d_ins_mm"]), temperature_c)
        a.append(p["a"]), b.append(p["b"]), z.append(p["z_diff"])
    out["a_phys"], out["b_phys"], out["z_phys"] = a, b, z
    return out


def _features(df: pd.DataFrame, target: str) -> tuple[np.ndarray, list[str]]:
    cols = ["foaming", "gauge_std_mm", "gauge_ripple_um", "line_speed_m_min"]
    X = [df[c].to_numpy(float) for c in cols]
    names = list(cols)
    for line in sorted(df["line"].unique()):
        X.append((df["line"] == line).to_numpy(float))
        names.append(f"line={line}")
    return np.column_stack(X), names


@dataclass
class ModelScore:
    model: str
    target: str
    r2: float
    rmse_rel: float                 # relative RMSE (fraction of the mean)
    interval_halfwidth_rel: float   # 90 % conformal half-width, relative


@dataclass
class Correlation:
    scores: list[ModelScore]
    sensitivities: dict            # target -> {feature: standardised coefficient}
    residual_quantiles: dict       # target -> 90 % abs residual (log space for a, b; relative for z)
    models: dict = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)

    def table(self) -> str:
        out = ["| target | model | R² (CV) | rel. RMSE | 90 % interval ± |", "|---|---|---|---|---|"]
        for s in self.scores:
            out.append(f"| {s.target} | {s.model} | {s.r2:.3f} | {100 * s.rmse_rel:.1f} % | {100 * s.interval_halfwidth_rel:.1f} % |")
        return "\n".join(out)


def fit_correlation(df: pd.DataFrame, n_splits: int = 5, seed: int = 0) -> Correlation:
    d = physics_prediction(df)
    groups = d["lot"].to_numpy()
    gkf = GroupKFold(n_splits=n_splits)
    scores, sens, quant, models = [], {}, {}, {}
    feat_names = None
    for tgt, (ycol, pcol, use_log) in TARGETS.items():
        y = d[ycol].to_numpy(float)
        yp = d[pcol].to_numpy(float)
        Xf, names = _features(d, tgt)
        feat_names = names
        ty = np.log(y) if use_log else y / yp            # target transform
        tp = np.log(yp) if use_log else np.ones_like(y)  # physics term
        # --- P: physics only
        pred_P = yp
        # --- P+: ridge on [physics term, features] with leave-lot-out CV
        Xpp = np.column_stack([tp, Xf])
        mu, sd = Xpp.mean(0), Xpp.std(0) + 1e-12
        Xs = (Xpp - mu) / sd
        pred_PP = np.empty_like(y)
        pred_ML = np.empty_like(y)
        for tr, te in gkf.split(Xs, ty, groups):
            m = Ridge(alpha=1e-3).fit(Xs[tr], ty[tr])
            t_hat = m.predict(Xs[te])
            pred_PP[te] = np.exp(t_hat) if use_log else t_hat * yp[te]
            g = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05, random_state=seed)
            g.fit(_raw(d.iloc[tr]), ty[tr])
            t_hat = g.predict(_raw(d.iloc[te]))
            pred_ML[te] = np.exp(t_hat) if use_log else t_hat * yp[te]
        full = Ridge(alpha=1e-3).fit(Xs, ty)
        models[tgt] = {"ridge": full, "mu": mu, "sd": sd, "use_log": use_log}
        sens[tgt] = {("physics term" if i == 0 else names[i - 1]): float(c / ty.std()) for i, c in enumerate(full.coef_)}
        for name, pred in (("P physics", pred_P), ("P+ physics+correction", pred_PP), ("ML boosted trees", pred_ML)):
            r = (np.log(pred) - np.log(y)) if use_log else (pred - y) / y
            ss_res = np.sum((pred - y) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            scores.append(ModelScore(name, tgt, 1 - ss_res / ss_tot, float(np.sqrt(np.mean(r ** 2))),
                                     float(np.quantile(np.abs(r), 0.9))))
            if name.startswith("P+"):
                quant[tgt] = float(np.quantile(np.abs(r), 0.9))
    return Correlation(scores, sens, quant, models, feat_names)


def _raw(df: pd.DataFrame) -> np.ndarray:
    cols = ["d_cond_mm", "foaming", "d_ins_mm", "line_speed_m_min", "screw_rpm", "gauge_std_mm", "gauge_ripple_um"]
    X = [df[c].to_numpy(float) for c in cols]
    for a in ["Cu-ETP", "CuAg0.1", "CuSn0.3", "CuMg0.2"]:
        X.append((df["alloy"] == a).to_numpy(float))
    for ins in ["PE", "PP", "FEP"]:
        X.append((df["insulation"] == ins).to_numpy(float))
    for line in ["L1", "L2", "L3"]:
        X.append((df["line"] == line).to_numpy(float))
    return np.column_stack(X)


def predict_coefficients(corr: Correlation, row: pd.Series | dict) -> dict:
    """P+ prediction for one production row: a, b, z with 90 % intervals."""
    r = pd.DataFrame([dict(row)])
    d = physics_prediction(r)
    out = {}
    for tgt, (ycol, pcol, use_log) in TARGETS.items():
        yp = float(d[pcol].iloc[0])
        Xf, _ = _features(pd.concat([d] * 1, ignore_index=True), tgt)
        # one-hot columns for lines absent in this row are zeros; _features derives them from the row's
        # own line only, so rebuild the column set in the training order
        feats = {"foaming": d["foaming"].iloc[0], "gauge_std_mm": d["gauge_std_mm"].iloc[0],
                 "gauge_ripple_um": d["gauge_ripple_um"].iloc[0], "line_speed_m_min": d["line_speed_m_min"].iloc[0]}
        for n in corr.feature_names[4:]:
            feats[n] = 1.0 if n == f"line={d['line'].iloc[0]}" else 0.0
        x = np.array([np.log(yp) if use_log else 1.0] + [feats[n] for n in corr.feature_names])
        m = corr.models[tgt]
        t_hat = float(m["ridge"].predict(((x - m["mu"]) / m["sd"])[None, :])[0])
        val = np.exp(t_hat) if use_log else t_hat * yp
        q = corr.residual_quantiles[tgt]
        lo, hi = (val * np.exp(-q), val * np.exp(q)) if use_log else (val * (1 - q), val * (1 + q))
        out[tgt] = {"value": float(val), "lo90": float(lo), "hi90": float(hi), "physics": yp}
    return out
