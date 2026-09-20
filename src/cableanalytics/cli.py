"""Command line: ``cableanalytics fit | derate | attribute | dataset | correlate | predict | demo``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__


def _line_args(args) -> dict | None:
    if args.line_speed is None:
        return None
    return {"line_speed_m_min": args.line_speed, "screw_rpm": args.screw_rpm, "capstan_d_m": args.capstan_d}


def cmd_fit(args):
    from .lab import evaluate_touchstone
    r = evaluate_touchstone(args.file, args.length, _line_args(args))
    print(json.dumps(r, indent=1, default=float))
    return 0


def cmd_derate(args):
    import re

    import numpy as np

    from .derating import fit_derating
    from .lab import il_rl_from_touchstone
    from .lossfit import fit_loss
    files = sorted(Path(args.dir).glob("*.s4p"))
    temps, fits = [], []
    for p in files:
        m = re.search(r"T([+-]?\d+)C", p.name)
        if not m:
            continue
        T = float(m.group(1))
        f, il, rl, s21 = il_rl_from_touchstone(p)
        temps.append(T)
        fits.append(fit_loss(f, il, args.length, fmin=5e6, basis=args.basis))
    if len(temps) < 2:
        sys.exit("need at least two files named *T<temp>C*.s4p")
    der = fit_derating(temps, fits)
    out = der.to_dict()
    out["fits"] = {str(T): ft.to_dict() for T, ft in zip(temps, fits)}
    f = np.linspace(1e6, 600e6, 600)
    if args.cable_type:
        from cablecheck.limits.library import load_cable_type
        ct = load_cable_type(args.cable_type)
        lim, mask = ct.limit_for("insertion_loss").evaluate(f, args.length)
        rows = {}
        for T in (-40, 23, 85, 105, 125):
            mg, fw = der.hot_margin(f[mask], lim[mask], args.length, T)
            rows[str(T)] = {"margin_db": mg, "f_worst_mhz": fw / 1e6, "derating_600MHz": float(der.derating(600e6, T)),
                            "max_length_m": der.max_length(f[mask], lim[mask], T)}
        out["hot_margins"] = rows
    print(json.dumps(out, indent=1, default=float))
    return 0


def cmd_dataset(args):
    from .production import generate_dataset
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = generate_dataset(args.n, args.seed, gauge_dir=out / "gauge")
    df.to_csv(out / "production_dataset.csv", index=False)
    print(f"wrote {out / 'production_dataset.csv'} ({len(df)} samples) and {args.n and 6} gauge logs")
    return 0


def cmd_correlate(args):
    import pandas as pd

    from .correlate import fit_correlation
    from .figures import correlation_figures
    df = pd.read_csv(args.csv)
    corr = fit_correlation(df)
    print(corr.table())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "correlation_table.md").write_text(corr.table())
    (out / "sensitivities.json").write_text(json.dumps(corr.sensitivities, indent=1))
    correlation_figures(df, corr, out)
    return 0


def cmd_predict(args):
    import pandas as pd

    from .correlate import fit_correlation
    from .predict import predict_performance
    df = pd.read_csv(args.csv)
    corr = fit_correlation(df)
    row = df[df["sample_id"] == args.sample].iloc[0]
    pr = predict_performance(corr, row, args.length, args.temperature, cable_type=args.cable_type)
    d = pr.to_dict()
    d["measured"] = {"a": float(row["a_meas"]), "b": float(row["b_meas"]), "z": float(row["z_meas"])}
    print(json.dumps(d, indent=1, default=float))
    return 0


def cmd_attribute(args):
    from .periodic import attribute
    a = attribute(args.period, args.line_speed, args.screw_rpm, args.capstan_d)
    print(json.dumps(a.to_dict(), indent=1, default=float))
    return 0


def cmd_demo(args):
    from .figures import run_demo
    out = args.out or args.out_opt or "docs/figures"
    run_demo(Path(out), Path(args.data))
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cableanalytics", description="Loss, temperature and production analytics")
    p.add_argument("--version", action="version", version=f"cableanalytics {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    def line_opts(sp):
        sp.add_argument("--line-speed", type=float, help="m/min"), sp.add_argument("--screw-rpm", type=float)
        sp.add_argument("--capstan-d", type=float, help="capstan diameter in m")

    a = sub.add_parser("fit", help="three-term fit + periodic signature of one Touchstone file")
    a.add_argument("file"), a.add_argument("--length", type=float, required=True)
    line_opts(a)
    a.set_defaults(func=cmd_fit)
    d = sub.add_parser("derate", help="derating model from a folder of *T<temp>C*.s4p files")
    d.add_argument("dir"), d.add_argument("--length", type=float, required=True)
    d.add_argument("--basis", default="physical", choices=["physical", "standard"])
    d.add_argument("--cable-type", default="1000base-t1-link-segment")
    d.set_defaults(func=cmd_derate)
    t = sub.add_parser("attribute", help="attribute a ripple period to a machine element")
    t.add_argument("period", type=float, help="ripple period in m")
    line_opts(t)
    t.set_defaults(func=cmd_attribute)
    s = sub.add_parser("dataset", help="generate the synthetic production data set")
    s.add_argument("--out", default="data"), s.add_argument("--n", type=int, default=240), s.add_argument("--seed", type=int, default=7)
    s.set_defaults(func=cmd_dataset)
    c = sub.add_parser("correlate", help="manufacturing -> physics correlation on a data set")
    c.add_argument("csv"), c.add_argument("--out", default="correlation")
    c.set_defaults(func=cmd_correlate)
    r = sub.add_parser("predict", help="predict IL(f, T) and pass probability from production data")
    r.add_argument("csv"), r.add_argument("--sample", required=True), r.add_argument("--length", type=float, default=15.0)
    r.add_argument("--temperature", type=float, default=105.0), r.add_argument("--cable-type", default="1000base-t1-link-segment")
    r.set_defaults(func=cmd_predict)
    m = sub.add_parser("demo", help="run the whole chain and write the report figures")
    # positional output directory, matching every other package in the toolchain;
    # --out stays as an alias so existing invocations keep working
    m.add_argument("out", nargs="?", default=None, help="output directory (default: docs/figures)")
    m.add_argument("--out", dest="out_opt", default=None), m.add_argument("--data", default="data")
    m.set_defaults(func=cmd_demo)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
