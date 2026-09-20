# cableanalytics — production to physics correlation

**Connects production-line records to measured high-frequency cable performance, and predicts one from the other.**

[![CI](https://github.com/anilram30/cableanalytics/actions/workflows/ci.yml/badge.svg)](https://github.com/anilram30/cableanalytics/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-14%20passing-brightgreen)](tests/)
[![Report](https://img.shields.io/badge/report-8%20pages-informational)](docs/report.pdf)

> **Part of the [HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** — seven packages that take a high-frequency cable from a raw measurement to a predicted Ethernet link.
> 
> [A · cablecheck](https://github.com/anilram30/cablecheck)  ·  [B · labauto](https://github.com/anilram30/labauto)  ·  [C · shieldeval](https://github.com/anilram30/shieldeval)  ·  [D · zprofile](https://github.com/anilram30/zprofile)  ·  **E · cableanalytics**  ·  [F · labplatform](https://github.com/anilram30/labplatform)  ·  [G · linktwin](https://github.com/anilram30/linktwin)

---

## The problem it solves

A cable's electrical performance is decided on the extrusion line, hours before anybody measures it. `cableanalytics` closes that loop. It fits the physical loss model to measurements, derives how performance derates with temperature, and correlates manufacturing parameters — line speed, diameter variation, material, foaming — against the resulting high-frequency behaviour. The result is a model that predicts a cable's loss and impedance, with honest prediction intervals, from its production record alone.

## At a glance

|  |  |
|---|---|
| **Takes** | Measured loss curves, climate-chamber temperature sweeps and extrusion-line production records |
| **Produces** | Fitted loss coefficients, a temperature derating model, and predicted loss and impedance with 90 % prediction intervals |
| **Checked against** | Cross-validation against held-out samples, with the physics-only prediction as the baseline to beat |
| **Technical report** | [`docs/report.pdf`](docs/report.pdf) — 8 pages, 16 references, every method stated with its mathematics and its limitations |
| **Tests** | 14, run against Python 3.11, 3.12 and 3.13 on every push |
| **Data** | Entirely synthetic. No proprietary or customer measurements are used anywhere in this toolchain. |

## Install

Python 3.11 or newer.

```sh
pip install "git+https://github.com/anilram30/cablecheck.git" \
            "git+https://github.com/anilram30/cableanalytics.git"
```

---

## What it does

| module | what it does |
|---|---|
| `physics` | design → `a, b, c0, Z, v` (alloy resistivity, conductor/insulation diameters, foaming via Lichtenecker, materials' temperature coefficients) |
| `lossfit` | robust (Huber) weighted three-term fit with covariance; **physical** basis `c0 + a√f + b f` (interpretable) and the standards' `a√f + b f + c/√f` |
| `derating` | climate-chamber sweeps → `a(T)`, `b(T)` material coefficients, derating curves, hot margin and maximum length against a `cablecheck` limit |
| `periodic` | Bragg resonance in return loss / periodogram of the impedance profile → ripple period → attribution to capstan / extruder screw (harmonics, take-up reel as non-stationary fallback) |
| `production` | synthetic production data set (240 samples, 3 lines, 4 designs, gauge logs) with realistic nuisances |
| `correlate` | physics-only, physics+correction (ridge in log space, leave-lot-out CV) and boosted-tree models; conformal intervals; standardised sensitivities |
| `predict` | production row → IL(f, T) with 5–95 % band, margin and probability of passing a limit |
| `lab` | the laboratory path on Touchstone files (through `cablecheck`) |

## Install and test

```bash
pip install -e ../cablecheck
pip install -e ".[dev]"
pytest              # 14 tests, ~1.5 min
```

## Use

```bash
cableanalytics fit sample.s4p --length 15 --line-speed 80 --screw-rpm 55 --capstan-d 0.315
cableanalytics derate chamber_folder/ --length 15          # files named *T+105C*.s4p etc.
cableanalytics attribute 0.99 --line-speed 80 --screw-rpm 55 --capstan-d 0.315
cableanalytics dataset --out data
cableanalytics correlate data/production_dataset.csv --out correlation/
cableanalytics predict data/production_dataset.csv --sample S0005 --temperature 105 --length 15
cableanalytics demo --out docs/figures --data data      # everything, for the report
```

## Documentation

`docs/report.md` / `docs/report.pdf`.
---

## Contributing

Bug reports, questions about the methods, and pull requests are all welcome — see
[CONTRIBUTING.md](CONTRIBUTING.md). Numerical changes need a numerical test, and a change to a method
is also a change to `docs/report.md`.

## Licence and attribution

MIT — see [LICENSE](LICENSE). Author: Sreeram Anil.

Built with AI assistance; the commit history records it. The engineering decisions, the validation
strategy and the limitations stated in the report are the substance of the work.

Part of the **[HF cable toolchain](https://github.com/anilram30/hf-cable-toolchain)** · [Report an issue](https://github.com/anilram30/cableanalytics/issues) ·
[Changelog](CHANGELOG.md)
