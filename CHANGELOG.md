# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-09-18

### Added
- First release: loss, temperature and production analytics.
- Loss fitting on the physical basis `c/√f + a√f + b·f` with weighted least squares and parameter
  covariance, giving conductor, dielectric and DC-resistance contributions separately.
- Derating model from climate-chamber sweeps, with material coefficients fitted across temperature and
  each point weighted by its fit uncertainty.
- Design physics: conductor alloy resistivity, insulation permittivity and loss tangent, foaming, and
  their temperature dependence.
- Manufacturing-to-performance correlation: physics prediction, ridge-regularised correction and a
  boosted-tree comparison, cross-validated, with 90 % prediction intervals on `a`, `b` and `Z`.
- Periodic-structure detection from return loss and from an impedance profile.
- Production data-set generator with gauge logs, CLI, figures, 14 tests and an 8-page technical report.
