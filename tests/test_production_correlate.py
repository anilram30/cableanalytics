import numpy as np
import pytest

from cableanalytics.correlate import fit_correlation, predict_coefficients
from cableanalytics.predict import predict_performance
from cableanalytics.production import (
    LINES,
    gauge_log,
    gauge_statistics,
    generate_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(120, seed=3)


def test_gauge_statistics_find_the_machine_period():
    line = LINES["L2"]                        # worn capstan
    z, d = gauge_log(line, 0.845, rng=np.random.default_rng(0))
    gs = gauge_statistics(z, d)
    assert abs(gs["gauge_period_m"] / (np.pi * line.capstan_d_m) - 1) < 0.03
    assert abs(gs["gauge_ripple_um"] - line.capstan_ecc_um) < 1.0
    line = LINES["L3"]                        # worn screw
    z, d = gauge_log(line, 0.845, rng=np.random.default_rng(0))
    gs = gauge_statistics(z, d)
    assert abs(gs["gauge_period_m"] / (line.line_speed_m_min / line.screw_rpm) - 1) < 0.03


def test_dataset_shape_and_physics(dataset):
    df = dataset
    assert len(df) == 120 and {"a_meas", "b_meas", "z_meas", "gauge_period_m", "rl_min_db"} <= set(df.columns)
    assert (df["z_true"].between(90, 115)).all()
    # worn lines show deeper Bragg dips than the healthy one
    assert df[df.line == "L2"]["rl_min_db"].mean() < df[df.line == "L1"]["rl_min_db"].mean() - 2


def test_correlation_beats_pure_physics_for_a(dataset):
    corr = fit_correlation(dataset, n_splits=4)
    s = {(x.target, x.model.split()[0]): x for x in corr.scores}
    assert s[("a", "P+")].r2 > s[("a", "P")].r2 and s[("a", "P+")].r2 > 0.95
    assert s[("z", "P")].rmse_rel < 0.01
    assert corr.sensitivities["a"]["physics term"] > 0.8


def test_prediction_intervals_cover(dataset):
    corr = fit_correlation(dataset, n_splits=4)
    hits = 0
    for _, row in dataset.iloc[:40].iterrows():
        p = predict_coefficients(corr, row)
        hits += p["a"]["lo90"] <= row["a_meas"] <= p["a"]["hi90"]
    assert hits >= 30           # ~90 % nominal coverage (in-sample, so slightly optimistic)
    pr = predict_performance(corr, dataset.iloc[0], 15.0, 105.0)
    assert 0 <= pr.p_pass <= 1 and pr.margin_db is not None and pr.il_hi_db[-1] > pr.il_lo_db[-1]
