import numpy as np
from ekf import simulate, run, position_rmse, gps_only_track, nees


def test_fusion_beats_gps_only():
    data = simulate(seed=1)
    est, _ = run(data)
    gps = np.column_stack([gps_only_track(data), np.zeros((len(data["t"]), 2))])
    assert position_rmse(est, data["truth"]) < position_rmse(gps, data["truth"])


def test_dropout_error_stays_bounded():
    data = simulate(seed=2)
    est, _ = run(data)
    drop = (data["t"] >= 60) & (data["t"] < 80)
    assert position_rmse(est, data["truth"], drop) < 25.0


def test_covariance_stays_symmetric_positive():
    data = simulate(seed=3, seconds=30)
    _, covs = run(data)
    for P in covs:
        assert np.allclose(P, P.T, atol=1e-8)
        assert np.all(np.linalg.eigvalsh(P) > 0)


def test_filter_is_roughly_consistent():
    data = simulate(seed=4)
    est, covs = run(data)
    assert 0.3 < nees(est[50:], covs[50:], data["truth"][50:]) < 8.0
