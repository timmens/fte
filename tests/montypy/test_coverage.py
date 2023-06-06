import numpy as np
from fte.confidence_bands.bands import Band
from fte.montypy.coverage import (
    _check_estimate_inside_confidence_interval,
    _coverage_results_processor,
    coverage_simulation_study,
)
from numpy.testing import assert_almost_equal


def test_check_estimate_inside_confidence_interval():
    res = {
        "confidence_interval": Band(lower=np.zeros(2), upper=np.ones(2), estimate=None),
        "true": 0.5 * np.ones(2),
    }
    assert _check_estimate_inside_confidence_interval(res)

    res = {
        "confidence_interval": Band(lower=np.ones(2), upper=np.ones(2), estimate=None),
        "true": np.zeros(2),
    }
    assert not _check_estimate_inside_confidence_interval(res)


def test_coverage_results_processor():
    res1 = {  # inside
        "confidence_interval": Band(lower=np.zeros(2), upper=np.ones(2), estimate=None),
        "true": 0.5 * np.ones(2),
    }
    res2 = {  # not inside
        "confidence_interval": Band(lower=np.ones(2), upper=np.ones(2), estimate=None),
        "true": np.zeros(2),
    }
    results = [res1, res2]
    expected = {
        "processed": [True, False],
        "coverage": 0.5,
    }
    got = _coverage_results_processor(results)
    assert expected == got


def test_coverage_simulation_study():
    def _sim_func(_id):
        # randomness in _sim_func implies a coverage of 0.25
        rng = np.random.default_rng(seed=_id)
        true = np.array([0, 1])
        return {
            "true": true,
            "confidence_interval": Band(
                lower=true + rng.uniform(-1, 1),
                upper=true + rng.uniform(-1, 1),
                estimate=None,
            ),
        }

    res = coverage_simulation_study(n_sims=2_000, sim_func=_sim_func)
    assert_almost_equal(res["processed"]["coverage"], 1 / 4, decimal=2)
