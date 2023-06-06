import numpy as np
import pytest
from fte.confidence_bands.bands import estimate_confidence_band
from fte.simulation import SIMULATION_EXAMPLES_REGRESSION
from fte.simulation.coverage import simulate_coverage_simultaneous_confidence_bands
from fte.simulation.processes import simulate_gaussian_process

COVERAGE_RELATIVE_TOLERANCE = 0.05
COVERAGE_ABSOLUTE_TOLERANCE = 0.01


@pytest.mark.xfail()
@pytest.mark.parametrize("example", ["regression"])
def test_coverage_simulation_with_defaults(example):
    n_sims = 100
    alpha = 0.05

    simulation_kwargs = {"n_samples": 1_000, "n_periods": 100, "n_params": 1}
    band_kwargs = {
        "coef_id": 0,
        "alpha": alpha,
        "n_int": 1,
        "distribution": "normal",
        "numerical_options": {"raise_error": False},
    }

    res = simulate_coverage_simultaneous_confidence_bands(
        n_sims=n_sims,
        simulation_kwargs=simulation_kwargs,
        data_process_kwargs=SIMULATION_EXAMPLES_REGRESSION[example],
        band_kwargs=band_kwargs,
    )

    assert len(res["raw_results"]) == n_sims
    assert (
        (1 - alpha) * (1 - COVERAGE_RELATIVE_TOLERANCE)
        <= res["processed"]["coverage"]
        <= (1 - alpha) * (1 + COVERAGE_RELATIVE_TOLERANCE)
    )


@pytest.mark.slow()
@pytest.mark.parametrize("alpha", [0.01, 0.05])
def test_coverage_simulation_mean_estimate(alpha):
    rng = np.random.default_rng()

    n_points = 100
    n_samples = 300

    inside = 0
    n_sims = 100
    for _ in range(n_sims):
        error = simulate_gaussian_process(
            n_samples=n_samples,
            kernel="RBF",
            rng=rng,
            n_periods=n_points,
        )
        estimate = error.mean(axis=1)
        cov = error @ error.T / n_samples
        cov_estimate = cov / n_samples
        band = estimate_confidence_band(
            estimate=estimate,
            n_samples=n_samples,
            cov=cov_estimate,
            alpha=alpha,
            n_int=1,
            numerical_options={"raise_error": False},
            distribution="normal",
        )
        if np.all(band.lower <= 0) and np.all(band.upper >= 0):
            inside += 1

    coverage = inside / n_sims
    assert np.abs(coverage - (1 - alpha)) <= COVERAGE_ABSOLUTE_TOLERANCE
