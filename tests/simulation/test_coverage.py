import numpy as np
import pytest
from fte.confidence_bands.bands import estimate_confidence_band
from fte.montypy.coverage import coverage_simulation_study
from fte.simulation import SIMULATION_EXAMPLES_REGRESSION
from fte.simulation.coverage import simulate_coverage_simultaneous_confidence_bands


@pytest.mark.skip(reason="Fails at the moment.")
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
    raise AssertionError  # check coverage


@pytest.mark.slow()
@pytest.mark.parametrize("alpha", [0.1, 0.05, 0.01])
@pytest.mark.parametrize("distribution", ["normal", "t"])
def test_coverage_simulation_mean_estimate(alpha, distribution):
    n_samples = 1_000

    def _sim_func(_id):
        rng = np.random.default_rng(seed=_id)

        # simulate data
        grid = np.linspace(0, 1, num=50)

        def rbf(s, t):
            return np.exp(-100 * (s - t) ** 2)

        cov = rbf(grid.reshape(-1, 1), grid.reshape(1, -1))
        data = rng.multivariate_normal(
            mean=np.zeros(len(grid)),
            cov=cov,
            size=n_samples,
        ).T

        # estimate mean and covariance
        estimate = data.mean(axis=1)
        error = data - estimate.reshape(-1, 1)
        cov_estimate = np.matmul(error, error.T) / (n_samples**2)

        # compute confidence band
        band = estimate_confidence_band(
            estimate=estimate,
            n_samples=n_samples,
            cov=cov_estimate,
            alpha=alpha,
            n_int=1,
            numerical_options={"raise_error": False},
            distribution=distribution,
        )
        return {
            "true": 0,
            "confidence_interval": band,
        }

    res = coverage_simulation_study(n_sims=200, sim_func=_sim_func)
    coverage = res["processed"]["coverage"]

    assert coverage >= 1 - alpha
    assert np.allclose(coverage, 1 - alpha, atol=alpha / 2)
