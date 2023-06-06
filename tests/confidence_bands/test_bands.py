import numpy as np
import pytest
from fte.confidence_bands.bands import (
    _any_none,
    _constant_band_adjustment,
    _cov_to_corr,
    _get_moment_generating_func,
    _get_roughness_func,
)
from fte.simulation.simulate import get_kernel
from numpy.testing import assert_array_almost_equal as aaae
from scipy import stats


def test_cov_to_corr():
    cov = np.array([[1, 0, 0.5], [0, 2, 0], [0.5, 0, 3]])
    expected = np.array([[1, 0, 0.5 / np.sqrt(3)], [0, 1, 0], [0.5 / np.sqrt(3), 0, 1]])
    got = _cov_to_corr(cov)
    aaae(expected, got)


def test_any_none():
    assert _any_none(str, np.ones(2), "test", None)
    assert not _any_none(str, np.ones(2), "test")


@pytest.mark.parametrize("length_scale", [1 / np.sqrt(2), 1, np.sqrt(2)])
@pytest.mark.parametrize("num", [50, 100, 150])
@pytest.mark.parametrize("interpolator", ["RectBivariateSpline"])
def test_roughness_func(length_scale, num, interpolator):
    grid = np.linspace(0, 1, num=num)
    kernel = get_kernel(kernel_name="RBF", kernel_kwargs={"length_scale": length_scale})
    cov = kernel(grid)
    roughness_func = _get_roughness_func(
        grid=grid,
        cov=cov,
        interpolator=interpolator,
    )
    expected = 1 / length_scale  # can be derived from structure of RBF
    got = roughness_func(grid)
    aaae(expected, got, decimal=5)


@pytest.mark.parametrize("alpha", [0.001, 0.01, 0.05, 0.1])
@pytest.mark.parametrize("distribution", [stats.t(df=10), stats.norm])
def test_constant_band_zero_roughness(alpha, distribution):
    # test case follows from formulae when integral of roughness function is zero
    quantile = distribution.ppf(1 - alpha / 2)
    value = _constant_band_adjustment(
        roughness_func=lambda _: 0,
        alpha=alpha,
        distribution=distribution,
        grid=np.linspace(0, 1, num=2),
    )
    aaae(quantile, value)


TEST_CASES = [
    (stats.norm, 0, 1),
    (stats.norm, 1, np.exp(-0.5)),
    (stats.t(df=10), 0, 1),
    (stats.t(df=10), 1, (1 + 0.1) ** (-5)),
]


@pytest.mark.parametrize(("distribution", "x", "expected"), TEST_CASES)
def test_get_moment_generating_func(distribution, x, expected):
    mgf = _get_moment_generating_func(distribution)
    got = mgf(x)
    aaae(got, expected)


@pytest.mark.parametrize("distribution", [stats.t, stats.poisson])
def test_get_moment_generating_func_error(distribution):
    with pytest.raises(NotImplementedError):
        _get_moment_generating_func(distribution)
