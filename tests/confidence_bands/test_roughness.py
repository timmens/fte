import numpy as np
import pytest
from fte.confidence_bands.roughness import _cov_to_corr, get_roughness_func
from fte.simulation.simulate import get_kernel
from numpy.testing import assert_array_almost_equal as aaae


@pytest.mark.parametrize("length_scale", [1 / np.sqrt(2), 1, np.sqrt(2)])
@pytest.mark.parametrize("num", [50, 100, 150])
@pytest.mark.parametrize("interpolator", ["RectBivariateSpline"])
def test_roughness_func(length_scale, num, interpolator):
    grid = np.linspace(0, 1, num=num)
    kernel = get_kernel(kernel_name="RBF", kernel_kwargs={"length_scale": length_scale})
    cov = kernel(grid)
    roughness_func = get_roughness_func(
        grid=grid,
        cov=cov,
        interpolator=interpolator,
    )
    expected = 1 / length_scale  # can be derived from structure of RBF
    got = roughness_func(grid)
    aaae(expected, got, decimal=5)


def test_cov_to_corr():
    cov = np.array([[1, 0, 0.5], [0, 2, 0], [0.5, 0, 3]])
    expected = np.array([[1, 0, 0.5 / np.sqrt(3)], [0, 1, 0], [0.5 / np.sqrt(3), 0, 1]])
    got = _cov_to_corr(cov)
    aaae(expected, got)
