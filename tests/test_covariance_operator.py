import numpy as np
import pytest
from fte.covariance_operator import _get_operator_homoskedasticity
from fte.covariance_operator import _outer_product_along_first_dim
from fte.covariance_operator import cov_from_residuals
from fte.covariance_operator import get_covariance_operator
from numpy.testing import assert_array_almost_equal as aaae
from numpy.testing import assert_array_equal


@pytest.mark.parametrize("cov_type", ["homoskedastic", "HC0", "HC1"])
@pytest.mark.parametrize("coef_id", [0, 1])
def test_cov_from_residuals(cov_type, coef_id):
    x = np.arange(6).reshape(3, 2)  # (n_samples, n_params) = (3, 2)
    residuals = 1 + np.arange(12).reshape(3, 4)  # (n_samples, n_points) = (3, 4)
    cov = cov_from_residuals(residuals, x=x, coef_id=coef_id, cov_type=cov_type)
    assert cov.shape == (4, 4)


def test_outer_product_along_first_dim():
    arr = np.arange(6).reshape(3, 2)
    got = _outer_product_along_first_dim(arr)
    expected = np.array([[[0, 0], [0, 1]], [[4, 6], [6, 9]], [[16, 20], [20, 25]]])
    assert_array_equal(got, expected)


def test_get_operator_homoskedasticity_simple():
    error_kernel = np.array([[1]])
    information_matrix = np.eye(1)
    operator = _get_operator_homoskedasticity(error_kernel, information_matrix)
    assert operator(0, 0) == 1


def test_get_operator_homoskedasticity_general():
    n_params = 3
    error_kernel = np.arange(4).reshape(2, 2)
    information_matrix = np.eye(n_params)
    operator = _get_operator_homoskedasticity(error_kernel, information_matrix)
    for i in [0, 1]:
        for j in [0, 1]:
            assert_array_equal(operator(i, j), error_kernel[i, j] * information_matrix)


def test_get_covariance_operator_raises_error():
    with pytest.raises(ValueError):  # noqa: PT011
        # wrong cov_type
        get_covariance_operator(None, None, cov_type="heteroskedastic")

    with pytest.raises(ValueError):  # noqa: PT011
        # 1d x
        get_covariance_operator(np.array([1]), np.eye(2))

    with pytest.raises(ValueError):  # noqa: PT011
        # 1d residuals
        get_covariance_operator(np.eye(2), np.array([1]))

    with pytest.raises(ValueError):  # noqa: PT011
        # x and residuals have different n_samples
        get_covariance_operator(np.eye(2), np.eye(3))


@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_get_covariance_operator_data(scale):
    n_samples = 10_000
    x = np.row_stack((np.eye(2), np.zeros((n_samples - 2, 2))))
    rng = np.random.default_rng()
    residuals = rng.normal(scale=scale, size=(n_samples, 1))
    operator = get_covariance_operator(residuals, x=x)
    aaae(np.diag(operator(0, 0)), [scale**2, scale**2], decimal=1)
