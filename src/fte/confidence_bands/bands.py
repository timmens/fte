from functools import partial
from typing import Any
from typing import Dict
from typing import NamedTuple
from typing import Tuple
from typing import Union

import numpy as np
from fte.covariance_operator import cov_from_residuals
from fte.simulation.simulate import SimulatedData
from scipy import integrate
from scipy import interpolate
from scipy import optimize
from scipy.stats import norm
from scipy.stats import t


MAX_INTEGRATION_ERROR = 1e-7


class Band(NamedTuple):
    """Confidence band data container."""

    lower: np.ndarray
    upper: np.ndarray
    estimate: np.ndarray


def estimate_confidence_band(
    result: Dict = None,
    estimate: np.ndarray = None,
    cov: np.ndarray = None,
    coef_id: int = None,
    data: SimulatedData = None,
    n_samples: int = None,
    cov_type: str = "homoskedastic",
    alpha: float = 0.05,
    distribution: str = "t",
    n_int: int = 1,
    grid: np.ndarray = None,
    numerical_options: Dict = None,
):
    """Estimate confidence band from covariance information.

    Args:
        result (dict): Results dictionary returned by fitter. Default None.
        estimate (np.ndarray): Estimate. 1d array of shape (n_points,). Default None.
        n_samples (int): Number of samples used to generate estimates. Default None.
        cov (np.ndarray): Covariance of estimate. 2d array of shape (n_points, n_points)
            Default None.
        alpha (float): Confidence level. Must be in (0, 1). Default 0.05.
        distribution (scipy.stats.rv_continous): A scipy.stats distribution.
            Currently only t and normal distribution are supported.
        n_int (int): Number of intervals to use in computation of band adjustment.
            Default 1.
        grid (np.ndarray): Time grid. Default None.
        numerical_options (dict): Options kwargs unpacked in _constant_band_adjustment.
            Can contain keys ['root_method', 'root_options', 'root_options',
            'raise_error']. See docstring of function for details.

    """
    # Consolidate inputs
    # ==================================================================================
    if numerical_options is None:
        numerical_options = {}

    # retrieve objects if result argument is given
    if result is not None and "treatment_effect" in result:
        estimate = result["treatment_effect"]
        cov = result.get("cov", None)
        n_samples = result.get("n_samples", None)
    elif result is not None and "slopes" in result:
        slopes = result.get("slopes", None)
        intercept = result.get("intercept", None)
        data = result.get("data", None)

    if not _any_none(estimate, cov, n_samples):
        # estimate band from kernel information directly
        n_points = len(estimate)
    elif not _any_none(slopes, intercept, coef_id, data):
        # estimate band from model residual information
        n_samples, n_points = data.y.shape
        y_predicted = intercept + data.x @ slopes.T
        residuals = data.y - y_predicted
        estimate = slopes[:, coef_id]
        cov = cov_from_residuals(
            residuals=residuals, x=data.x, coef_id=coef_id, cov_type=cov_type
        )
    else:
        msg = (
            "Either ('estimate', 'cov', 'n_samples'), ('slopes', 'intercept', "
            "'coef_id', 'data') or ('result') have to be non None."
        )
        raise ValueError(msg)

    built_in_distributions = {"normal": norm, "t": t(df=n_samples - 1)}
    distribution = built_in_distributions[distribution]

    # Estimate roughness function
    # ==================================================================================
    grid = np.linspace(0, 1, num=n_points) if grid is None else grid
    roughness_func = _get_roughness_func(grid, cov=cov)

    # Estimate confidence band
    # ==================================================================================
    band = _confidence_band_from_roughness(
        estimate=estimate,
        roughness_func=roughness_func,
        cov=cov,
        alpha=alpha,
        n_int=n_int,
        distribution=distribution,
        grid=grid,
        numerical_options=numerical_options,
    )
    return band


def _confidence_band_from_roughness(
    estimate, cov, roughness_func, alpha, distribution, n_int, grid, numerical_options
):
    """Estimate confidence band from covariance and roughness function.

    Args:
        estimate (np.ndarray): Estimate. 1d array of shape (n_points,).
        cov (np.ndarray): Covariance of estimate. 2d array of shape (n_points, n_points)
        roughness_func (callable): The roughness function.
        alpha (float): Confidence level. Must be in (0, 1).
        distribution (scipy.stats.rv_continous): A scipy.stats distribution.
            Currently only t and normal distribution are supported.
        n_int (int): Number of intervals to use in computation of band adjustment.
        numerical_options (dict): Options kwargs unpacked in _constant_band_adjustment.
            Can contain keys ['root_method', 'root_options', 'root_options',
            'raise_error']. See docstring of function for details.

    """
    if n_int == 1:
        adjustment = _constant_band_adjustment(
            roughness_func,
            alpha=alpha,
            distribution=distribution,
            grid=grid,
            **numerical_options,
        )
    else:
        adjustment = _nonconstant_band_adjustment(
            roughness_func,
            alpha=alpha,
            distribution=distribution,
            grid=grid,
        )

    lower = estimate - adjustment * np.sqrt(np.diag(cov))
    upper = estimate + adjustment * np.sqrt(np.diag(cov))

    band = Band(lower=lower, upper=upper, estimate=estimate.copy())
    return band


# ======================================================================================
# Roughness function
# ======================================================================================


def _get_roughness_func(grid, cov, interpolator="RectBivariateSpline", info=False):
    """Get the parameter roughness function.

    Args:
        grid (np.ndarray): Time grid. Default None.
        cov (np.ndarray): The estimated covariance matrix.
        interpolator (str or callable): The interpolator which is used to smooth
            the covariance matrix. Implemented are {"interp2d", "RectBivariateSpline"}.
            Default is RectBivariateSpline.
        info (bool): Add extra information to output function.

    Returns:
        callable: The parameter roughness function.

    """
    # Validate inputs
    # ==================================================================================

    built_in_interpolator = {
        "interp2d": partial(interpolate.interp2d, kind="cubic"),
        "RectBivariateSpline": interpolate.RectBivariateSpline,
    }

    if isinstance(interpolator, str) and interpolator in built_in_interpolator:
        interpolator = built_in_interpolator[interpolator]
        _name = interpolator
    elif callable(interpolator):
        _name = interpolator.__name__
    else:
        msg = f"Interpolator must be callable or in {built_in_interpolator.keys()}."
        raise ValueError(msg)

    # Compute roughness function
    # ==================================================================================

    corr = _cov_to_corr(cov)
    smooth_corr = interpolator(grid, grid, corr)

    @np.vectorize
    def _roughness(t):
        return np.sqrt(smooth_corr(t, t, dx=1, dy=1))

    if info:
        info = {"smooth_corr": smooth_corr, "interpolator_name": _name}
        _roughness.info = info

    return _roughness


# ======================================================================================
# Constant band
# ======================================================================================


def _constant_band_adjustment(
    roughness_func: callable,
    alpha: float,
    distribution: Union[norm, t],
    grid: np.ndarray,
    root_method: str = "brentq",
    root_bracket: Tuple[float] = (0.0, 15.0),
    root_options: Dict[str, Any] = None,
    raise_error: bool = True,
):
    """Get the band adjustment for the constant case.

    This is equivalent to standard Kac-Rice band adjustment.

    Args:
        roughness_func (callable): The roughness function.
        alpha (float): Confidence level. Must be in (0, 1).
        distribution (scipy.stats.rv_continous): A scipy.stats distribution.
            Currently only t and normal distribution are supported.
        grid (np.ndarray): Time grid.
        root_method (str):  Type of solver. Should be in {'bisect', 'brentq', 'brenth'
            'ridder', 'toms748', 'newton', 'secant', 'halley'}. Default 'brentq'.
        root_bracket (tuple): An interval bracketing a root. Default is (0.0, 15.0).
        root_options (dict): Additional options passed to scipy.optimize.root_scalar.
            Default None.
        raise_error (bool): Whether to raise an error if integration or root finding
            did not converge. Default True.

    """
    # Consolidate inputs
    # ==================================================================================
    if root_options is None:
        root_options = {}

    # Integrate roughness function
    # ==================================================================================
    integrated_roughness, integration_error = integrate.quad(
        roughness_func, a=grid.min(), b=grid.max()
    )

    if raise_error and integration_error > MAX_INTEGRATION_ERROR:
        msg = (
            "Integration of roughness function is not accurate enough. You can "
            "deactivate this error by passing raise_error=False to this function."
        )
        raise RuntimeError(msg)

    # Get moment generating function used in root finding
    # ==================================================================================
    mgf = _get_moment_generating_func(distribution)

    # Define function of which we want to find the root
    # ==================================================================================
    root_func = partial(
        _root_func,
        alpha=alpha,
        distribution=distribution,
        integrated_roughness=integrated_roughness,
        mgf=mgf,
    )

    # Find root
    # ==================================================================================
    res = optimize.root_scalar(
        root_func, method=root_method, bracket=root_bracket, **root_options
    )

    if raise_error and not res.converged:
        msg = (
            "Root finding has not converged. Please use different optimize_kwargs, or "
            "deactivate this error by passing raise_error=False to this function."
        )
        raise RuntimeError(msg)

    return res.root


def _root_func(x, alpha, distribution, integrated_roughness, mgf):
    r"""Calculate function values.

    _root_func(x) = P[X > x] + (\int tau(t) dt) MGF(x) / (2 pi) - alpha / 2

    Args:
        x (float or np.ndarray): Function input.
        alpha (float): Confidence level. Must be in (0, 1).
        distribution (scipy.stats.rv_continous): A scipy.stats distribution.
            Currently only t and normal distribution are supported.
        integrated_roughness (float): Integral of roughness function from grid.min()
            to grid.max().
        mgf (callable): Moment generating function.

    Returns:
        float or np.ndarray: Function value. If x is an np.ndarray so is the output.

    """
    out = distribution.sf(x) + integrated_roughness * mgf(x) / (2 * np.pi) - alpha / 2
    return out


def _get_moment_generating_func(distribution):
    r"""Get moment generating function of \mathcal{V}.

    Args:
        distribution (scipy.stats.rv_continous): A scipy.stats distribution.
            Currently only t and normal distribution are supported.

    Returns:
        callable: The moment generating function.

    """
    try:
        name = distribution.name
    except AttributeError:
        name = distribution.dist.name

    def _normal(x):
        return np.exp(-(x**2) / 2)

    def _t(x, df):
        return (1 + (x**2) / df) ** (-df / 2)

    if name == "norm":
        _mgf = _normal
    elif name == "t" and hasattr(distribution, "kwds"):
        _mgf = partial(_t, df=distribution.kwds["df"])
    else:
        raise NotImplementedError("distribution has to be normal or t.")

    return _mgf


# ======================================================================================
# Non-constant band
# ======================================================================================


def _nonconstant_band_adjustment(
    roughness_func: callable,
    alpha: float,
    distribution: Union[norm, t],
    n_int: int,
    grid: np.ndarray,
):
    coef = np.zeros(n_int + 1)  # container for coefficients of piecewise-linear func

    knots = np.linspace(grid.min(), grid.max(), num=n_int + 1)
    tau_init, _ = integrate.quad(roughness_func, a=knots[0], b=knots[1])
    root_fun = _get_root_func(tau_init, n_int, alpha, distribution)
    root = optimize.brentq(root_fun, a=0.0, b=10.0)

    coef[0] = root

    pwl_func = _get_piecewise_linear_function(n_int)

    for j in range(1, n_int):

        root_func = _get_root_func_j(
            j, coef, n_int, pwl_func, knots, roughness_func, distribution, alpha
        )
        root = optimize.brentq(root_func, a=-20.0, b=20.0)
        coef[j] = root

    critical_values = pwl_func(grid, coef=coef, knots=knots)  # noqa: F841
    raise NotImplementedError("This band adjument is not properly implemented yet.")


def _get_root_func_j(j, coef, n_int, pwl_func, knots, tau_f, distribution, alpha):
    if j == 1:
        coef_sum = 0
    else:
        coef_sum = coef[1:j]

    def ufun_j(t, x):
        _coef = coef.copy()
        _coef[j] = x
        return pwl_func(t=t, coef=_coef, knots=knots)

    def fn1(t, cj):
        out = (
            tau_f(t)
            / (2 * np.pi)
            * np.exp(-ufun_j(t, cj) ** 2 / 2)
            * np.exp(-np.sum([coef_sum, cj])) ** 2
            / (2 * tau_f(t) ** 2)
        )
        return out

    def fn2(t, cj):
        out = (
            np.sum([coef_sum, cj])
            / np.sqrt(2 * np.pi)
            * np.exp(-ufun_j(t, cj) ** 2 / 2)
            * distribution.cdf(np.sum([coef_sum, cj]) / tau_f(t))
        )
        return out

    def fn3(t, cj):
        out = (
            np.sum([coef_sum, cj])
            / np.sqrt(2 * np.pi)
            * np.exp(-ufun_j(t, cj) ** 2 / 2)
            * distribution.cdf(-np.sum([coef_sum, cj]) / tau_f(t))
        )
        return out

    def _root_func(x):
        integr1, _ = integrate.quad(partial(fn1, cj=x), a=knots[j], b=knots[j + 1])

        if j % 2 == 0:
            integr2, _ = integrate.quad(partial(fn2, cj=x), a=knots[j], b=knots[j + 1])
            integr3 = 0
            res = distribution.cdf(-pwl_func(knots[j], coef=coef, knots=knots))
        else:
            integr2 = 0
            integr3, _ = integrate.quad(partial(fn3, cj=x), a=knots[j], b=knots[j + 1])
            res = distribution.cdf(-ufun_j(knots[j + 1], x))

        integral_part = integr1 + integr2 + integr3 - (alpha / (2 * n_int))
        res += integral_part
        return res

    return _root_func


def _get_root_func(tau_integrated, n_int, alpha, distribution):
    def _func(x):
        return (
            (1 - distribution.cdf(x))
            + tau_integrated * np.exp(-(x**2) / 2) / (2 * np.pi)
            - (alpha / (2 * n_int))
        )

    return _func


def _get_piecewise_linear_function(n_int):
    def pwl_func(t, coef, knots):
        out = coef[0]
        for i in range(1, n_int + 1):
            out += coef[i] * np.maximum(t - knots[i], 0)
        return out

    return pwl_func


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _any_none(*vars):
    return bool([var for var in vars if var is None])


def _cov_to_corr(cov):
    standard_errors = np.sqrt(np.diag(cov))
    corr = cov.copy()
    corr /= standard_errors.reshape(1, -1)
    corr /= standard_errors.reshape(-1, 1)
    return corr
