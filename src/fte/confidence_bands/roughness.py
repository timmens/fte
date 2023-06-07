import numpy as np
from scipy.interpolate import RectBivariateSpline


def get_roughness_func(
    grid: np.ndarray,
    cov: np.ndarray,
    interpolator: str = "RectBivariateSpline",
    info: bool = False,
):
    """Get the parameter roughness function.

    Args:
        grid (np.ndarray): Time grid. Default None.
        cov (np.ndarray): The estimated covariance matrix.
        interpolator (str or callable): The interpolator which is used to smooth the
            covariance matrix. Implemented are {"RectBivariateSpline"}. Default is
            RectBivariateSpline.
        info (bool): Add extra information to output function.

    Returns:
        callable: The parameter roughness function.

    """
    # Validate inputs
    # ==================================================================================
    built_in_interpolator = {
        "RectBivariateSpline": RectBivariateSpline,
    }

    if isinstance(interpolator, str) and interpolator in built_in_interpolator:
        interpolator = built_in_interpolator[interpolator]
        _name = interpolator
    elif callable(interpolator):
        _name = interpolator.__name__
    else:
        msg = f"Interpolator must be callable or in {set(built_in_interpolator)}."
        raise ValueError(msg)

    # Compute roughness function
    # ==================================================================================
    corr = _cov_to_corr(cov)
    smooth_corr = interpolator(grid, grid, corr)
    smooth_corr_deriv = smooth_corr.partial_derivative(dx=1, dy=1)

    @np.vectorize
    def _roughness(t):
        return np.sqrt(smooth_corr_deriv(t, t))

    if info:
        _roughness.info = {"smooth_corr": smooth_corr, "interpolator_name": _name}

    return _roughness


def _cov_to_corr(cov: np.ndarray):
    standard_errors = np.sqrt(np.diag(cov))
    corr = cov.copy()
    corr /= standard_errors.reshape(1, -1)
    corr /= standard_errors.reshape(-1, 1)
    return corr
