import numpy as np


def cov_from_residuals(residuals, x=None, coef_id=None, cov_type="homoskedastic"):
    if x is None:
        cov = residuals.T @ residuals / len(residuals)
    else:
        n_points = residuals.shape[1]
        cov_operator = get_covariance_operator(residuals, x=x, cov_type=cov_type)

        cov = np.empty((n_points, n_points))
        for j in range(n_points):
            for k in range(n_points):
                cov[j, k] = cov_operator(j, k)[coef_id, coef_id]

    return cov


def get_covariance_operator(residuals, x, cov_type="homoskedastic"):
    r"""Construct covariance operator.

    # Note =============================================================================

    For residuals {r_i(t): i=1,...,n; t=0,...,1} = r(t) and data X = {x_i: i=1,...,n}
    this function computes covariance operators of the form

        cov(s, t) = (X'X)^{-1} X' E[r(s)r(t)'] X (X'X)^{-1}.

    For cov_type = 'homoskedastic' we assume that E[r_i(s)r_i(t)] = \sigma(s, t) for all
    i=1,...,n; which allows us to do the standard simplification

        cov(s, t) = E[r_i(s)r_i(t)] (X'X)^{-1}.

    # ==================================================================================

    Args:
        x (np.ndarray): Data matrix, of shape (n_samples, n_features).
        residuals (np.ndarray): Model residuals, of shape (n_samples, n_points).
        cov_type (str): The type of robust sandwich estimator to use. Default
            'homoskedastic', which assumes homoskedasticity. Must be in {None, 'HC0',
            'HC1'}. For reference see section 4.16 in Econometrics by Bruce Hansen.

    Returns:
        callable: The covariance operator, callable at points (j, k) for
            j, k = 1, ..., n_points. At a point (j, k) the operator is matrix-valued
            with shape (n_features, n_features).

    """
    if cov_type not in {"homoskedastic", "HC0", "HC1"}:
        raise ValueError("cov_type must be in {'homoskedastic', 'HC0', 'HC1'}")

    if residuals.ndim != 2 or x.ndim != 2:
        raise ValueError("residuals and x need to be a 2-dimensional array.")

    if len(residuals) != len(x):
        raise ValueError("residuals and x need to have the same length.")

    information_matrix = np.linalg.inv(np.matmul(x.T, x))

    if cov_type == "homoskedastic":
        error_kernel = residuals.T @ residuals / len(residuals)
        _operator = _get_operator_homoskedasticity(error_kernel, information_matrix)
    else:
        residuals_outer = _outer_product_along_first_dim(residuals)
        _operator = _get_operator_heteroskedasticity(
            residuals_outer, information_matrix, x=x, cov_type=cov_type
        )

    return _operator


def _get_operator_homoskedasticity(error_kernel, information_matrix):
    def _operator(j, k):
        return error_kernel[j, k] * information_matrix

    return _operator


def _get_operator_heteroskedasticity(
    residuals_outer, information_matrix, *, x, cov_type
):
    n_samples, n_features = x.shape
    _, _, n_time_points = residuals_outer.shape

    pseudo_inv = np.matmul(information_matrix, x.T)

    triu_idx = np.triu_indices(n_time_points)
    n_triu = len(triu_idx[0])

    operator_triu = np.empty((n_triu, n_features, n_features))

    for i in range(n_triu):
        diag = np.diag(residuals_outer[:, triu_idx[0][i], triu_idx[1][i]])
        operator_triu[i, ...] = np.linalg.multi_dot((pseudo_inv, diag, pseudo_inv.T))

    idx_lookup = np.empty((n_time_points, n_time_points), dtype=np.int64)
    idx_lookup[triu_idx[0], triu_idx[1]] = np.arange(n_triu, dtype=np.int64)

    if cov_type == "HC1":
        operator_triu *= (n_samples - n_features) / n_samples

    def _operator(j, k):
        (j, k) = (
            (k, j) if j > k else (j, k)
        )  # switch indices, because we only store one case, and the operator is
        # symmetric.
        out = operator_triu[idx_lookup[j, k]]
        return out

    return _operator


def _outer_product_along_first_dim(arr):
    """Compute outer product along first dimension.

    Args:
        arr (np.ndarray): Array of shape (s1, s2).

    Returns:
        np.ndarray: Array of shape (s1, s2, s2).

    """
    products = [np.outer(a, a) for a in arr]
    out = np.array(products)
    return out
