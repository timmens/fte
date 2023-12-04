from jax import Array
from jax.typing import ArrayLike


def doubly_robust_fitter_template(
    x,
    y,
    d,
    conditional_expectation,
    propensity_score,
):
    """Doublye robust fitter template.

    Args:
        x (jnp.ndarray): Feature array, shape (n_samples, n_features).
        y (jnp.ndarray): Response array, shape (n_samples, n_time_points).
        d (jnp.ndarray): Treatment status array, shape (n_samples, ).
        conditional_expectation (callable): Function that computes the conditional
            expectation of y given x and d. Note that this function returns an array of
            shape (len(x), n_time_points).
        propensity_score (callable): Function that computes the propensity score given
            x. Note that this function returns an array of shape (len(x), ).

    Returns:
        dict:
            - estimate (jnp.ndarray): Array of shape (n_time_points, ).
            - variance (jnp.ndarray): Array of shape (n_time_points, ).

    """
    ps = propensity_score(x)

    ce_0 = conditional_expectation(x=x, d=0)
    ce_1 = conditional_expectation(x=x, d=1)

    return _aipwe_formula(
        y=y,
        d=d,
        ce_0=ce_0,
        ce_1=ce_1,
        ps=ps,
    )


def _aipwe_formula(
    y: ArrayLike,
    d: ArrayLike,
    ce_0: ArrayLike,
    ce_1: ArrayLike,
    ps: ArrayLike,
) -> Array:
    """Formula of the augmented inverse probability weighted estimator.

    Args:
        y (jnp.ndarray): Response array, shape (n_samples, n_time_points).
        ce_0 (jnp.ndarray): Conditional expectation of y given x and d=0, shape
            (n_samples, n_time_points).
        ce_1 (jnp.ndarray): Conditional expectation of y given x and d=1, shape
            (n_samples, n_time_points).
        d (jnp.ndarray): Treatment status array, shape (n_samples, ).
        ps (jnp.ndarray): Propensity score, shape (n_samples, ).

    Returns:
        dict:
            - estimate (jnp.ndarray): Array of shape (n_time_points, ).
            - variance (jnp.ndarray): Array of shape (n_time_points, ).

    """
    _d = d.reshape(-1, 1)
    _ps = ps.reshape(-1, 1)

    # Compute components of sum
    # ==================================================================================
    components = ce_1 - ce_0 + _d * (y - ce_1) / _ps - (1 - _d) * (y - ce_0) / (1 - _ps)

    # Compute mean over individuals
    # ==================================================================================
    estimate = components.mean(axis=0)
    variance = components.var(axis=0)

    return {
        "estimate": estimate,
        "variance": variance,
    }
