import inspect
import warnings
from functools import partial
from functools import wraps
from typing import Dict
from typing import NamedTuple
from typing import Union

import numpy as np
from fte.simulation.processes import get_kernel
from fte.simulation.processes import simulate_gaussian_process
from fte.utilities import set_attribute
from scipy.special import expit


class SimulatedData(NamedTuple):
    """Container for simulated data."""

    y: np.ndarray
    x: np.ndarray
    params: Dict[str, Union[Dict[str, np.ndarray], np.ndarray]]
    has_intercept: bool
    is_causal: bool = False
    treatment_status: np.ndarray = None
    propensity_score: np.ndarray = None


def get_data_simulator(
    model_func="linear",
    model_func_kwargs=None,
    simulate_params="default_regression",
    simulate_features="uniform",
    kernel="RBF",
    kernel_kwargs=None,
    heteroskedastic=False,
    error_scale=1.0,
    has_intercept=True,
    template="default_template",
):
    """Return a data simulation function.

    Args:
        model_func (str, callable): Default. "linear".
        model_func_kwargs (dict): Default None.
        simulate_params (str, callable): Default "default_regression".
        simulate_features (str, callable): Default "uniform".
        kernel (str, callable): Default "RBF".
        kernel_kwargs (dict): Default None.
        heteroskedastic (bool): Default False.
        has_intercept (bool): Default True.
        template (str, callable): Default "default_template".

    Returns:
        callable: The data simulation function. Depends on arguments 'n_samples',
            'n_periods', 'n_params' and 'seed'; returns object SimulatedData.

    """
    # Get the template simulator
    # ==================================================================================

    IMPLEMENTED_SIMULATOR_TEMPLATES = {  # noqa: N806
        "default_template": _data_simulator_template,
    }

    if isinstance(template, str) and template in IMPLEMENTED_SIMULATOR_TEMPLATES:
        _simulator_template = IMPLEMENTED_SIMULATOR_TEMPLATES[template]
        _simulator_template_name = template
    elif callable(template):
        _simulator_template = template
        try:
            _simulator_template_name = template.__name__
        except AttributeError as error:
            msg = "If data_process is callable it needs to have a __name__ attribute."
            raise ValueError(msg) from error
    else:
        raise ValueError(
            f"Invalid data_process: {template}. Must be one of "
            f"{list(IMPLEMENTED_SIMULATOR_TEMPLATES)} or a callable."
        )

    # Get model building blocks
    # ==================================================================================

    simulate_params = get_params_simulator(simulate_params)
    simulate_features = get_feature_simulator(simulate_features)
    kernel = get_kernel(kernel, kernel_kwargs)
    model_func = get_model_func(model_func, model_func_kwargs)

    simulator_kwargs = {
        "simulate_params": simulate_params,
        "simulate_features": simulate_features,
        "kernel": kernel,
        "model_func": model_func,
        "has_intercept": has_intercept,
        "heteroskedastic": heteroskedastic,
        "error_scale": error_scale,
    }

    # Consolidate arguments
    # ==================================================================================

    args = set(inspect.signature(_simulator_template).parameters)
    mandatory_args = {
        "n_samples",
        "n_periods",
        "n_params",
        "seed",
    }

    problematic = mandatory_args - args
    if problematic:
        raise ValueError(
            "The following mandatory arguments are missing in "
            f"{_simulator_template_name}: {problematic}."
        )

    valid_options = args

    reduced = {
        key: val for key, val in simulator_kwargs.items() if key in valid_options
    }
    ignored = {
        key: val for key, val in simulator_kwargs.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible "
            f"with {_simulator_template_name}:\n\n {ignored}"
        )

    # Consolidate output
    # ==================================================================================
    template = partial(_simulator_template, **reduced)
    template.is_causal = model_func.is_causal
    return template


def _data_simulator_template(
    n_samples,
    n_periods,
    n_params,
    simulate_params,
    simulate_features,
    kernel,
    model_func,
    has_intercept,
    heteroskedastic,
    error_scale,
    seed=None,
):
    # Simulation
    # ==================================================================================
    rng = np.random.default_rng(seed)

    features = simulate_features(n_samples=n_samples, n_params=n_params, rng=rng)

    params = simulate_params(
        n_periods=n_periods,
        n_params=n_params,
        has_intercept=has_intercept,
    )

    error = simulate_gaussian_process(
        n_samples=n_samples,
        n_periods=n_periods,
        kernel=kernel,
        rng=rng,
        scale=error_scale,
    )

    if heteroskedastic:
        error = make_error_heteroskedastic(
            error=error,
            features=features,
        )

    if model_func.is_causal:
        treatment_status, propensity_score = simulate_treatment_status(
            features=features, coef=params["treatment_status"], rng=rng
        )
        treatment_effect = compute_treatment_effect(
            features=features,
            treatment_status=treatment_status,
            coef=params["treatment_effect"],
        )
        params["model_func"]["treatment_effect"] = treatment_effect

    y = model_func(coef=params["model_func"], features=features, error=error)

    # Output
    # ==================================================================================
    data = SimulatedData(
        y=y.T,
        x=features,
        params=params,
        has_intercept=has_intercept,
    )
    if model_func.is_causal:
        data = data._replace(
            **{
                "is_causal": True,
                "treatment_status": treatment_effect,
                "propensity_score": propensity_score,
            }
        )
    return data


# ======================================================================================
# Causal effects
# ======================================================================================


def simulate_treatment_status(features, coef, rng):
    z = _linear_model_func(coef=coef, features=features, error=0)
    propensity_score = expit(z)
    treatment_status = rng.binomial(n=1, p=propensity_score)
    return treatment_status, propensity_score


def compute_treatment_effect(features, treatment_status, coef):
    effect = np.zeros(len(features))
    effect[treatment_status.astype(bool)] = coef
    return effect


# ======================================================================================
# Heteroskedasticity
# ======================================================================================


def make_error_heteroskedastic(error, features):
    """Scale errors using the values of the first feature dimension.

    We map the first feature dimension into the space [0.5, 1.5], where the largest
    value is mapped to 1.5 and the smallest to 0.5, respectively.

    """
    scale = features[:, 0].flatten()
    scale = scale / (scale.max() - scale.min()) + 0.5
    return scale * error


# ======================================================================================
# Model function
# ======================================================================================


def get_model_func(model_func, model_func_kwargs):
    if model_func_kwargs is None:
        model_func_kwargs = {}

    MODELS = {  # noqa: N806
        "linear": _linear_model_func,
        "linear_causal": _linear_causal_model_func,
    }

    if model_func in MODELS:
        model_func = MODELS[model_func]

    if callable(model_func) and hasattr(model_func, "is_causal"):
        out = wraps(model_func)(partial(model_func, **model_func_kwargs))
    elif callable(model_func):
        msg = (
            "model_func needs to have the attribute 'is_causal', determining whether "
            "the model is causal or not."
        )
        raise ValueError(msg)
    else:
        msg = f"model_func {model_func} is not in implemented models: {MODELS.keys()}."
        raise ValueError(msg)

    return out


@set_attribute("is_causal", False)
def _linear_model_func(coef, features, error):
    """Linear model.

    This function implements the linear model:

        y = intercept + features * slopes + error

    """
    outcome = coef["slopes"] @ features.T + error
    if "intercept" in coef:
        outcome += coef["intercept"]
    return outcome


@set_attribute("is_causal", True)
def _linear_causal_model_func(coef, features, error):
    """Linear model with potential outcomes.

    This function implements the linear potential outcome model:

        y(0) = intercept + features * slopes + error
        y(1) = y(0) + effect

    """
    outcome = _linear_model_func(coef, features, error)
    outcome += coef["treatment_effect"]
    return outcome


# ======================================================================================
# Params simulator
# ======================================================================================


def get_params_simulator(params_simulator):
    PARAMS_SIMULATORS = {  # noqa: N806
        "default_regression": _default_regression_params_simulator,
        "increasing_sin": _increasing_sin_regression_params_simulator,
        "causal": _causal_params_coef_simulator,
    }

    if isinstance(params_simulator, str) and params_simulator in PARAMS_SIMULATORS:
        simulator = PARAMS_SIMULATORS[params_simulator]
    elif callable(params_simulator):
        simulator = params_simulator
    else:
        raise ValueError(
            f"params_simulator must be callable or in {PARAMS_SIMULATORS.keys()}"
        )
    return simulator


def _default_regression_params_simulator(
    n_params, has_intercept, n_periods=None, grid=None
):
    if grid is None and n_periods is None:
        raise ValueError("One of grid and n_periods needs to be non-None.")
    if grid is None:
        grid = np.linspace(0, 1, num=n_periods).reshape(-1, 1)

    slopes = np.full((n_periods, n_params), 2)
    model_func_params = {"slopes": slopes}
    if has_intercept:
        model_func_params["intercept"] = -np.ones((n_periods, 1))

    params = {"model_func": model_func_params}
    return params


def _increasing_sin_regression_params_simulator(
    n_params, has_intercept, n_periods=None, grid=None
):
    if grid is None and n_periods is None:
        raise ValueError("One of grid and n_periods needs to be non-None.")
    if grid is None:
        grid = np.linspace(0, 1, num=n_periods).reshape(-1, 1)

    slopes = np.tile(np.arange(1, n_params + 1), n_periods).reshape(n_periods, n_params)

    model_func_params = {"slopes": slopes * np.sin(grid)}
    if has_intercept:
        model_func_params["intercept"] = -np.ones((n_periods, 1))

    params = {"model_func": model_func_params}
    return params


def _causal_params_coef_simulator(n_params, has_intercept, n_periods=None, grid=None):
    params = _default_regression_params_simulator(
        n_params=n_params,
        n_periods=n_periods,
        has_intercept=has_intercept,
        grid=grid,
    )
    params["treatment_status"] = {"slopes": -params["model_func"]["slopes"][1, :]}
    params["treatment_effect"] = 1
    return params


# ======================================================================================
# Feature simulator
# ======================================================================================


def get_feature_simulator(feature_simulator):
    FEATURE_SIMULATORS = {  # noqa: N806
        "uniform": _uniform_feature_simulator,
        "normal": _normal_feature_simulator,
    }

    if isinstance(feature_simulator, str) and feature_simulator in FEATURE_SIMULATORS:
        simulator = FEATURE_SIMULATORS[feature_simulator]
    elif callable(feature_simulator):
        simulator = feature_simulator
    else:
        raise ValueError(
            f"feature_simulator must be callable or in {FEATURE_SIMULATORS.keys()}"
        )
    return simulator


def _uniform_feature_simulator(n_samples, n_params, rng):
    x = rng.uniform(size=(n_samples, n_params))
    return x


def _normal_feature_simulator(n_samples, n_params, rng):
    x = rng.normal(size=(n_samples, n_params))
    return x
