import inspect
import warnings

DEFAULT_NUISANCE_FUNCTIONS_KWARGS = {
    "x": None,
    "y": None,
    "d": None,
    "conditional_expectation": None,
    "propensity_score": None,
}


def compute_nuisance_functions(name, **kwargs):
    """Compute nuisance functions (conditional expectation and propensity score).

    Args:
        name (str or callable): The fitter to use.
        **kwargs: Keyword arguments to pass to the fitter.

    Returns:
        dict: A dictionary with the nuisance functions. Has entries
        "conditional_expectation" and "propensity_score".

    """
    updated_fitter_kwargs = _update_dictionary(
        DEFAULT_NUISANCE_FUNCTIONS_KWARGS, kwargs,
    )

    implemented_fitters = {
        "oracle": _oracle,
    }

    if isinstance(name, str) and name in implemented_fitters:
        _fitter = implemented_fitters[name]
        _fitter_name = name
    elif callable(name):
        _fitter = name
        _fitter_name = getattr(name, "__name__", "your fitter")
    else:
        raise ValueError(
            f"Invalid fitter: {name}. Must be one of {list(implemented_fitters)} or a "
            "callable.",
        )

    args = set(inspect.signature(_fitter).parameters)
    mandatory_args = {"x", "y", "d"}

    problematic = mandatory_args - args
    if problematic:
        raise ValueError(
            f"The following mandatory arguments are missing in {_fitter_name}: "
            f"{problematic}",
        )

    valid_options = args

    reduced = {
        key: val for key, val in updated_fitter_kwargs.items() if key in valid_options
    }
    ignored = {
        key: val
        for key, val in updated_fitter_kwargs.items()
        if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible with "
            f"{_fitter_name}:\n\n {ignored}",
            stacklevel=1,
        )

    return _fitter(**reduced)


# ======================================================================================
# Nuisance function calculators
# ======================================================================================


def _oracle(x, y, d, conditional_expectation, propensity_score):  # noqa: ARG001
    return {
        "conditional_expectation": conditional_expectation,
        "propensity_score": propensity_score,
    }


# ======================================================================================
# Auxiliary functions
# ======================================================================================


def _update_dictionary(d, update):
    if update is None:
        update = {}
    return d | update
