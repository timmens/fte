import inspect
import warnings
from functools import partial

import numpy as np
import pandas as pd

from fte.fitting.doubly_robust import fit_func_on_scalar_doubly_robust

DEFAULT_FITTER_KWARGS = {
    "fit_intercept": True,
    "mean_learner": "RidgeCV",
    "mean_learner_kwargs": {},
    "ps_learner": "LogisticRegression",
    "ps_learner_kwargs": {},
    "tol": 1e-6,
    "seed": None,
}


IMPLEMENTED_FITTER = {
    "func_on_scalar_doubly_robust": fit_func_on_scalar_doubly_robust,
}


def get_fitter(fitter, fitter_kwargs=None):
    """Get a function that fits a function on scalar data.

    Args:
        fitter (str or callable): The fitter to use.
        fitter_kwargs (dict): Keyword arguments to pass to the fitter.

    Returns:
        callable: The fitter function. It always

    """
    updated_fitter_kwargs = _update_dictionary(DEFAULT_FITTER_KWARGS, fitter_kwargs)

    if isinstance(fitter, str) and fitter in IMPLEMENTED_FITTER:
        _fitter = IMPLEMENTED_FITTER[fitter]
        _fitter_name = fitter
    elif callable(fitter):
        _fitter = fitter
        _fitter_name = getattr(fitter, "__name__", "your fitter")
    else:
        raise ValueError(
            f"Invalid fitter: {fitter}. Must be one of {list(IMPLEMENTED_FITTER)} or a "
            "callable.",
        )

    args = set(inspect.signature(_fitter).parameters)
    mandatory_args = {"data"}

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

    return partial(_fitter_template, fitter=_fitter, fitter_kwargs=reduced)


def _fitter_template(
    fitter,
    fitter_kwargs,
    data=None,
    x=None,
    y=None,
    t=None,
):
    # Prepare data inputs
    # ==================================================================================
    if data is not None:
        y = data.y
        x = data.x
        t = data.treatment_status

    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.DataFrame):
        index = y.columns.astype(np.int64).rename("time")
        y = y.to_numpy()
    else:
        index = None

    if isinstance(t, pd.DataFrame):
        t = t.to_numpy()

    t = t.flatten()

    # Call fitter
    # ==================================================================================
    res = fitter(x=x, y=y, t=t, **fitter_kwargs)

    # Results processing
    # ==================================================================================
    if index is not None:
        res["effect"] = pd.DataFrame(res["effect"], columns=["value"], index=index)
        if "kernel" in res:
            res["kernel"] = pd.DataFrame(res["kernel"], columns=index, index=index)

    return res | {"n_samples": len(y)}


def _update_dictionary(d, update):
    if update is None:
        update = {}
    return d | update


# ======================================================================================
# Non-treatment-effect fitters
# ======================================================================================


def _fit_func_on_scalar(data=None, *, x=None, y=None, fit_intercept=True):
    """Fit a function on scalar data."""
    if data is not None:
        y = data.y
        x = data.x

    columns = None
    index = None

    if isinstance(x, pd.DataFrame):
        columns = x.columns
        x = x.to_numpy()

    if fit_intercept:
        x = np.column_stack((np.ones(len(x)), x))

    if isinstance(y, pd.DataFrame):
        index = y.columns.astype(np.int64).rename("time")
        y = y.to_numpy()

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = coef.T

    intercept = coef[:, 0]
    slopes = coef[:, 1:]

    if columns is not None or index is not None:
        slopes = pd.DataFrame(slopes, columns=columns, index=index)

    return {"intercept": intercept, "slopes": slopes, "data": data}
