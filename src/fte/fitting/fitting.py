import inspect
import warnings
from functools import partial
from functools import wraps

import numpy as np
import pandas as pd
from fte.fitting.doubly_robust import fit_func_on_scalar_doubly_robust


def get_fitter(fitter, fitter_kwargs=None):
    if fitter_kwargs is None:
        fitter_kwargs = {}

    IMPLEMENTED_FITTER = {  # noqa: N806
        "func_on_scalar": _fit_func_on_scalar,
        "func_on_scalar_doubly_robust": fit_func_on_scalar_doubly_robust,
    }

    if isinstance(fitter, str) and fitter in IMPLEMENTED_FITTER:
        _fitter = IMPLEMENTED_FITTER[fitter]
        _fitter_name = fitter
    elif callable(fitter):
        _fitter = fitter
        _fitter_name = getattr(fitter, "__name__", "your fitter")
    else:
        raise ValueError(
            f"Invalid fitter: {fitter}. Must be one of {list(IMPLEMENTED_FITTER)} or a "
            "callable."
        )

    args = set(inspect.signature(_fitter).parameters)
    mandatory_args = {"data"}

    problematic = mandatory_args - args
    if problematic:
        raise ValueError(
            f"The following mandatory arguments are missing in {_fitter_name}: "
            f"{problematic}"
        )

    valid_options = args

    reduced = {key: val for key, val in fitter_kwargs.items() if key in valid_options}
    ignored = {
        key: val for key, val in fitter_kwargs.items() if key not in valid_options
    }

    if ignored:
        warnings.warn(
            "The following options were ignored because they are not compatible with "
            f"{_fitter_name}:\n\n {ignored}"
        )

    fitter = wraps(_fitter)(partial(_fitter, **reduced))
    return fitter


def _fit_func_on_scalar(data=None, *, x=None, y=None, fit_intercept=True):
    if data is not None:
        y = data.y
        x = data.x

    columns = None
    index = None

    if isinstance(x, pd.DataFrame):
        columns = x.columns
        x = x.values

    if fit_intercept:
        x = np.column_stack((np.ones(len(x)), x))

    if isinstance(y, pd.DataFrame):
        index = y.columns.astype(np.int64).rename("time")
        y = y.values

    coef, *_ = np.linalg.lstsq(x, y, rcond=None)
    coef = coef.T

    intercept = coef[:, 0]
    slopes = coef[:, 1:]

    if columns is not None or index is not None:
        slopes = pd.DataFrame(slopes, columns=columns, index=index)

    out = {"intercept": intercept, "slopes": slopes, "data": data}
    return out
