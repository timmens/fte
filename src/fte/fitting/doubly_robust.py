from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.base import is_classifier
from sklearn.base import is_regressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures


class TreatmentData(NamedTuple):
    x: np.ndarray
    x_other: np.ndarray  # feature matrix of the other sample
    y: np.ndarray
    t: np.ndarray
    t_bool: np.ndarray


def fit_func_on_scalar_doubly_robust(
    data=None,
    *,
    x=None,
    y=None,
    t=None,
    fit_intercept=True,
    mean_learner="RidgeCV",
    mean_learner_kwargs=None,
    ps_learner="LogisticRegression",
    ps_learner_kwargs=None,
    tol=1e-6,
    seed=None,
):
    # ==================================================================================
    # Prepare data inputs
    # ==================================================================================

    if data is not None:
        y = data.y
        x = data.x
        t = data.treatment_status

    index = None

    if isinstance(x, pd.DataFrame):
        x = x.to_numpy()

    if isinstance(y, pd.DataFrame):
        index = y.columns.astype(np.int64).rename("time")
        y = y.to_numpy()

    if isinstance(t, pd.DataFrame):
        t = t.to_numpy()

    t = t.flatten()

    # ==================================================================================
    # Default kwargs
    if ps_learner_kwargs is None:
        ps_learner_kwargs = {}

    if mean_learner_kwargs is None:
        mean_learner_kwargs = {}

    # ==================================================================================
    # Fit nuisance functions
    # ==================================================================================
    feature_transformer = PolynomialFeatures(degree=1, include_bias=False)

    # ==================================================================================
    # Sample splitting

    x_0, x_1, y_0, y_1, t_0, t_1 = train_test_split(
        x, y, t, test_size=0.5, random_state=seed
    )

    sample0 = TreatmentData(
        x=x_0, x_other=x_1, y=y_0, t=t_0, t_bool=t_0.astype(bool).flatten()
    )
    sample1 = TreatmentData(
        x=x_1, x_other=x_0, y=y_1, t=t_1, t_bool=t_1.astype(bool).flatten()
    )

    # ==================================================================================
    # Propensity score computation

    pipe = _get_propensity_score_pipe(
        learner=ps_learner,
        learner_kwargs=ps_learner_kwargs,
        feature_transform=feature_transformer,
        fit_intercept=fit_intercept,
    )

    ps = [
        pipe.fit(sample.x, sample.t).predict_proba(sample.x_other)[:, [1]]
        for sample in (sample0, sample1)
    ]
    ps = [a.clip(min=tol, max=1.0 - tol) for a in ps]

    # ==================================================================================
    # Conditional Mean Computation

    pipe = _get_conditional_mean_pipe(
        learner=mean_learner,
        learner_kwargs=mean_learner_kwargs,
        feature_transform=feature_transformer,
        fit_intercept=fit_intercept,
    )

    mean_1 = [
        pipe.fit(sample.x[sample.t_bool], sample.y[sample.t_bool]).predict(
            sample.x_other
        )
        for sample in (sample0, sample1)
    ]

    mean_0 = [
        pipe.fit(sample.x[~sample.t_bool], sample.y[~sample.t_bool]).predict(
            sample.x_other
        )
        for sample in (sample0, sample1)
    ]

    # ==================================================================================
    # Compute treatment effect and covariance kernel
    # ==================================================================================

    effect_0 = _compute_treatment_effect(
        sample0.y, sample0.t, mean_1[1], mean_0[1], ps[1]
    )
    effect_1 = _compute_treatment_effect(
        sample1.y, sample1.t, mean_1[0], mean_0[0], ps[0]
    )

    kernel_0 = _compute_treatment_effect_kernel(
        sample0.y, sample0.t, mean_1[1], mean_0[1], ps[1]
    )
    kernel_1 = _compute_treatment_effect_kernel(
        sample1.y, sample1.t, mean_1[0], mean_0[0], ps[0]
    )

    effect = 0.5 * (effect_0 + effect_1)
    kernel = 0.5 * (kernel_0 + kernel_1)

    # ==================================================================================
    # Results processing
    # ==================================================================================

    if index is not None:
        effect = pd.DataFrame(effect, columns=["value"], index=index)
        kernel = pd.DataFrame(kernel, columns=index, index=index)

    out = {"treatment_effect": effect, "cov": kernel, "ps": ps, "n_samples": len(y)}
    return out


def _compute_treatment_effect(y, t, mean_1, mean_0, ps):
    t = t.reshape(-1, 1)

    treatment_effect = np.mean(t * (y - mean_1) / ps + mean_1, axis=0) - np.mean(
        (1 - t) * (y - mean_0) / (1 - ps) + mean_0, axis=0
    )
    return treatment_effect


def _compute_treatment_effect_kernel(y, t, mean_1, mean_0, ps):
    t = t.reshape(-1, 1)

    naive_effect = mean_1 - mean_0
    naive_effect = naive_effect - naive_effect.mean(axis=0)

    outer = _outer_product_along_first_dim(naive_effect)
    kernel_a = outer.mean(axis=0)

    adjustment = t / ps * (y - mean_1) + (1 - t) / (1 - ps) * (y - mean_0)
    adjustment = adjustment - adjustment.mean(axis=0)
    outer = _outer_product_along_first_dim(adjustment)
    kernel_b = outer.mean(axis=0)

    kernel = kernel_a + kernel_b
    return kernel


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


def _get_propensity_score_pipe(
    learner, learner_kwargs, feature_transform, fit_intercept
):
    implemented_learners = {
        "LogisticRegression": LogisticRegression,
    }

    default_kwargs = {
        "LogisticRegression": {
            "penalty": "l1",
            "C": 1e-6,
            "max_iter": 10_000,
            "solver": "liblinear",
        },
    }

    kwargs = default_kwargs.get(learner, {})
    learner_kwargs = {**kwargs, **learner_kwargs}

    if isinstance(learner, str):
        learner = implemented_learners.get(learner, None)

    if not is_classifier(learner):
        msg = (
            f"learner {learner} is not a sklearn classifier. Either use a valid ",
            "sklearn classifier or pass a name from {implemented_models.keys()}.",
        )
        raise ValueError(msg)

    pipe = Pipeline(
        steps=[
            ("feature_transform", feature_transform),
            ("learner", learner(fit_intercept=fit_intercept, **learner_kwargs)),
        ]
    )
    return pipe


def _get_conditional_mean_pipe(
    learner, learner_kwargs, feature_transform, fit_intercept
):
    implemented_learners = {
        "LinearRegression": LinearRegression,
        "RidgeCV": RidgeCV,
    }

    default_kwargs = {"RidgeCV": {"alphas": [1e-3, 1e-2, 1e-1, 1]}}

    kwargs = default_kwargs.get(learner, {})
    learner_kwargs = {**kwargs, **learner_kwargs}

    if isinstance(learner, str):
        learner = implemented_learners.get(learner, None)

    if not is_regressor(learner):
        msg = (
            f"learner {learner} is not a sklearn regressor. Either use a valid ",
            "sklearn regressor or pass a name from {implemented_models.keys()}.",
        )
        raise ValueError(msg)

    pipe = Pipeline(
        steps=[
            ("feature_transform", feature_transform),
            ("learner", learner(fit_intercept=fit_intercept, **learner_kwargs)),
        ]
    )
    return pipe
