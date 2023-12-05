import jax.numpy as jnp
import pytest
from fte.fitting.aipwe import _aipwe_formula, get_aipwe_estimator
from numpy.testing import assert_array_equal


@pytest.fixture()
def test_data():
    y = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    d = jnp.array([0, 1])
    ce_0 = jnp.array(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ],
    )
    ce_1 = jnp.array(
        [
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0],
        ],
    )
    ps = jnp.array([0.3, 0.7])
    return {"y": y, "d": d, "ce_0": ce_0, "ce_1": ce_1, "ps": ps}


def test_aipwe_formula(test_data):
    got = _aipwe_formula(**test_data)
    exp = jnp.ones(3) - 0.5 / 0.7
    assert_array_equal(got["estimate"], exp)


def test_get_aipwe_estimator(test_data):
    def conditional_expectation(x, d):  # noqa: ARG001
        return test_data["ce_0" if d == 0 else "ce_1"]

    def propensity_score(x):  # noqa: ARG001
        return test_data["ps"]

    got_aipwe_estimator = get_aipwe_estimator(
        name="oracle",
        conditional_expectation=conditional_expectation,
        propensity_score=propensity_score,
    )

    got = got_aipwe_estimator(
        x=None,
        y=test_data["y"],
        d=test_data["d"],
    )
    exp = jnp.ones(3) - 0.5 / 0.7
    assert_array_equal(got["estimate"], exp)
