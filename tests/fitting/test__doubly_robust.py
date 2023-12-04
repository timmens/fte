import jax.numpy as jnp
from fte.fitting._doubly_robust import _aipwe_formula
from numpy.testing import assert_array_equal


def test_aipwe_formula():
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

    got = _aipwe_formula(y=y, d=d, ce_0=ce_0, ce_1=ce_1, ps=ps)

    exp = jnp.ones(3) - 0.5 / 0.7
    assert_array_equal(got["estimate"], exp)
