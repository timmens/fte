import pytest
from fte.montypy.generic import generic_simulation_study, get_evaluator


@pytest.mark.parametrize("show_progress", [True, False])
@pytest.mark.parametrize("n_cores", [1, 2])
@pytest.mark.parametrize("evaluator", ["loop", "joblib"])
def test_get_evaluator(show_progress, n_cores, evaluator):
    evaluator = get_evaluator(
        evaluator,
        func=lambda x: x,
        n_cores=n_cores,
        show_progress=show_progress,
    )
    params_list = [{"x": k} for k in [1, 2, 3]]
    got = evaluator(params_list=params_list)
    assert [1, 2, 3] == got


def test_generic_simulation_study():
    def _results_processor(_list):
        return [x**2 for x in _list]

    params_list = [{"x": k} for k in [1, 2, 3]]

    got = generic_simulation_study(
        sim_func=lambda x: x,
        params_list=params_list,
        results_processor=_results_processor,
    )
    assert [1, 2, 3] == got["raw_results"]
    assert [1, 4, 9] == got["processed"]
