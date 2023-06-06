import numpy as np

from fte.montypy.generic import generic_simulation_study


def coverage_simulation_study(
    n_sims,
    sim_func,
    params_list=None,
    evaluator="loop",
    n_cores=1,
    show_progress=True,
):
    if params_list is None:
        params_list = [{"_id": k} for k in range(n_sims)]

    return generic_simulation_study(
        sim_func=sim_func,
        params_list=params_list,
        results_processor=_coverage_results_processor,
        evaluator=evaluator,
        n_cores=n_cores,
        show_progress=show_progress,
    )


def _coverage_results_processor(results):
    processed = [_check_estimate_inside_confidence_interval(res) for res in results]
    return {
        "processed": processed,
        "coverage": np.mean(processed),
    }


def _check_estimate_inside_confidence_interval(res):
    band = res["confidence_interval"]
    true = res["true"]

    inside = (band.lower <= true) & (true <= band.upper)
    return all(inside)
