import contextlib
from functools import partial

import joblib
from joblib import delayed
from joblib import Parallel
from tqdm import tqdm


def generic_simulation_study(
    sim_func,
    params_list,
    results_processor=None,
    evaluator="loop",
    n_cores=1,
    show_progress=True,
):

    # ==================================================================================
    # (Parallel) Simulation
    # ==================================================================================

    evaluator = get_evaluator(
        evaluator, func=sim_func, n_cores=n_cores, show_progress=show_progress
    )

    raw_results = evaluator(params_list)

    # ==================================================================================
    # Process Results
    # ==================================================================================

    out = {"raw_results": raw_results}

    if results_processor is not None:
        results = results_processor(raw_results)
        out["processed"] = results

    return out


def get_evaluator(evaluator, *, func, n_cores, show_progress):
    if evaluator == "loop":

        def _evaluator(params_list):
            iterator = tqdm(params_list) if show_progress else params_list
            return [func(**params) for params in iterator]

    elif evaluator == "joblib":

        def _evaluator(params_list):
            with parallel_context(show_progress, params_list):
                return Parallel(n_jobs=n_cores)(
                    delayed(func)(**params) for params in params_list
                )

    elif callable(evaluator):
        _evaluator = partial(
            evaluator, sim_func=func, n_cores=n_cores, show_progress=show_progress
        )
    else:
        msg = "evaluator needs to be either callable or in {'loop', 'joblib'}."
        raise ValueError(msg)
    return _evaluator


def parallel_context(show_progress, params_list):
    if show_progress:
        context = tqdm_joblib(tqdm(total=len(params_list)))
    else:
        context = contextlib.nullcontext()
    return context


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""

    def tqdm_print_progress(self):
        if self.n_completed_tasks > tqdm_object.n:
            n_completed = self.n_completed_tasks - tqdm_object.n
            tqdm_object.update(n=n_completed)

    original_print_progress = joblib.parallel.Parallel.print_progress
    joblib.parallel.Parallel.print_progress = tqdm_print_progress

    try:
        yield tqdm_object
    finally:
        joblib.parallel.Parallel.print_progress = original_print_progress
        tqdm_object.close()
