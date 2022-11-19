from fte.confidence_bands.bands import estimate_confidence_band
from fte.fitting.fitting import get_fitter
from fte.montypy import coverage_simulation_study
from fte.simulation.simulate import get_data_simulator


def simulate_coverage_simultaneous_confidence_bands(
    n_sims,
    simulation_kwargs,
    data_process_kwargs,
    fitter=None,
    fitter_kwargs=None,
    band_kwargs=None,
    evaluator="loop",
    n_cores=1,
    show_progress=True,
):
    data_simulator = get_data_simulator(**data_process_kwargs)

    if fitter is None:
        _fitter = _get_default_fitter(data_simulator.is_causal)
    else:
        _fitter = fitter
    fitter = get_fitter(_fitter, fitter_kwargs)

    band_kwargs = {} if band_kwargs is None else band_kwargs

    def _sim_func(_id):
        data = data_simulator(seed=_id, **simulation_kwargs)
        res = fitter(data=data)
        band = estimate_confidence_band(result=res, **band_kwargs)

        if data.is_causal:
            true = data.params["treatment_effect"]
        else:
            true = data.params["model_func"]["slopes"][:, band_kwargs["coef_id"]]

        out = {
            "true": true,
            "confidence_interval": band,
        }
        return out

    out = coverage_simulation_study(
        n_sims=n_sims,
        sim_func=_sim_func,
        evaluator=evaluator,
        n_cores=n_cores,
        show_progress=show_progress,
    )
    return out


def _get_default_fitter(is_causal):
    if is_causal:
        fitter = "func_on_scalar_doubly_robust"
    else:
        fitter = "func_on_scalar"
    return fitter
