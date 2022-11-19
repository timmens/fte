SIMULATION_EXAMPLES_REGRESSION = {
    # "name": {"data_process": data_process, "data_process_kwargs": data_process_kwargs}
    "regression": {
        "model_func": "linear",
        "simulate_params": "default_regression",
        "simulate_features": "uniform",
        "kernel": "RBF",
        "has_intercept": True,
        "error_scale": 0.5,
        "heteroskedastic": False,
    },
    "regression_heteroskedastic": {
        "model_func": "linear",
        "simulate_params": "default_regression",
        "simulate_features": "uniform",
        "kernel": "RBF",
        "has_intercept": True,
        "heteroskedastic": True,
    },
}

SIMULATION_EXAMPLES_CAUSAL = {
    "causal": {
        "model_func": "linear_causal",
        "simulate_params": "causal",
        "simulate_features": "uniform",
        "kernel": "RBF",
        "has_intercept": True,
        "heteroskedastic": False,
    },
}


__all__ = [
    SIMULATION_EXAMPLES_REGRESSION,
    SIMULATION_EXAMPLES_CAUSAL,
]
