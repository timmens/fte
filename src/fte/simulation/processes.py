import numpy as np
import sklearn.gaussian_process.kernels as sklearn_kernels


def simulate_gaussian_process(
    n_samples, kernel, rng, *, scale=1.0, n_periods=None, grid=None
):
    """Simulate mean-zero gaussian process using specified kernel.

    Args:
        n_samples (int): Number of simulated processes.
        kernel (callable or str): Kernel function of two arguments.
        rng (np.random.Generator): Random number generator.
        scale (float): Scale the process by this value.
        n_periods (int): Number of time realizations of each process. Realizations
            points are given by an equidistant grid over [0, 1] if grid is None.
        grid (np.ndarray): Grid on which to simulate.

    Returns:
        process (np.ndarray): Simulated process of shape (n_periods, n_sim).

    """
    if grid is None and n_periods is None:
        raise ValueError("One of n_periods and grid has to be non-None.")

    if isinstance(kernel, str):
        kernel = get_kernel(kernel_name=kernel)

    if grid is None:
        grid = np.linspace(0, 1, n_periods)
    cov = scale * kernel(grid)
    process = rng.multivariate_normal(np.zeros(n_periods), cov, size=n_samples).T
    return process


def get_kernel(kernel_name, kernel_kwargs=None):
    """Return kernel function.

    Args:
        kernel (str): Kernel. Available kernels are in ["WhiteKernel", "RBF",
            "Matern", "BrownianMotion", "SelfSimilar"].
        kernel_kwargs: Keyword arguments passed to the specified kernel.

    Returns:
        kernel (callable): Kernel function of two arguments.

    """
    kernel_kwargs = _add_defaults_to_kwargs(kernel_name, kernel_kwargs)
    if kernel_name == "BrownianMotion":

        def _kernel(x, y=None):
            sigma = kernel_kwargs["sigma"]
            if y is None:
                x, y = np.meshgrid(x, x)
            else:
                x = x.reshape(-1, 1)
                y = np.atleast_2d(y)
                if len(y) > 1:
                    raise ValueError("Second argument has to be a scalar.")
            return sigma**2 * np.minimum(x, y).flatten()

    elif kernel_name == "SelfSimilar":

        def _kernel(x, y=None):
            sigma = kernel_kwargs["sigma"]
            kappa = kernel_kwargs["kappa"]
            if y is None:
                x, y = np.meshgrid(x, x)
            else:
                x = x.reshape(-1, 1)
                y = np.atleast_2d(y)
                if len(y) > 1:
                    raise ValueError("Second argument has to be a scalar.")
            return sigma * (x**kappa + y**kappa - np.abs(x - y) ** kappa).flatten()

    else:
        kernel_name = getattr(sklearn_kernels, kernel_name)(**kernel_kwargs)

        def _kernel(x, y=None):
            if y is None:
                return kernel_name(x.reshape(-1, 1))
            else:
                y = np.atleast_2d(y)
                if len(y) > 1:
                    raise ValueError("Second argument has to be a scalar.")
                return 0.5 * kernel_name(x.reshape(-1, 1), x).flatten()

    return _kernel


def _add_defaults_to_kwargs(kernel_name, kwargs):
    kwargs = {} if kwargs is None else kwargs
    default_kwargs = {
        "Matern": {"nu": 0.5, "length_scale": 0.1},
        "RBF": {"length_scale": 0.5},
        "BrownianMotion": {"sigma": np.sqrt(2)},
        "SelfSimilar": {"sigma": 3.0, "kappa": 3.0},
    }
    kwargs = {**default_kwargs[kernel_name], **kwargs}
    return kwargs
