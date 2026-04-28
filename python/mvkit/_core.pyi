import numpy as np

__version__: str

def simulate_cucker_smale(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    spatial_dim: int,
    beta: float = ...,
    sigma: float = ...,
    record_every: int = ...,
    seed: int = ...,
) -> np.ndarray: ...
def simulate_linear_quadratic(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    a: float,
    b: float,
    sigma: float,
    record_every: int = ...,
    seed: int = ...,
) -> np.ndarray: ...
