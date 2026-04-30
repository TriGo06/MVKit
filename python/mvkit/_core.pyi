from typing import Optional

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
    scheme: str = ...,
    increments: Optional[np.ndarray] = ...,
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
    scheme: str = ...,
    increments: Optional[np.ndarray] = ...,
) -> np.ndarray: ...
def simulate_kuramoto(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    coupling_k: float,
    omegas: np.ndarray,
    sigma: float,
    record_every: int = ...,
    seed: int = ...,
    scheme: str = ...,
    increments: Optional[np.ndarray] = ...,
) -> np.ndarray: ...
def simulate_mean_field_cir(
    x0: np.ndarray,
    t_final: float,
    n_steps: int,
    kappa: float,
    theta: float,
    b: float,
    sigma: float,
    record_every: int = ...,
    seed: int = ...,
    scheme: str = ...,
    increments: Optional[np.ndarray] = ...,
) -> np.ndarray: ...
