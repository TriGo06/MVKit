"""Fixed stream roles for reproducible LQ simulations."""

from enum import IntEnum

import numpy as np


class StreamRole(IntEnum):
    # Values form part of the seed mapping; never derive them from enum order.
    SCALAR_INITIAL = 0
    SCALAR_DYNAMICS = 1
    VECTOR_INITIAL = 2
    VECTOR_DYNAMICS = 3


def rng_for_role(seed: int, role: StreamRole) -> np.random.Generator:
    """Separate model/initial/dynamic streams without folding roles into seed.

    Reconstructing a role intentionally replays the same noise across outer
    iterations. Different master seeds do not exchange roles as with XOR masks.
    """
    sequence = np.random.SeedSequence(seed, spawn_key=(int(role),))
    return np.random.Generator(np.random.PCG64(sequence))
