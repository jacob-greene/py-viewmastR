"""R's ``set.seed(seed); sample(n, size)`` in Python.

R viewmastR draws its train split with ``sample(1:n, round(train_frac * n))``. With an
integer ``seed`` the Python package reproduces that draw exactly, so a Python model and an
R model made with the same ``set.seed()`` train on the same cells in the same order.

R defaults since 3.6.0: Mersenne-Twister, sample.kind = "Rejection" (R source
``src/main/RNG.c`` and ``src/main/random.c``).
"""

from __future__ import annotations

import numpy as np

_MASK = 0xFFFFFFFF


def _r_mt_state(seed: int) -> np.random.MT19937:
    """The Mersenne-Twister state that ``set.seed(seed)`` creates (RNG_Init in RNG.c)."""
    s = seed & _MASK
    for _ in range(50):  # initial scrambling
        s = (69069 * s + 1) & _MASK
    key = np.empty(625, dtype=np.uint32)
    for j in range(625):  # dummy[0] (the position) followed by the 624 state words
        s = (69069 * s + 1) & _MASK
        key[j] = s
    bg = np.random.MT19937()
    # FixupSeeds sets the position to 624, so the first draw regenerates the whole state.
    bg.state = {"bit_generator": "MT19937", "state": {"key": key[1:], "pos": 624}}
    return bg


class _RUnif:
    """``unif_rand()``: 32-bit Mersenne-Twister words scaled to (0, 1)."""

    def __init__(self, seed: int):
        self._bg = _r_mt_state(seed)

    def __call__(self) -> float:
        u = int(self._bg.random_raw()) * 2.3283064365386963e-10
        # fixup(): keep the value strictly inside (0, 1)
        return min(max(u, 0.5 * 2.328306437080797e-10), 1.0 - 0.5 * 2.328306437080797e-10)


def r_sample(n: int, size: int, seed: int) -> np.ndarray:
    """0-based indices of R's ``set.seed(seed); sample(n, size)`` (without replacement).

    Matches R >= 3.6 for ``n <= 1e7`` (R switches to a hash algorithm above that).
    """
    unif = _RUnif(seed)

    def unif_index(dn: int) -> int:  # R_unif_index with rejection sampling
        if dn <= 0:
            return 0
        b = int(np.ceil(np.log2(dn)))
        while True:
            v = 0
            for _ in range(0, b + 1, 16):  # rbits(): 16 random bits per draw
                v = 65536 * v + int(np.floor(unif() * 65536))
            v &= (1 << b) - 1
            if v < dn:
                return v

    pool = list(range(n))
    out = np.empty(size, dtype=np.int64)
    m = n
    for i in range(size):  # do_sample, no replacement: swap-remove from the pool
        j = unif_index(m)
        out[i] = pool[j]
        m -= 1
        pool[j] = pool[m]
    return out
