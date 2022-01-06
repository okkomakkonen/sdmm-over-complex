"""Provides some helper functions for SDMM implementations"""

import numpy as np


def partition_matrix(M, *, horizontally=None, vertically=None):
    n, m = M.shape
    if horizontally is not None and vertically is None:
        # split horizontally
        if m % horizontally != 0:
            raise ValueError("matrix can't be evenly split")
        ms = m // horizontally
        return [M[:, i * ms : (i + 1) * ms] for i in range(horizontally)]
    if horizontally is None and vertically is not None:
        # split vertically
        if n % vertically != 0:
            raise ValueError("matrix can't be evenly split")
        ns = n // vertically
        return [M[i * ns : (i + 1) * ns, :] for i in range(vertically)]
    if horizontally is not None and vertically is not None:
        # split both
        if n % vertically != 0 or m % horizontally != 0:
            raise ValueError("matrix can't be evenly split")
        ms = m // horizontally
        ns = n // vertically
        return [
            [M[i * ns : (i + 1) * ns, j * ms : (j + 1) * ms] for j in range(vertically)]
            for i in range(vertically)
        ]
    raise ValueError("matrix must be split either horizontally or vertically (or both)")


def circular_complex_normal(loc=0.0, scale=1.0, size=None):
    return np.random.normal(
        loc=loc.real, scale=scale / 2.0, size=size
    ) + 1j * np.random.normal(loc=loc.imag, scale=scale / 2.0, size=size)
