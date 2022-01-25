"""Implement a simple version of Vandemonde OPP scheme over the complex numbers"""

from math import pi, sqrt, log
from random import sample

import numpy as np

from .utils import circular_complex_normal, partition_matrix


class AnalogGASP:
    def __init__(self, m, n, X, N=None, delta=None):
        # set internal variables
        self.m = m
        self.n = n
        self.X = X
        self.K = 2 * m * n + 2 * X - 1
        if N is None:
            N = self.K
        self.N = N
        self.delta = delta

        if self.N < self.K:
            raise ValueError("too few servers")

        # choose evaluation points
        self.alphas = np.exp([2j * pi * i / N for i in range(N)])

        # compute required variance for random part
        U = np.diag(self.alphas[:X] ** (m * n)) @ np.vander(
            self.alphas[:X], X, increasing=True
        )
        V = np.vander(self.alphas[:X], m, increasing=True)
        Gamma = U @ U.conjugate().transpose()
        Sigma = V @ V.conjugate().transpose()
        M = np.linalg.inv(Gamma) @ Sigma
        self.trr = M.trace().real

        U = np.diag(self.alphas[:X] ** (m * n)) @ np.vander(
            self.alphas[:X], X, increasing=True
        )
        V = np.vander(self.alphas[:X] ** m, n, increasing=True)
        Gamma = U @ U.conjugate().transpose()
        Sigma = V @ V.conjugate().transpose()
        M = np.linalg.inv(Gamma) @ Sigma
        self.trs = M.trace().real

    def __str__(self):

        return f"AnalogGASP(m={self.m}, n={self.n}, X={self.X}, N={self.N})"

    def __call__(self, A, B, delta=None):

        if delta is None:
            delta = self.delta

        t, sA = A.shape
        sB, r = B.shape

        if sA != sB:
            raise ValueError("matrices not conformable")

        s = sA

        var = (t * s * self.trr / self.m + s * r * self.trs / self.n) / (delta * log(2))
        sigma = sqrt(var)

        # encode A
        AP = partition_matrix(A, vertically=self.m)
        R = [
            circular_complex_normal(scale=sigma, size=(t // self.m, s))
            for _ in range(self.X)
        ]
        AT = [
            sum(a * alpha ** i for i, a in enumerate(AP))
            + sum(r * alpha ** (self.m * self.n + i) for i, r in enumerate(R))
            for alpha in self.alphas
        ]

        # encode B
        BP = partition_matrix(B, horizontally=self.n)
        S = [
            circular_complex_normal(scale=sigma, size=(s, r // self.n))
            for _ in range(self.X)
        ]
        BT = [
            sum(b * alpha ** (self.m * i) for i, b in enumerate(BP))
            + sum(s * alpha ** (self.m * self.n + i) for i, s in enumerate(S))
            for alpha in self.alphas
        ]

        # computing matrix products, "@" computes matrix product, "*" computes elementwise product
        CT = [At @ Bt for At, Bt in zip(AT, BT)]

        # choose random subset of K servers and interpolate
        subset = sample(list(range(self.N)), k=self.N)[: self.K]
        alphar = self.alphas[subset]
        CTr = np.array(CT)[subset]
        G = np.linalg.inv(np.vander(alphar, increasing=True))[: self.m * self.n, :]
        C = [sum(Ct * g for Ct, g in zip(CTr, G[i])) for i in range(self.m * self.n)]
        C = np.block(
            [[C[i + self.m * j] for j in range(self.n)] for i in range(self.m)]
        )

        return C
