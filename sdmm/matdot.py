"""Implement a simple version of MatDot over the complex numbers"""

from math import pi, sqrt, log
from random import sample

import numpy as np

from .utils import circular_complex_normal, partition_matrix


class AnalogMatDot:
    def __init__(self, p, X, N=None, delta=None):
        # set internal variables
        self.p = p
        self.X = X
        self.K = 2 * p + 2 * X - 1
        if N is None:
            N = self.K
        self.N = N
        self.delta = delta

        if self.N < self.K:
            raise ValueError("too few servers")

        # choose evaluation points
        self.alphas = np.exp([2j * pi * i / N for i in range(N)])

        # compute required variance for random part
        V = np.vander(self.alphas[:X], p, increasing=True)
        U = np.diag(self.alphas[:X] ** p) @ np.vander(
            self.alphas[:X], X, increasing=True
        )
        M = np.linalg.inv(U) @ V
        M = M.conjugate().transpose() @ M
        self.trr = M.trace().real

        V = np.vander(self.alphas[:X], p, increasing=False)
        U = np.diag(self.alphas[:X] ** p) @ np.vander(
            self.alphas[:X], X, increasing=True
        )
        M = np.linalg.inv(U) @ V
        M = M.conjugate().transpose() @ M
        self.trs = M.trace().real

    def __str__(self):

        return f"AnalogMatDot(p={self.p}, X={self.X}, N={self.N})"

    def __call__(self, A, B, delta=None):

        if delta is None:
            delta = self.delta

        t, sA = A.shape
        sB, r = B.shape

        if sA != sB:
            raise ValueError("matrices not conformable")

        s = sA

        var = (t * s * self.trr + s * r * self.trs) / (self.p * delta * log(2))
        sigma = sqrt(var)

        # encode A
        AP = partition_matrix(A, horizontally=self.p)
        R = [
            circular_complex_normal(scale=sigma, size=(t, s // self.p))
            for _ in range(self.X)
        ]
        APR = AP + R
        AT = [sum(a * alpha ** i for i, a in enumerate(APR)) for alpha in self.alphas]

        # encode B
        BP = partition_matrix(B, vertically=self.p)
        S = [
            circular_complex_normal(scale=sigma, size=(s // self.p, r))
            for _ in range(self.X)
        ]
        BPS = list(reversed(BP)) + S
        BT = [sum(b * alpha ** i for i, b in enumerate(BPS)) for alpha in self.alphas]

        # computing matrix products, "@" computes matrix product, "*" computes elementwise product
        CT = [At @ Bt for At, Bt in zip(AT, BT)]

        # choose random subset of K servers and interpolate
        subset = sample(list(range(self.N)), k=self.N)[: self.K]
        alphar = self.alphas[subset]
        CTr = np.array(CT)[subset]
        G = np.linalg.inv(np.vander(alphar, increasing=True))[self.p - 1, :]
        C = sum(Ct * g for g, Ct in zip(G, CTr))

        return C
