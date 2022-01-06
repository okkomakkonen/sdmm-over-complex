from math import pi, e, log2
from statistics import fmean, stdev

import numpy as np
import matplotlib.pyplot as plt

from sdmm import MatDot, SimpleOPP

# rounds to do
ROUNDS = 100


def compute_errors(deltas, sdmm_algorithm, rounds=ROUNDS):

    print(f"Starting {sdmm_algorithm}")

    # mean error for each delta
    ave_errors = []

    # raw data of all errors
    errors = []

    for delta in deltas:

        errs = []

        for _ in range(rounds):

            A = np.random.normal(loc=0.0, scale=1.0, size=(t, s))
            B = np.random.normal(loc=0.0, scale=1.0, size=(s, r))

            C = sdmm_algorithm(A, B, delta=delta)

            errs.append(np.linalg.norm(A @ B - C, "fro"))

        errors.append(errs)
        err = fmean(errs)
        ave_errors.append(err)

        print(delta, err)
    
    print()
    return ave_errors


"""
# parameters
t, s, r = 100, 100, 100
p = 10
X = 1
K = 2 * p + 2 * X - 1

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

# compute entropy of input and normalize the deltas
hA = 0.5 * t * s * log2(2 * pi * e * 1.0 ** 2)
hB = 0.5 * s * r * log2(2 * pi * e * 1.0 ** 2)
deltas = rel_deltas * (hA + hB)

err0 = compute_errors(deltas, MatDot(p=p, X=X, N=K + 0))
err1 = compute_errors(deltas, MatDot(p=p, X=X, N=K + 1))
err2 = compute_errors(deltas, MatDot(p=p, X=X, N=K + 2))
err3 = compute_errors(deltas, MatDot(p=p, X=X, N=K + 3))
err4 = compute_errors(deltas, MatDot(p=p, X=X, N=K + 4))

plt.loglog(rel_deltas, err0, "x-", label="0 stragglers")
plt.loglog(rel_deltas, err1, "x-", label="1 straggler")
plt.loglog(rel_deltas, err2, "x-", label="2 stragglers")
plt.loglog(rel_deltas, err3, "x-", label="3 stragglers")
plt.loglog(rel_deltas, err4, "x-", label="4 stragglers")

plt.grid()
plt.legend()
plt.title(f"Relative information leakage vs. mean Frobenius norm of error")
plt.xlabel("relative information leakage")
plt.ylabel("error")
plt.savefig("plot.eps")
plt.show()
"""

"""
# parameters
t, s, r = 100, 100, 100
m = n = 4
X = 1
K = 2 * m * n + 2 * X - 1

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

# compute entropy of input and normalize the deltas
hA = 0.5 * t * s * log2(2 * pi * e * 1.0 ** 2)
hB = 0.5 * s * r * log2(2 * pi * e * 1.0 ** 2)
deltas = rel_deltas * (hA + hB)

err0 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K + 0))
err1 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K + 1))
err2 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K + 2))
err3 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K + 3))
err4 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K + 4))

plt.loglog(rel_deltas, err0, "x-", label="0 stragglers")
plt.loglog(rel_deltas, err1, "x-", label="1 straggler")
plt.loglog(rel_deltas, err2, "x-", label="2 stragglers")
plt.loglog(rel_deltas, err3, "x-", label="3 stragglers")
plt.loglog(rel_deltas, err4, "x-", label="4 stragglers")

plt.grid()
plt.legend()
plt.title(f"Relative information leakage vs. mean Frobenius norm of error")
plt.xlabel("relative information leakage")
plt.ylabel("error")
plt.savefig("plot.eps")
plt.show()
"""

"""
# parameters
t, s, r = 100, 100, 100
m = n = 2
p = m * n
X = 3
K = 2 * p + 2 * X - 1

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

# compute entropy of input and normalize the deltas
hA = 0.5 * t * s * log2(2 * pi * e * 1.0 ** 2)
hB = 0.5 * s * r * log2(2 * pi * e * 1.0 ** 2)
deltas = rel_deltas * (hA + hB)

err1 = compute_errors(deltas, MatDot(p=p, X=X, N=K))
err2 = compute_errors(deltas, SimpleOPP(m=m, n=n, X=X, N=K))

plt.loglog(rel_deltas, err1, "x-", label="MatDot")
plt.loglog(rel_deltas, err2, "x-", label="SimpleOPP")

plt.grid()
plt.legend()
plt.title(f"Relative information leakage vs. mean Frobenius norm of error")
plt.xlabel("relative information leakage")
plt.ylabel("error")
plt.savefig("plot.eps")
plt.show()
"""

# parameters
t, s, r = 120, 120, 120
m = n = 2
p = m * n
X = 3
K = 2 * p + 2 * X - 1

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

# compute entropy of input and normalize the deltas
hA = 0.5 * t * s * log2(2 * pi * e * 1.0 ** 2)
hB = 0.5 * s * r * log2(2 * pi * e * 1.0 ** 2)
deltas = rel_deltas * (hA + hB)

err10 = compute_errors(deltas, MatDot(p=1, X=1))
err11 = compute_errors(deltas, MatDot(p=2, X=2))
err12 = compute_errors(deltas, MatDot(p=3, X=3))
err13 = compute_errors(deltas, MatDot(p=4, X=4))

plt.loglog(rel_deltas, err10, "bx-", label="p = 1")
plt.loglog(rel_deltas, err11, "bx--", label="p = 2")
plt.loglog(rel_deltas, err12, "bx-.", label="p = 3")
plt.loglog(rel_deltas, err13, "bx:", label="p = 4")

plt.grid()
plt.legend()
plt.xlabel("relative information leakage")
plt.ylabel("error")
plt.savefig("plot.eps")
plt.show()