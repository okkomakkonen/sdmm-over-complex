"""This script plots the security vs. numerical error plots for different SDMM schemes"""

from math import pi, e, log2
from statistics import fmean

import numpy as np
import matplotlib.pyplot as plt

from sdmm import AnalogMatDot, AnalogGASP

# rounds to do
ROUNDS = 1000

# sizes of matrices, A is t x s, B is s x r
t, s, r = 36, 36, 36


def compute_errors(rel_deltas, sdmm_algorithm, rounds=ROUNDS):

    print(f"Starting {sdmm_algorithm}")

    # compute entropy of input and normalize the deltas
    hA = 0.5 * t * s * log2(2 * pi * e * 1.0 ** 2)
    hB = 0.5 * s * r * log2(2 * pi * e * 1.0 ** 2)
    deltas = rel_deltas * (hA + hB)

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


def plot_errors(ax, rel_deltas, sdmm_algorithm, *args, **kwargs):
    err = compute_errors(rel_deltas, sdmm_algorithm)
    return ax.loglog(rel_deltas, err, *args, **kwargs)


"""
# Varying number of colluding servers

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, tight_layout=True, frameon=False, dpi=200.0, figsize=(7, 4)
)

plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=1), "b.-")
plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=2), "b.--")
plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=3), "b.-.")
plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=4), "b.:")

plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=1), "r.-")
plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=2), "r.--")
plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=3), "r.-.")
plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=4), "r.:")

ax1.loglog([], [], "k.-", label="1 colluding")
ax1.loglog([], [], "k.--", label="2 colluding")
ax1.loglog([], [], "k.-.", label="3 colluding")
ax1.loglog([], [], "k.:", label="4 colluding")

ax1.grid()
ax1.legend()
ax1.set_xlabel("relative information leakage")
ax1.set_ylabel("error")

plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=1), "b.-")
plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=2), "b.--")
plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=3), "b.-.")
plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=4), "b.:")

plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=1), "r.-")
plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=2), "r.--")
plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=3), "r.-.")
plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=4), "r.:")

ax2.loglog([], [], "k.-", label="1 colluding")
ax2.loglog([], [], "k.--", label="2 colluding")
ax2.loglog([], [], "k.-.", label="3 colluding")
ax2.loglog([], [], "k.:", label="4 colluding")

ax2.grid()
ax2.legend()
ax2.set_xlabel("relative information leakage")
ax2.set_ylabel("error")

plt.savefig("plot.eps")
plt.show()
"""


# varying number of straggling servers

# relative information leakages
rel_deltas = np.logspace(-10, -5, 6)

fig, (ax1, ax2) = plt.subplots(
    1, 2, sharey=True, tight_layout=True, frameon=False, dpi=200.0, figsize=(7, 4)
)

plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=3, N=13), "b.-")
plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=3, N=14), "b.--")
plot_errors(ax1, rel_deltas, AnalogMatDot(p=4, X=3, N=15), "b.-.")

plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=3, N=13), "r.-")
plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=3, N=14), "r.--")
plot_errors(ax1, rel_deltas, AnalogGASP(m=2, n=2, X=3, N=15), "r.-.")

ax1.loglog([], [], "k.-", label="0 straggling")
ax1.loglog([], [], "k.--", label="1 straggling")
ax1.loglog([], [], "k.-.", label="2 straggling")

ax1.grid()
ax1.set_aspect("equal")
ax1.legend()
ax1.set_xlabel("relative information leakage")
ax1.set_ylabel("error")

plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=2, N=21), "b.-")
plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=2, N=22), "b.--")
plot_errors(ax2, rel_deltas, AnalogMatDot(p=9, X=2, N=23), "b.-.")

plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=2, N=21), "r.-")
plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=2, N=22), "r.--")
plot_errors(ax2, rel_deltas, AnalogGASP(m=3, n=3, X=2, N=23), "r.-.")

ax2.loglog([], [], "k.-", label="0 straggling")
ax2.loglog([], [], "k.--", label="1 straggling")
ax2.loglog([], [], "k.-.", label="2 straggling")

ax2.grid()
ax2.set_aspect("equal")
ax2.legend()
ax2.set_xlabel("relative information leakage")
ax2.set_ylabel("error")

plt.savefig("plot.eps")
plt.show()
