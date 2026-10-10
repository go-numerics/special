"""SciPy side of the speed comparison (driven by nd/perf_test.go).

Reads one command per line on stdin: "<name> <n>"; evaluates the SciPy
function on the same n points the Go side uses and prints the elapsed
wall time in seconds of the vectorised call (array allocation of the result
included, as on the Go side). "quit" ends. The points are generated exactly
as in perf_test.go: x_i = lo + (hi - lo) * frac(i * 0.6180339887498949)
(the same IEEE operations in both languages).
"""

import sys
import time

import numpy as np
import scipy.special as sc

GOLD = 0.6180339887498949


def points(n, lo, hi):
    i = np.arange(n, dtype=np.float64)
    f = np.mod(i * GOLD, 1.0)
    return lo + (hi - lo) * f


# name -> (callable on arrays, list of (lo, hi) per argument); must match
# perfCases in nd/perf_test.go.
CASES = {
    "Gamma": (sc.gamma, [(0.1, 170)]),
    "LogGamma": (sc.gammaln, [(0.1, 1000)]),
    "Digamma": (sc.digamma, [(0.1, 100)]),
    "Beta": (sc.beta, [(0.1, 50), (0.1, 50)]),
    "LogBeta": (sc.betaln, [(0.1, 1000), (0.1, 1000)]),
    "Erf": (sc.erf, [(-5, 5)]),
    "Erfc": (sc.erfc, [(-5, 25)]),
    "Erfcx": (sc.erfcx, [(-5, 50)]),
    "ErfInv": (sc.erfinv, [(-0.999, 0.999)]),
    "ErfcInv": (sc.erfcinv, [(1e-10, 1.999)]),
    "Ndtr": (sc.ndtr, [(-30, 10)]),
    "Ndtri": (sc.ndtri, [(1e-12, 0.999999)]),
    "LogNdtr": (sc.log_ndtr, [(-100, 10)]),
    "Expit": (sc.expit, [(-50, 50)]),
    "Logit": (sc.logit, [(1e-6, 0.999999)]),
    "Log1pExp": (sc.softplus, [(-50, 50)]),
    "Xlogy": (sc.xlogy, [(0.1, 10), (0.1, 10)]),
    "Zeta": (sc.zeta, [(1.01, 60)]),
    "HurwitzZeta": (sc.zeta, [(1.5, 20), (0.1, 20)]),
    "Polygamma1": (lambda x: sc.polygamma(1, x), [(0.1, 100)]),
    "GammaP": (sc.gammainc, [(0.5, 50), (0.1, 60)]),
    "GammaPInv": (sc.gammaincinv, [(0.5, 50), (1e-6, 0.999999)]),
    "BetaInc": (sc.betainc, [(0.5, 50), (0.5, 50), (0.01, 0.99)]),
    "EllipK": (sc.ellipk, [(-10, 0.999)]),
    "EllipE": (sc.ellipe, [(-10, 0.999)]),
}


def main():
    cache = {}
    for line in sys.stdin:
        cmd = line.split()
        if not cmd or cmd[0] == "quit":
            break
        name, n = cmd[0], int(cmd[1])
        fn, ranges = CASES[name]
        key = (name, n)
        if key not in cache:
            cache[key] = [points(n, lo, hi) for lo, hi in ranges]
        args = cache[key]
        with np.errstate(all="ignore"):
            t0 = time.perf_counter()
            fn(*args)
            dt = time.perf_counter() - t0
        print(repr(dt), flush=True)


if __name__ == "__main__":
    main()
