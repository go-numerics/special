"""Combinatorics, logistic/log helpers and complete elliptic integrals."""

import math

import numpy as np
import scipy.special as sc
from mpmath import mp, mpf

from common import SPECIALS, loguniform, rng, uniform

N = 1000


def nan_guard(fn):
    def f(*a):
        if any(math.isnan(x) for x in a):
            return math.nan
        return fn(*a)
    return f


# ---------------------------------------------------------------- factorials

@nan_guard
def factorial_mp(n):
    if n < 0:
        return 0.0
    if math.isinf(n):
        return math.inf
    return mp.gamma(mpf(n) + 1)


def factorial_cases():
    r = rng("factorial")
    c = [("special", (x,)) for x in [math.nan, math.inf, -1.0, -0.5, 0.0, 1.0, 170.0, 171.0, 170.62, 170.63]]
    c += [("integers", (float(n),)) for n in range(0, 171)]
    c += [("(0,1)", (x,)) for x in uniform(r, 1e-6, 1, N)]
    c += [("(1,170.6)", (x,)) for x in uniform(r, 1, 170.6, N)]
    return c


def factorial2_mp(n):
    n = int(n)
    if n < -1:
        return 0.0
    if n <= 0:
        return 1.0
    p = 1
    for k in range(n, 1, -2):
        p *= k
    return mpf(p)


def factorial2_cases():
    return [("n in [-5,320]", (float(n),)) for n in range(-5, 321)]


@nan_guard
def log_factorial_mp(n):
    if n < 0:
        return math.nan
    if math.isinf(n):
        return math.inf
    return mp.loggamma(mpf(n) + 1)


def log_factorial_cases():
    r = rng("log_factorial")
    c = [("special", (x,)) for x in [math.nan, math.inf, -1.0, 0.0, 1.0, 2.0, 5e-324]]
    c += [("(0,3)", (x,)) for x in uniform(r, 1e-8, 3, N)]
    c += [("[3,1e5)", (x,)) for x in loguniform(r, 3, 1e5, N)]
    c += [("[1e5,1e300]", (x,)) for x in loguniform(r, 1e5, 1e300, N)]
    c += [("integers", (float(n),)) for n in range(0, 300)]
    return c


# ---------------------------------------------------------------- comb/perm

def comb_mp(n, k):
    n, k = int(n), int(k)
    if k < 0 or n < 0 or k > n:
        return 0.0
    return mpf(mp.binomial(n, k)) if n < 10**6 else mp.binomial(mpf(n), mpf(k))


def comb_cases():
    r = rng("comb")
    c = [("n<=64", (float(n), float(k))) for n in range(0, 65) for k in range(-1, n + 2, max(1, n // 8))]
    c += [("n<=1100", (float(n), float(r.randint(0, n)))) for n in [r.randint(65, 1100) for _ in range(N)]]
    c += [("huge n, small k", (float(n), float(r.randint(0, 64)))) for n in [int(math.exp(r.uniform(math.log(1e3), math.log(1e15)))) for _ in range(N)]]
    c += [("huge n, large k", (float(n), float(int(n * r.uniform(0.0, 1.0))))) for n in [int(math.exp(r.uniform(math.log(1e3), math.log(1e15)))) for _ in range(N)]]
    return c


def perm_mp(n, k):
    n, k = int(n), int(k)
    if k < 0 or n < 0 or k > n:
        return 0.0
    if k < 2000 and n < 10**7:
        p = 1
        for i in range(k):
            p *= n - i
        return mpf(p)
    with mp.extraprec(200):
        return mp.exp(mp.loggamma(mpf(n) + 1) - mp.loggamma(mpf(n - k) + 1))


def perm_cases():
    r = rng("perm")
    c = [("n<=64", (float(n), float(k))) for n in range(0, 65) for k in range(-1, n + 2, max(1, n // 8))]
    c += [("n<=200", (float(n), float(r.randint(0, n)))) for n in [r.randint(65, 200) for _ in range(N)]]
    c += [("huge n, small k", (float(n), float(r.randint(0, 64)))) for n in [int(math.exp(r.uniform(math.log(1e3), math.log(1e15)))) for _ in range(N)]]
    c += [("huge n, k>64", (float(n), float(r.randint(65, 300)))) for n in [int(math.exp(r.uniform(math.log(1e3), math.log(1e15)))) for _ in range(N)]]
    return c


@nan_guard
def binom_mp(n, k):
    n, k = mpf(n), mpf(k)
    with mp.extraprec(100):
        try:
            return mp.binomial(n, k)
        except (ValueError, ZeroDivisionError):
            return math.nan


def binom_cases():
    r = rng("binom")
    c = [("k integer <=64, n real", (n, float(r.randint(0, 64)))) for n in uniform(r, -50, 200, N)]
    c += [("n,k > 0 real", (n, k)) for n, k in [(n, n * r.uniform(0, 1)) for n in loguniform(r, 0.1, 1e10, N)]]
    c += [("n>0, k real small", (n, k)) for n, k in zip(loguniform(r, 1, 1e5, N), uniform(r, -0.9, 10, N))]
    return c


# ---------------------------------------------------------------- logistic

@nan_guard
def expit_mp(x):
    if math.isinf(x):
        return 1.0 if x > 0 else 0.0
    x = mpf(x)
    return 1 / (1 + mp.exp(-x))


def expit_cases():
    r = rng("expit")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("[-746,-30)", (x,)) for x in uniform(r, -746, -30, N)]
    c += [("[-30,30)", (x,)) for x in uniform(r, -30, 30, N)]
    c += [("[30,40)", (x,)) for x in uniform(r, 30, 40, N)]
    c += [("tiny", (x,)) for x in loguniform(r, 1e-300, 1e-5, N // 2) + loguniform(r, 1e-300, 1e-5, N // 2, -1)]
    return c


@nan_guard
def log1pexp_mp(x):
    if math.isinf(x):
        return math.inf if x > 0 else 0.0
    x = mpf(x)
    with mp.extraprec(60):
        return mp.log1p(mp.exp(x))


def log1pexp_cases():
    r = rng("log1pexp")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("[-746,-30)", (x,)) for x in uniform(r, -746, -30, N)]
    c += [("[-30,30)", (x,)) for x in uniform(r, -30, 30, N)]
    c += [("[30,1e300]", (x,)) for x in loguniform(r, 30, 1e300, N)]
    return c


@nan_guard
def log_expit_mp(x):
    v = log1pexp_mp(-x)
    return -v


@nan_guard
def logit_mp(p):
    if p < 0 or p > 1:
        return math.nan
    if p == 0:
        return -math.inf
    if p == 1:
        return math.inf
    p = mpf(p)
    return mp.log(p / (1 - p))


def logit_cases():
    r = rng("logit")
    c = [("special", (x,)) for x in [math.nan, 0.0, 1.0, -0.1, 1.1, 0.5, 5e-324]]
    c += [("near 0", (x,)) for x in loguniform(r, 1e-300, 1e-3, N)]
    c += [("near 1/2", (0.5 + d,)) for d in uniform(r, -1e-3, 1e-3, N)]
    c += [("(0,1)", (x,)) for x in uniform(r, 1e-3, 1 - 1e-3, N)]
    c += [("near 1", (1 - d,)) for d in loguniform(r, 1.2e-16, 1e-3, N)]
    return c


@nan_guard
def xlogy_mp(x, y):
    if x == 0:
        return 0.0
    if y < 0:
        return math.nan
    if y == 0:
        return -math.inf * math.copysign(1, x)
    if math.isinf(x) or math.isinf(y):
        return float(x * math.log(y)) if y != 1 else math.nan
    return mpf(x) * mp.log(mpf(y))


def xlogy_cases():
    r = rng("xlogy")
    c = [("special", a) for a in [(0, 0), (0, -1), (0, math.nan), (1, 0), (math.nan, 1), (2, -1), (1, 1), (math.inf, 2), (-3, math.inf)]]
    c += [("moderate", (x, y)) for x, y in zip(uniform(r, -10, 10, N), loguniform(r, 1e-300, 1e300, N))]
    c += [("y near 1", (x, 1 + d)) for x, d in zip(uniform(r, -10, 10, N), uniform(r, -1e-6, 1e-6, N))]
    return c


@nan_guard
def xlog1py_mp(x, y):
    if x == 0:
        return 0.0
    if y < -1:
        return math.nan
    if y == -1:
        return -math.inf * math.copysign(1, x)
    if math.isinf(x) or math.isinf(y):
        return float(x * math.log1p(y)) if y != 0 else math.nan
    with mp.extraprec(60):
        return mpf(x) * mp.log1p(mpf(y))


def xlog1py_cases():
    r = rng("xlog1py")
    c = [("special", a) for a in [(0, -1), (0, -2), (0, math.nan), (1, -1), (math.nan, 1), (2, -2), (1, 0)]]
    c += [("moderate", (x, y)) for x, y in zip(uniform(r, -10, 10, N), loguniform(r, 1e-6, 1e300, N))]
    c += [("y small", (x, y)) for x, y in zip(uniform(r, -10, 10, N), loguniform(r, 1e-300, 1e-6, N // 2) + loguniform(r, 1e-300, 1e-6, N // 2, -1))]
    c += [("y near -1", (x, -1 + d)) for x, d in zip(uniform(r, -10, 10, N), loguniform(r, 1e-16, 1e-2, N))]
    return c


def lse_mp(*xs):
    if any(math.isnan(x) for x in xs):
        return math.nan
    if any(x == math.inf for x in xs):
        return math.inf
    xs = [x for x in xs if x != -math.inf]
    if not xs:
        return -math.inf
    with mp.extraprec(60):
        return mp.log(mp.fsum(mp.exp(mpf(x)) for x in xs))


def vec_cases(name):
    r = rng(name)
    c = [("special", a) for a in [(math.nan, 0, 0, 0), (math.inf, 0, 0, 0), (-math.inf,) * 4, (-math.inf, 0, 1, 2), (0, 0, 0, 0)]]
    c += [("moderate", tuple(uniform(r, -10, 10, 4))) for _ in range(N)]
    c += [("one dominant", tuple([r.uniform(-1, 1)] + uniform(r, -60, -30, 3))) for _ in range(N)]
    c += [("huge", tuple(uniform(r, 600, 800, 4))) for _ in range(N)]
    c += [("wide spread", tuple(uniform(r, -800, 800, 4))) for _ in range(N)]
    c += [("near cancellation", tuple([0.0] + uniform(r, -40, -36, 3))) for _ in range(N)]
    return c


def softmax0_mp(*xs):
    if any(math.isnan(x) or x == math.inf for x in xs):
        return math.nan
    if all(x == -math.inf for x in xs):
        return math.nan
    with mp.extraprec(60):
        es = [mp.exp(mpf(x)) if x != -math.inf else mpf(0) for x in xs]
        return es[0] / mp.fsum(es)


def log_softmax0_mp(*xs):
    if any(math.isnan(x) or x == math.inf for x in xs) or all(x == -math.inf for x in xs):
        return math.nan
    if xs[0] == -math.inf:
        return -math.inf
    # x0 - lse cancels when x0 dominates: -log(1 + Σ exp(x_i - x0)) instead.
    with mp.extraprec(100):
        return -mp.log1p(mp.fsum(mp.exp(mpf(x) - mpf(xs[0])) for x in xs[1:] if x != -math.inf))


# ---------------------------------------------------------------- elliptic

@nan_guard
def ellipk_mp(m):
    if m > 1:
        return math.nan
    if m == 1:
        return math.inf
    if m == -math.inf:
        return 0.0
    with mp.extraprec(60):
        return mp.ellipk(mpf(m))


@nan_guard
def ellipkm1_mp(p):
    if p < 0:
        return math.nan
    if p == 0:
        return math.inf
    if p == math.inf:
        return 0.0
    with mp.extraprec(60):
        return mp.ellipk(mp.fsub(1, mpf(p), exact=True))


@nan_guard
def ellipe_mp(m):
    if m > 1:
        return math.nan
    if m == 1:
        return 1.0
    if m == -math.inf:
        return math.inf
    with mp.extraprec(60):
        return mp.ellipe(mpf(m))


def ellip_cases(name):
    r = rng(name)
    c = [("special", (x,)) for x in [math.nan, 1.0, 0.0, -math.inf, 2.0, 0.5, -1.0]]
    c += [("m<-1", (x,)) for x in loguniform(r, 1, 1e300, N, -1)]
    c += [("[-1,0.5)", (x,)) for x in uniform(r, -1, 0.5, N)]
    c += [("[0.5,1-1e-3)", (x,)) for x in uniform(r, 0.5, 1 - 1e-3, N)]
    c += [("near 1", (1 - d,)) for d in loguniform(r, 1.2e-16, 1e-3, N)]
    return c


def ellipkm1_cases():
    r = rng("ellipkm1")
    c = [("special", (x,)) for x in [math.nan, 0.0, math.inf, -1.0, 1.0, 5e-324]]
    c += [("p tiny", (x,)) for x in loguniform(r, 1e-300, 1e-3, N)]
    c += [("(1e-3,1]", (x,)) for x in uniform(r, 1e-3, 1, N)]
    c += [("p>1", (x,)) for x in loguniform(r, 1, 1e300, N)]
    return c


def lse4(*a):
    return sc.logsumexp(np.array(a))


FIXTURES = {
    "factorial": lambda: (1, factorial_cases(), factorial_mp, lambda n: sc.factorial(np.array([n]))[0], "n! = Γ(n+1)"),
    "factorial2": lambda: (1, factorial2_cases(), factorial2_mp, lambda n: sc.factorial2(int(n)), "n!!"),
    "log_factorial": lambda: (1, log_factorial_cases(), log_factorial_mp, lambda n: sc.gammaln(n + 1), "log n! (SciPy: gammaln(n+1))"),
    "comb": lambda: (2, comb_cases(), comb_mp, lambda n, k: sc.comb(int(n), int(k)), "C(n, k)"),
    "perm": lambda: (2, perm_cases(), perm_mp, lambda n, k: sc.perm(int(n), int(k)), "n!/(n-k)!"),
    "binom": lambda: (2, binom_cases(), binom_mp, sc.binom, "generalised binomial"),
    "expit": lambda: (1, expit_cases(), expit_mp, sc.expit, "1/(1+e^-x)"),
    "log_expit": lambda: (1, expit_cases(), log_expit_mp, sc.log_expit, "log expit(x)"),
    "log1pexp": lambda: (1, log1pexp_cases(), log1pexp_mp, sc.softplus, "log(1+e^x) (SciPy: softplus)"),
    "logit": lambda: (1, logit_cases(), logit_mp, sc.logit, "log(p/(1-p))"),
    "xlogy": lambda: (2, xlogy_cases(), xlogy_mp, sc.xlogy, "x log y"),
    "xlog1py": lambda: (2, xlog1py_cases(), xlog1py_mp, sc.xlog1py, "x log1p y"),
    "logsumexp4": lambda: (4, vec_cases("logsumexp4"), lse_mp, lse4, "log Σ exp(x_i), 4 terms"),
    "softmax4": lambda: (4, vec_cases("softmax4"), softmax0_mp, lambda *a: sc.softmax(np.array(a))[0], "softmax(x)_0, 4 terms"),
    "log_softmax4": lambda: (4, vec_cases("log_softmax4"), log_softmax0_mp, lambda *a: sc.log_softmax(np.array(a))[0], "log_softmax(x)_0, 4 terms"),
    "ellipk": lambda: (1, ellip_cases("ellipk"), ellipk_mp, sc.ellipk, "K(m)"),
    "ellipkm1": lambda: (1, ellipkm1_cases(), ellipkm1_mp, sc.ellipkm1, "K(1-p)"),
    "ellipe": lambda: (1, ellip_cases("ellipe"), ellipe_mp, sc.ellipe, "E(m)"),
}
