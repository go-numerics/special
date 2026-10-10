"""Zeta family fixtures: Riemann zeta, Hurwitz zeta, polygamma."""

import math

import scipy.special as sc
from mpmath import mp, mpf

from common import loguniform, rng, uniform

N = 1000


def zeta_mp(s):
    if math.isnan(s) or s == -math.inf:
        return math.nan
    if s == math.inf:
        return 1.0
    if s == 1:
        return math.inf
    if s >= 1100:
        return 1.0  # 2^-s below 2^-1100
    s = mpf(s)
    if s < 0 and s == mp.floor(s) and int(s) % 2 == 0:
        return 0.0
    with mp.extraprec(60):
        return mp.zeta(s)


def zeta_cases():
    r = rng("zeta")
    c = [("special", (x,)) for x in [math.nan, math.inf, -math.inf, 1.0, 0.0, -0.0, -2.0, -4.0, 2.0, 0.5, -1.0, 5e-324, -5e-324]]
    c += [("near pole", (1 + d,)) for d in loguniform(r, 1e-15, 1e-2, N // 2)]
    c += [("near pole", (1 - d,)) for d in loguniform(r, 1e-15, 1e-2, N // 2)]
    c += [("[-1,2]", (x,)) for x in uniform(r, -1, 2, N)]
    c += [("(2,64)", (x,)) for x in uniform(r, 2, 64, N)]
    c += [("[64,1e300]", (x,)) for x in loguniform(r, 64, 1e300, 50)]
    c += [("(-260,-1)", (x,)) for x in uniform(r, -260, -1, N)]
    c += [("near trivial zeros", (-2 * r.randint(1, 120) + r.choice((-1, 1)) * math.exp(r.uniform(math.log(1e-12), math.log(1e-2))),)) for _ in range(N)]
    c += [("s<-260", (x,)) for x in loguniform(r, 260, 1e300, 50, -1)]
    return c


def hurwitz_mp(s, q):
    if math.isnan(s) or math.isnan(q):
        return math.nan
    if s == 1:
        return math.inf
    if s < 1:
        return math.nan
    if q == math.inf:
        return 0.0
    if q <= 0 and q == math.floor(q):
        return math.inf
    if q < 0 and s != math.floor(s):
        return math.nan
    with mp.extraprec(80):
        return mp.zeta(mpf(s), mpf(q))


def hurwitz_cases():
    r = rng("zeta_hurwitz")
    c = [("special", a) for a in [(math.nan, 1), (2, math.nan), (1, 2), (0.5, 2), (2, 0), (2, -1), (2.5, -0.5), (2, math.inf), (2, 1), (2, 0.5)]]
    c += [("s in (1,50), q in (0,10)", (s, q)) for s, q in zip(uniform(r, 1, 50, N), loguniform(r, 1e-3, 10, N))]
    c += [("s near 1", (1 + d, q)) for d, q in zip(loguniform(r, 1e-12, 1e-2, N), loguniform(r, 1e-2, 1e3, N))]
    c += [("q tiny", (s, q)) for s, q in zip(uniform(r, 1.01, 30, N), loguniform(r, 1e-300, 1e-5, N))]
    c += [("q large", (s, q)) for s, q in zip(uniform(r, 1.01, 30, N), loguniform(r, 1e5, 1e300, N))]
    c += [("s large", (s, q)) for s, q in zip(loguniform(r, 50, 1e3, N), loguniform(r, 0.5, 100, N))]
    c += [("q<0, integer s", (float(r.randint(2, 30)), q)) for q in uniform(r, -30, 0, N)]
    return c


def polygamma_mp(n, x):
    n = int(n)
    if math.isnan(x) or x == -math.inf:
        return math.nan
    if x == math.inf:
        return 0.0
    if x <= 0 and x == math.floor(x):
        return math.inf if n % 2 == 1 else -math.inf
    with mp.extraprec(80):
        return mp.polygamma(n, mpf(x))


NS = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]


def polygamma_cases():
    r = rng("polygamma")
    c = [("special", (float(n), x)) for n in (1, 2) for x in [math.nan, math.inf, -math.inf, 0.0, -1.0, 1.0, 0.5]]
    c += [("x tiny", (float(r.choice(NS)), x)) for x in loguniform(r, 1e-30, 1e-5, N)]
    c += [("x in (0,10)", (float(r.choice(NS)), x)) for x in uniform(r, 1e-5, 10, N)]
    c += [("x in [10,1e10)", (float(r.choice(NS)), x)) for x in loguniform(r, 10, 1e10, N)]
    c += [("x<0", (float(r.choice(NS)), x)) for x in uniform(r, -20, 0, N)]
    return c


FIXTURES = {
    "zeta": lambda: (1, zeta_cases(), zeta_mp, sc.zeta, "Riemann ζ(s)"),
    "zeta_hurwitz": lambda: (2, hurwitz_cases(), hurwitz_mp, sc.zeta, "Hurwitz ζ(s, q)"),
    "polygamma": lambda: (2, polygamma_cases(), polygamma_mp, lambda n, x: sc.polygamma(int(n), x), "ψ⁽ⁿ⁾(x)"),
}
