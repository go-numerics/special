"""Gamma family fixtures: gamma, loggamma, digamma, beta, logbeta."""

import math

import mpmath
import scipy.special as sc
from mpmath import mp, mpf

from common import SPECIALS, loguniform, rng, uniform

N = 1000


def near(r, centers, n, lo=1e-16, hi=1e-2):
    out = []
    for _ in range(n):
        c = r.choice(centers)
        d = math.exp(r.uniform(math.log(lo), math.log(hi)))
        out.append(c + r.choice((-1, 1)) * d)
    return out


def gamma_mp(x):
    if math.isnan(x) or x == -math.inf:
        return math.nan
    if x == math.inf:
        return math.inf
    if x == 0:
        return math.copysign(math.inf, x)
    if x < 0 and x == math.floor(x):
        return math.nan
    return mp.gamma(mpf(x))


def gamma_cases():
    r = rng("gamma")
    c = [("special", (x,)) for x in SPECIALS + [-1.0, -2.0, -170.0, 1.0, 2.0, 0.5]]
    c += [("tiny", (x,)) for x in loguniform(r, 1e-300, 1e-5, N)]
    c += [("tiny", (x,)) for x in loguniform(r, 1e-300, 1e-5, N, -1)]
    c += [("(0,2)", (x,)) for x in uniform(r, 1e-5, 2, N)]
    c += [("[2,33)", (x,)) for x in uniform(r, 2, 33, N)]
    c += [("[33,171.6]", (x,)) for x in uniform(r, 33, 171.62, N)]
    c += [("integers", (float(n),)) for n in range(1, 172)]
    c += [("overflow", (x,)) for x in uniform(r, 171.63, 200, 20) + [1e10, 1e300]]
    c += [("(-33,0)", (x,)) for x in uniform(r, -33, 0, N)]
    c += [("near poles", (x,)) for x in near(r, [-float(k) for k in range(1, 34)], N)]
    c += [("(-184,-33]", (x,)) for x in uniform(r, -184, -33, N)]
    return c


def loggamma_mp(x):
    if math.isnan(x):
        return math.nan
    if math.isinf(x):
        return math.inf
    if x <= 0 and x == math.floor(x):
        return math.inf
    return mp.log(abs(mp.gamma(mpf(x)))) if abs(x) < 1e5 else mp.loggamma(mpf(x)).real


def loggamma_cases():
    r = rng("loggamma")
    c = [("special", (x,)) for x in SPECIALS + [-1.0, -2.0, 1.0, 2.0, 0.5]]
    c += [("tiny", (x,)) for x in loguniform(r, 1e-300, 1e-5, N) + loguniform(r, 1e-300, 1e-5, N, -1)]
    c += [("near 1 and 2", (x,)) for x in near(r, [1.0, 2.0], N, 1e-16, 0.2)]
    c += [("(0,3)", (x,)) for x in uniform(r, 1e-5, 3, N)]
    c += [("[3,13)", (x,)) for x in uniform(r, 3, 13, N)]
    c += [("[13,1e5)", (x,)) for x in loguniform(r, 13, 1e5, N)]
    c += [("[1e5,1e305]", (x,)) for x in loguniform(r, 1e5, 1e305, N)]
    c += [("(-34,0)", (x,)) for x in uniform(r, -34, 0, N)]
    c += [("near poles", (x,)) for x in near(r, [-float(k) for k in range(0, 34)], N)]
    c += [("(-1e5,-34]", (x,)) for x in loguniform(r, 34, 1e5, N, -1)]
    return c


# zeros of digamma: positive x0 and the negative roots
def digamma_mp(x):
    if math.isnan(x) or x == -math.inf:
        return math.nan
    if x == math.inf:
        return math.inf
    if x == 0:
        return -math.copysign(math.inf, x)
    if x < 0 and x == math.floor(x):
        return math.nan
    return mp.digamma(mpf(x))


def digamma_cases():
    r = rng("digamma")
    with mp.workprec(80):
        roots = [float(mp.findroot(mp.digamma, 1.46))]
        roots += [float(mp.findroot(mp.digamma, -k + 0.3 + 0.2 * (k > 1))) for k in range(0, 1)]
    with mp.workprec(80):
        neg = []
        for k in range(0, 40):
            a, b = mpf(-k - 1) + mpf(1e-12), mpf(-k) - mpf(1e-12)
            neg.append(float(mp.findroot(mp.digamma, (a, b), solver="anderson")))
    c = [("special", (x,)) for x in SPECIALS + [-1.0, -2.0, 1.0, 2.0]]
    c += [("tiny", (x,)) for x in loguniform(r, 1e-300, 1e-5, N) + loguniform(r, 1e-300, 1e-5, N, -1)]
    c += [("(0,10)", (x,)) for x in uniform(r, 1e-5, 10, N)]
    c += [("near positive root", (x,)) for x in near(r, roots[:1], N, 1e-16, 1e-2)]
    c += [("[10,1e300]", (x,)) for x in loguniform(r, 10, 1e300, N)]
    c += [("(-40,0)", (x,)) for x in uniform(r, -40, 0, N)]
    c += [("near negative roots", (x,)) for x in near(r, neg, N, 1e-16, 1e-3)]
    c += [("(-1e15,-40]", (x,)) for x in loguniform(r, 40, 1e15, N, -1)]
    return c


def beta_mp(a, b):
    if math.isnan(a) or math.isnan(b):
        return math.nan
    # a+b must be formed exactly and log-gamma differences of huge
    # arguments cancel by up to ~1000 bits: work with that much extra.
    try:
        with mp.extraprec(1200):
            a, b = mpf(a), mpf(b)
            if a > 0 and b > 0:
                return mp.exp(mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(mp.fadd(a, b, exact=True)))
            return mp.beta(a, b)
    except (ZeroDivisionError, ValueError):
        return math.nan


def beta_cases():
    r = rng("beta")
    c = [("special", (a, b)) for a, b in [(math.nan, 1), (1, math.nan), (1, 1), (0.5, 0.5), (1e-300, 1), (2, 3)]]
    pos = lambda n, lo, hi: loguniform(r, lo, hi, n)
    c += [("a,b in (0,1]", (a, b)) for a, b in zip(pos(N, 1e-10, 1), pos(N, 1e-10, 1))]
    c += [("a,b in (1,100]", (a, b)) for a, b in zip(pos(N, 1, 100), pos(N, 1, 100))]
    c += [("a,b in (100,1e5]", (a, b)) for a, b in zip(pos(N, 100, 1e5), pos(N, 100, 1e5))]
    c += [("a small, b large", (a, b)) for a, b in zip(pos(N, 1e-10, 1), pos(N, 1e3, 1e15))]
    c += [("a<0 non-integer", (a, b)) for a, b in zip(uniform(r, -30, 0, N), pos(N, 0.1, 30))]
    return c


def logbeta_mp(a, b):
    v = beta_mp(a, b)
    if isinstance(v, float):
        return v
    if v == 0:
        return -math.inf
    with mp.extraprec(1200):
        if a > 0 and b > 0:
            a, b = mpf(a), mpf(b)
            return mp.loggamma(a) + mp.loggamma(b) - mp.loggamma(mp.fadd(a, b, exact=True))
        return mp.log(abs(v))


def logbeta_cases():
    r = rng("logbeta")
    pos = lambda n, lo, hi: loguniform(r, lo, hi, n)
    c = [("special", (a, b)) for a, b in [(math.nan, 1), (1, 1), (0.5, 0.5), (2, 3)]]
    c += [("a,b in (0,1]", (a, b)) for a, b in zip(pos(N, 1e-300, 1), pos(N, 1e-300, 1))]
    c += [("a,b in (1,100]", (a, b)) for a, b in zip(pos(N, 1, 100), pos(N, 1, 100))]
    c += [("a,b in (100,1e300]", (a, b)) for a, b in zip(pos(N, 100, 1e300), pos(N, 100, 1e300))]
    c += [("a small, b large", (a, b)) for a, b in zip(pos(N, 1e-10, 1), pos(N, 1e3, 1e300))]
    c += [("near log B = 0", (a, b)) for a, b in zip(uniform(r, 0.9, 1.1, N), uniform(r, 0.9, 1.1, N))]
    return c


FIXTURES = {
    "gamma": lambda: (1, gamma_cases(), gamma_mp, sc.gamma, "Γ(x)"),
    "loggamma": lambda: (1, loggamma_cases(), loggamma_mp, sc.gammaln, "log|Γ(x)|"),
    "digamma": lambda: (1, digamma_cases(), digamma_mp, sc.digamma, "ψ(x)"),
    "beta": lambda: (2, beta_cases(), beta_mp, sc.beta, "B(a,b)"),
    "logbeta": lambda: (2, logbeta_cases(), logbeta_mp, sc.betaln, "log|B(a,b)|"),
}
