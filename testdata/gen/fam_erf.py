"""Error function family fixtures."""

import math

import scipy.special as sc
from mpmath import mp, mpf

from common import SPECIALS, loguniform, rng, uniform

N = 1000


def both(xs):
    return xs + [-x for x in xs]


def ex(fn):
    """Wrap an mpmath function of one argument with exact specials."""
    def f(x):
        if math.isnan(x):
            return math.nan
        return fn(mpf(x))
    return f


# mpmath 1.4.1's erfc overflows internally for x ~ 1e300 (erfc_check_series);
# beyond |x| = 30 the float64 results are exact constants anyway (the
# correction is below 1e-390), so they are returned as such.

def erf_mp(x):
    if math.isinf(x) or abs(x) > 30:
        return math.copysign(1.0, x)
    return ex(mp.erf)(x)


def erfc_mp(x):
    if math.isinf(x) or abs(x) > 30:
        return 0.0 if x > 0 else 2.0
    return ex(mp.erfc)(x)


def erfcx_asym(x):
    """erfcx(x) for x >= 1e4 by its asymptotic series
    1/(x sqrt pi) sum (-1)^k (2k-1)!!/(2x^2)^k, summed while the terms
    decrease (the truncation error is below the first omitted term)."""
    t = mpf(1)
    s = mpf(0)
    k = 0
    while True:
        s += t
        k += 1
        nt = -t * (2 * k - 1) / (2 * x * x)
        if abs(nt) < mpf(2) ** (-mp.prec - 10) * abs(s) or abs(nt) > abs(t):
            break
        t = nt
    return s / (x * mp.sqrt(mp.pi))


def erfcx_mp(x):
    if x == math.inf:
        return 0.0
    if x == -math.inf:
        return math.inf
    if math.isnan(x):
        return math.nan
    if x >= 1e4:
        return erfcx_asym(mpf(x))
    with mp.extraprec(60):
        x = mpf(x)
        return mp.exp(x * x) * mp.erfc(x)


def erf_cases():
    r = rng("erf")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("tiny", (x,)) for x in both(loguniform(r, 1e-300, 1e-5, N // 2))]
    c += [("|x|<=0.5", (x,)) for x in uniform(r, -0.5, 0.5, N)]
    c += [("0.5<|x|<6", (x,)) for x in both(uniform(r, 0.5, 6, N // 2))]
    c += [("|x|>=6", (x,)) for x in both(loguniform(r, 6, 1e300, 50))]
    return c


def erfc_cases():
    r = rng("erfc")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("[-6,0.5)", (x,)) for x in uniform(r, -6, 0.5, N)]
    c += [("[0.5,5)", (x,)) for x in uniform(r, 0.5, 5, N)]
    c += [("[5,26.5) tail", (x,)) for x in uniform(r, 5, 26.5, N)]
    c += [("[26.5,27.3] subnormal", (x,)) for x in uniform(r, 26.5, 27.3, 200)]
    c += [("x<-6", (x,)) for x in loguniform(r, 6, 1e300, 50, -1)]
    c += [("tiny", (x,)) for x in both(loguniform(r, 1e-300, 1e-5, N // 2))]
    return c


def erfcx_cases():
    r = rng("erfcx")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("[-26.6,-0.5)", (x,)) for x in uniform(r, -26.6, -0.5, N)]
    c += [("[-0.5,0.5)", (x,)) for x in uniform(r, -0.5, 0.5, N)]
    c += [("[0.5,8)", (x,)) for x in uniform(r, 0.5, 8, N)]
    c += [("[8,1e5)", (x,)) for x in loguniform(r, 8, 1e5, N)]
    c += [("[1e5,1e300]", (x,)) for x in loguniform(r, 1e5, 1e300, N)]
    return c


def erfinv_mp(y):
    if math.isnan(y) or abs(y) > 1:
        return math.nan
    if abs(y) == 1:
        return math.copysign(math.inf, y)
    if y == 0:
        return y
    # 1-|y| can be ~1e-16: erfinv near 1 needs extra precision.
    with mp.extraprec(80):
        return mp.erfinv(mpf(y))


def erfcinv_mp(q):
    if math.isnan(q) or q < 0 or q > 2:
        return math.nan
    if q == 0:
        return math.inf
    if q == 2:
        return -math.inf
    if q == 1:
        return 0.0
    if q > 1:
        return -erfcinv_mp(2 - q)  # 2-q is exact in float64
    q = mpf(q)
    x0 = float(sc.erfcinv(float(q)))
    if math.isinf(x0):
        x0 = math.sqrt(-math.log(float(q) if float(q) > 0 else 5e-324))
    x0 = mpf(x0)
    with mp.extraprec(40):
        return mp.findroot(lambda t: mp.log(mp.erfc(t)) - mp.log(q), x0)


def erfinv_cases():
    r = rng("erfinv")
    c = [("special", (x,)) for x in [math.nan, 1.0, -1.0, 0.0, -0.0, 1.5, -2.0, 5e-324, 0.5]]
    c += [("tiny", (x,)) for x in both(loguniform(r, 1e-300, 1e-5, N // 2))]
    c += [("|y|<=0.5", (x,)) for x in both(uniform(r, 1e-5, 0.5, N // 2))]
    c += [("0.5<|y|<0.999", (x,)) for x in both(uniform(r, 0.5, 0.999, N // 2))]
    c += [("near ±1", (1 - d,)) for d in loguniform(r, 1.2e-16, 1e-3, N // 2)]
    c += [("near ±1", (-(1 - d),)) for d in loguniform(r, 1.2e-16, 1e-3, N // 2)]
    return c


def erfcinv_cases():
    r = rng("erfcinv")
    c = [("special", (x,)) for x in [math.nan, 0.0, 2.0, 1.0, -0.1, 2.1, 5e-324, 1e-310]]
    c += [("subnormal", (x,)) for x in loguniform(r, 5e-324, 2.2e-308, 200)]
    c += [("[1e-300,1e-10)", (x,)) for x in loguniform(r, 1e-300, 1e-10, N)]
    c += [("[1e-10,0.5)", (x,)) for x in loguniform(r, 1e-10, 0.5, N)]
    c += [("[0.5,1.5]", (x,)) for x in uniform(r, 0.5, 1.5, N)]
    c += [("(1.5,2)", (x,)) for x in uniform(r, 1.5, 2, N)]
    return c


def ndtr_mp(x):
    if math.isnan(x):
        return math.nan
    if math.isinf(x) or abs(x) > 40:
        return 1.0 if x > 0 else 0.0
    return mp.ncdf(mpf(x))


def ndtr_cases():
    r = rng("ndtr")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("[-38.5,-10)", (x,)) for x in uniform(r, -38.5, -10, N)]
    c += [("[-10,-1)", (x,)) for x in uniform(r, -10, -1, N)]
    c += [("[-1,1)", (x,)) for x in uniform(r, -1, 1, N)]
    c += [("[1,8.3)", (x,)) for x in uniform(r, 1, 8.3, N)]
    c += [("x>=8.3", (x,)) for x in loguniform(r, 8.3, 1e300, 50)]
    return c


def ndtri_mp(p):
    if math.isnan(p) or p < 0 or p > 1:
        return math.nan
    if p == 0:
        return -math.inf
    if p == 1:
        return math.inf
    if p > 0.5:
        return -ndtri_mp(1 - p)  # exact in float64
    with mp.extraprec(40):
        return -mp.sqrt(2) * erfcinv_mp_exact(2 * mpf(p))


def erfcinv_mp_exact(q):
    x0 = float(sc.erfcinv(float(q))) if float(q) > 0 else 27.0
    if math.isinf(x0):
        x0 = math.sqrt(-math.log(float(q)))
    x0 = mpf(x0)
    return mp.findroot(lambda t: mp.log(mp.erfc(t)) - mp.log(q), x0)


def ndtri_cases():
    r = rng("ndtri")
    c = [("special", (x,)) for x in [math.nan, 0.0, 1.0, 0.5, -0.1, 1.1, 5e-324]]
    c += [("subnormal", (x,)) for x in loguniform(r, 5e-324, 2.2e-308, 200)]
    c += [("[1e-300,1e-10)", (x,)) for x in loguniform(r, 1e-300, 1e-10, N)]
    c += [("[1e-10,0.5)", (x,)) for x in loguniform(r, 1e-10, 0.5, N)]
    c += [("[0.5,1)", (x,)) for x in uniform(r, 0.5, 1, N)]
    c += [("near 1", (1 - d,)) for d in loguniform(r, 1.2e-16, 1e-3, N)]
    return c


def log_ndtr_mp(x):
    if math.isnan(x):
        return math.nan
    if x == math.inf:
        return 0.0
    if x == -math.inf:
        return -math.inf
    if x > 40:
        # log1p(-Φ(-x)) ~ -Φ(-x) < 1e-350: rounds to -0.
        return -0.0
    x = mpf(x)
    if x > 0:
        with mp.extraprec(60):
            return mp.log1p(-mp.ncdf(-x))
    if x < -1e4:
        # log Φ(x) = -x²/2 - log(-x) - log sqrt(2π) + log(sqrt(2)... ) with
        # Φ(x) = erfc(z)/2 = exp(-z²) erfcx(z)/2, z = -x/sqrt 2.
        z = -x / mp.sqrt(2)
        return -z * z + mp.log(erfcx_asym(z) / 2)
    return mp.log(mp.ncdf(x))


def log_ndtr_cases():
    r = rng("log_ndtr")
    c = [("special", (x,)) for x in SPECIALS]
    c += [("x<-1e5", (x,)) for x in loguniform(r, 1e5, 1e200, N, -1)]
    c += [("[-1e5,-40)", (x,)) for x in loguniform(r, 40, 1e5, N, -1)]
    c += [("[-40,-1)", (x,)) for x in uniform(r, -40, -1, N)]
    c += [("[-1,1)", (x,)) for x in uniform(r, -1, 1, N)]
    c += [("[1,10)", (x,)) for x in uniform(r, 1, 10, N)]
    c += [("[10,38)", (x,)) for x in uniform(r, 10, 38, N)]
    return c


FIXTURES = {
    "erf": lambda: (1, erf_cases(), erf_mp, sc.erf, "erf(x)"),
    "erfc": lambda: (1, erfc_cases(), erfc_mp, sc.erfc, "erfc(x)"),
    "erfcx": lambda: (1, erfcx_cases(), erfcx_mp, sc.erfcx, "erfcx(x) = exp(x²) erfc(x)"),
    "erfinv": lambda: (1, erfinv_cases(), erfinv_mp, sc.erfinv, "erfinv(y)"),
    "erfcinv": lambda: (1, erfcinv_cases(), erfcinv_mp, sc.erfcinv, "erfcinv(q)"),
    "ndtr": lambda: (1, ndtr_cases(), ndtr_mp, sc.ndtr, "Φ(x)"),
    "ndtri": lambda: (1, ndtri_cases(), ndtri_mp, sc.ndtri, "Φ⁻¹(p)"),
    "log_ndtr": lambda: (1, log_ndtr_cases(), log_ndtr_mp, sc.log_ndtr, "log Φ(x)"),
}
