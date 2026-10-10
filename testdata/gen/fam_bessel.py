"""Bessel family fixtures (real order, real argument) and the constant tables
of bessel_tables.go.

    uv run --with mpmath==1.4.1 --with scipy==1.18.1 --with numpy==2.5.4 \
        python testdata/gen/gen.py j0 j1 ...          # fixtures
    uv run --with mpmath==1.4.1 python testdata/gen/fam_bessel.py tables > bessel_tables.go

Conventions of the oracle (they are the conventions of the Go functions):
J and Y tend to 0 as x -> +inf, so J(v, +inf) = Y(v, +inf) = 0 (SciPy returns
NaN there); I(v, +inf) = +inf, K(v, +inf) = 0 and the scaled forms tend to 0;
values at x = 0 are the limits (signed infinities with the sign of the
leading term of the small-x expansion). Non-integer order with x < 0 is NaN,
integer order follows the parity of the function.
"""

import math
import sys

import mpmath
from mpmath import mp, mpf

N = 1000


# ------------------------------------------------------------------ helpers

def isint(v):
    return math.isfinite(v) and v == math.floor(v)


def near(r, centers, n, lo=1e-12, hi=1e-3):
    out = []
    for _ in range(n):
        c = r.choice(centers)
        d = math.exp(r.uniform(math.log(lo), math.log(hi)))
        out.append(c * (1 + r.choice((-1, 1)) * d))
    return out


def zeros_of(kind, n, count):
    with mp.workprec(80):
        f = mp.besseljzero if kind == "J" else mp.besselyzero
        return [float(f(n, k)) for k in range(1, count + 1)]


def xp(x):
    """Extra precision for an argument of size |x|: mpmath's series for
    Bessel functions cancel like e^|x|, and the reduction of a huge x needs
    its exponent in bits."""
    ax = abs(x)
    if not math.isfinite(ax) or ax < 1:
        return 20
    return 20 + int(math.log2(ax)) + 10


def sign_gamma(t):
    """sign of Gamma(t), t not a non-positive integer."""
    if t > 0:
        return 1
    return 1 if math.floor(t) % 2 == 0 else -1


# --------------------------------------------------------------- oracles J/Y

def jv_mp(v, x):
    if math.isnan(v) or math.isnan(x):
        return math.nan
    if math.isinf(v):
        if v > 0 and math.isfinite(x) and (x >= 0 or True):
            return 0.0
        return math.nan
    if x < 0 and not isint(v):
        return math.nan
    if math.isinf(x):
        return 0.0
    if x == 0:
        if v == 0:
            return 1.0
        if v > 0 or isint(v):
            return 0.0
        # J_v(x) ~ (x/2)^v / Gamma(v+1)
        return math.inf * sign_gamma(v + 1)
    with mp.extraprec(xp(x) + int(abs(v) * 0.1)):
        return mp.besselj(mpf(v), mpf(x))


def yv_mp(v, x):
    if math.isnan(v) or math.isnan(x):
        return math.nan
    if math.isinf(v):
        if v > 0 and x > 0 and math.isfinite(x):
            return -math.inf
        if v > 0 and x == 0:
            return -math.inf
        return math.nan
    if x < 0:
        return math.nan
    if math.isinf(x):
        return 0.0
    if x == 0:
        if isint(v):
            n = int(abs(v))
            return -math.inf if (v >= 0 or n % 2 == 0) else math.inf
        if v > 0:
            return -math.inf
        # Y_{-nu} = cos(nu pi) Y_nu + sin(nu pi) J_nu, nu = -v > 0
        nu = -v
        c = float(mp.cospi(mpf(nu)))
        if c == 0:
            return 0.0
        return -math.inf if c > 0 else math.inf
    with mp.extraprec(xp(x) + int(abs(v) * 0.1)):
        return mp.bessely(mpf(v), mpf(x))


def jn_mp(n, x):
    return jv_mp(n, x)


def yn_mp(n, x):
    return yv_mp(n, x)


# --------------------------------------------------------------- oracles I/K

def iv_mp(v, x, scaled=False):
    if math.isnan(v) or math.isnan(x):
        return math.nan
    if math.isinf(v):
        if v > 0 and math.isfinite(x):
            return 0.0
        return math.nan
    if x < 0 and not isint(v):
        return math.nan
    if math.isinf(x):
        if scaled:
            return 0.0 if x > 0 or int(v) % 2 == 0 else -0.0
        return math.inf if (x > 0 or int(v) % 2 == 0) else -math.inf
    if x == 0:
        if v == 0:
            return 1.0
        if v > 0 or isint(v):
            return 0.0
        return math.inf * sign_gamma(v + 1)
    with mp.extraprec(xp(x) + int(abs(v) * 0.1)):
        X = mpf(x)
        r = mp.besseli(mpf(v), X)
        if scaled:
            r = r * mp.exp(-abs(X))
        return r


def kv_mp(v, x, scaled=False):
    if math.isnan(v) or math.isnan(x):
        return math.nan
    if x < 0:
        return math.nan
    if math.isinf(v):
        return math.inf if math.isfinite(x) else math.nan
    if math.isinf(x):
        return 0.0
    if x == 0:
        return math.inf
    with mp.extraprec(xp(x) + int(abs(v) * 0.1)):
        X = mpf(x)
        r = mp.besselk(mpf(v), X)
        if scaled:
            r = r * mp.exp(X)
        return r



# --------------------------------------------- large-x oracles (Hankel, exact)

def _hankel_terms(v, X):
    """a_k(v)/X^k of DLMF 10.17.1 until below the working precision; None
    if they start growing first (then the caller uses mpmath directly)."""
    mu = 4 * mpf(v) ** 2
    eps = mpf(2) ** (-mp.prec - 10)
    terms = [mpf(1)]
    t = mpf(1)
    k = 1
    while True:
        t = t * (mu - (2 * k - 1) ** 2) / (k * 8 * X)
        if abs(t) < eps:
            return terms
        if k > 2 and abs(t) > abs(terms[-1]):
            return None
        terms.append(t)
        k += 1
        if k > 400:
            return None


def jy_large(v, x, which):
    X = mpf(x)
    terms = _hankel_terms(v, X)
    if terms is None:
        return None
    P = sum(t * (-1) ** (k // 2) for k, t in enumerate(terms) if k % 2 == 0)
    Q = sum(t * (-1) ** (k // 2) for k, t in enumerate(terms) if k % 2 == 1)
    chi = X - (mpf(v) / 2 + mpf(1) / 4) * mp.pi
    a = mp.sqrt(2 / (mp.pi * X))
    if which == "J":
        return a * (P * mp.cos(chi) - Q * mp.sin(chi))
    return a * (P * mp.sin(chi) + Q * mp.cos(chi))


def ik_large_scaled(v, x, which):
    X = mpf(x)
    terms = _hankel_terms(v, X)
    if terms is None:
        return None
    if which == "I":
        return sum(t * (-1) ** k for k, t in enumerate(terms)) / mp.sqrt(2 * mp.pi * X)
    return sum(terms) * mp.sqrt(mp.pi / (2 * X))


def jv_any(v, x):
    if math.isfinite(x) and math.isfinite(v) and abs(x) >= 1e4 and not math.isnan(v):
        if x > 0:
            r = jy_large(v, x, "J")
            if r is not None:
                return r
        elif isint(v):
            r = jy_large(v, -x, "J")
            if r is not None:
                return r if int(v) % 2 == 0 else -r
    return jv_mp(v, x)


def yv_any(v, x):
    if math.isfinite(x) and math.isfinite(v) and x >= 1e4:
        r = jy_large(v, x, "Y")
        if r is not None:
            return r
    return yv_mp(v, x)


def ive_any(v, x):
    if math.isfinite(x) and math.isfinite(v) and abs(x) >= 1e4 and (x > 0 or isint(v)):
        r = ik_large_scaled(v, abs(x), "I")
        if r is not None:
            return r if (x > 0 or int(v) % 2 == 0) else -r
    return iv_mp(v, x, True)


def iv_any(v, x):
    if math.isfinite(x) and abs(x) >= 1e4 and math.isfinite(v) and (x > 0 or isint(v)):
        return math.inf if (x > 0 or int(v) % 2 == 0) else -math.inf
    return iv_mp(v, x)


def kve_any(v, x):
    if math.isfinite(x) and math.isfinite(v) and x >= 1e4:
        r = ik_large_scaled(v, x, "K")
        if r is not None:
            return r
    return kv_mp(v, x, True)


def kv_any(v, x):
    if math.isfinite(x) and x >= 1e4 and math.isfinite(v):
        return 0.0
    return kv_mp(v, x)


# ------------------------------------------------------------------- cases

def specials_1():
    return [math.nan, math.inf, -math.inf, 0.0, -0.0, 5e-324, 1e-310, 2.2250738585072014e-308,
            1.0, -1.0, 8.0, 100.0, 1.7976931348623157e308]


V_SPECIAL = [0.0, 1.0, 2.0, 3.0, -1.0, -2.0, 0.5, -0.5, 1.5, -1.5, 2.3, -2.3, 100.0, -100.5,
             math.inf, -math.inf, math.nan]
X_SPECIAL = [0.0, -0.0, math.inf, -math.inf, math.nan, -1.0, -2.5, 1e-310, 1.0, 30.0, 1e5]


def one_arg_cases(name, kind):
    """kind: 'J' (J0/J1: even/odd), 'Y', 'I', 'K', 'Ie', 'Ke'."""
    from common import loguniform, rng, uniform
    r = rng(name)
    c = [("special", (x,)) for x in specials_1()]
    c += [("small x", (x,)) for x in loguniform(r, 1e-300, 1e-3, N)]
    c += [("moderate", (x,)) for x in uniform(r, 1e-3, 25, N)]
    if kind in ("J", "Y"):
        c += [("large x", (x,)) for x in loguniform(r, 25, 1e4, N)]
        c += [("huge x", (x,)) for x in loguniform(r, 1e4, 1e300, N)]
        n = 0 if name.endswith("0") else 1
        zs = zeros_of(kind, n, 30)
        c += [("near zeros", (x,)) for x in near(r, zs, N)]
        if kind == "J":
            c += [("negative x", (x,)) for x in loguniform(r, 1e-3, 1e4, N, -1)]
    else:
        c += [("large x", (x,)) for x in uniform(r, 25, 700, N)]
        if kind in ("I", "K"):
            c += [("over/underflow edge", (x,)) for x in uniform(r, 700, 750, N)]
        c += [("huge x", (x,)) for x in loguniform(r, 750, 1e300, N)]
        if kind in ("I", "Ie"):
            c += [("negative x", (x,)) for x in loguniform(r, 1e-3, 750, N, -1)]
    return c


def two_arg_cases(name, kind, integer=False):
    from common import loguniform, rng, uniform
    r = rng(name)
    c = [("special", (v, x)) for v in V_SPECIAL for x in X_SPECIAL
         if not (integer and not isint(v))]

    def order(lo, hi):
        v = r.uniform(lo, hi)
        return float(round(v)) if integer else v

    def add(region, n, vlo, vhi, xs):
        for x in xs:
            c.append((region, (order(vlo, vhi), x)))

    add("small x", N, 0, 10, loguniform(r, 1e-300, 1e-3, N))
    add("moderate", N, 0, 20, uniform(r, 1e-3, 25, N))
    if kind in ("J", "Y"):
        add("large x", N, 0, 50, loguniform(r, 25, 1e4, N))
        add("huge x", N, 0, 20, loguniform(r, 1e4, 1e300, N))
        # near the zeros of a few orders
        orders = [0, 1, 2, 5, 10] if integer else [0.25, 1.5, 2.7, 7.3, 0.5]
        for v in orders:
            zs = zeros_of(kind, v, 30)
            for x in near(r, zs, N // len(orders)):
                c.append(("near zeros", (float(v), x)))
        if kind == "J":
            for _ in range(N // 2 if integer else 0):
                c.append(("negative x", (order(0, 30), -math.exp(r.uniform(math.log(1e-3), math.log(1e3))))))
    else:
        add("large x", N, 0, 50, uniform(r, 25, 700, N))
        add("over/underflow edge", N, 0, 5, uniform(r, 700, 750, N))
        if kind in ("Ie", "Ke"):
            add("huge x", N, 0, 20, loguniform(r, 750, 1e300, N))
    # large order, including the transition region x ~ v
    for _ in range(N):
        v = math.exp(r.uniform(math.log(50), math.log(1000)))
        v = float(round(v)) if integer else v
        x = v * math.exp(r.uniform(math.log(0.1), math.log(3)))
        c.append(("large order", (v, x)))
    for _ in range(N // 2):
        v = math.exp(r.uniform(math.log(50), math.log(1000)))
        v = float(round(v)) if integer else v
        x = v + r.uniform(-3, 3) * v ** (1 / 3)
        c.append(("v ~ x", (v, x)))
    if not integer:
        for _ in range(N):
            v = -r.uniform(0, 20)
            if isint(v):
                continue
            c.append(("negative v", (v, math.exp(r.uniform(math.log(1e-3), math.log(100))))))
        for _ in range(N):
            v = float(r.randint(-30, 30))
            c.append(("integer v", (v, math.exp(r.uniform(math.log(1e-3), math.log(100))))))
    else:
        for _ in range(N):
            v = float(r.randint(-30, -1))
            c.append(("negative n", (v, math.exp(r.uniform(math.log(1e-3), math.log(100))))))
    return c


def _sp():
    import scipy.special as sc
    return sc


def fixtures():
    sc = _sp()
    return {
        "j0": lambda: (1, one_arg_cases("j0", "J"), lambda x: jv_any(0.0, x), sc.j0, "J0(x)"),
        "j1": lambda: (1, one_arg_cases("j1", "J"), lambda x: jv_any(1.0, x), sc.j1, "J1(x)"),
        "y0": lambda: (1, one_arg_cases("y0", "Y"), lambda x: yv_any(0.0, x), sc.y0, "Y0(x)"),
        "y1": lambda: (1, one_arg_cases("y1", "Y"), lambda x: yv_any(1.0, x), sc.y1, "Y1(x)"),
        "i0": lambda: (1, one_arg_cases("i0", "I"), lambda x: iv_any(0.0, x), sc.i0, "I0(x)"),
        "i1": lambda: (1, one_arg_cases("i1", "I"), lambda x: iv_any(1.0, x), sc.i1, "I1(x)"),
        "k0": lambda: (1, one_arg_cases("k0", "K"), lambda x: kv_any(0.0, x), sc.k0, "K0(x)"),
        "k1": lambda: (1, one_arg_cases("k1", "K"), lambda x: kv_any(1.0, x), sc.k1, "K1(x)"),
        "i0e": lambda: (1, one_arg_cases("i0e", "Ie"), lambda x: ive_any(0.0, x), sc.i0e, "I0(x) e^-|x|"),
        "i1e": lambda: (1, one_arg_cases("i1e", "Ie"), lambda x: ive_any(1.0, x), sc.i1e, "I1(x) e^-|x|"),
        "k0e": lambda: (1, one_arg_cases("k0e", "Ke"), lambda x: kve_any(0.0, x), sc.k0e, "K0(x) e^x"),
        "k1e": lambda: (1, one_arg_cases("k1e", "Ke"), lambda x: kve_any(1.0, x), sc.k1e, "K1(x) e^x"),
        "jv": lambda: (2, two_arg_cases("jv", "J"), jv_any, sc.jv, "J_v(x)"),
        "yv": lambda: (2, two_arg_cases("yv", "Y"), yv_any, sc.yv, "Y_v(x)"),
        "jn": lambda: (2, two_arg_cases("jn", "J", True), jv_any, sc.jv, "J_n(x), n = args[0]"),
        "yn": lambda: (2, two_arg_cases("yn", "Y", True), yv_any,
                       lambda n, x: sc.yn(int(n), x), "Y_n(x), n = args[0]"),
        "iv": lambda: (2, two_arg_cases("iv", "I"), iv_any, sc.iv, "I_v(x)"),
        "kv": lambda: (2, two_arg_cases("kv", "K"), kv_any, sc.kv, "K_v(x)"),
        "ive": lambda: (2, two_arg_cases("ive", "Ie"), ive_any, sc.ive, "I_v(x) e^-|x|"),
        "kve": lambda: (2, two_arg_cases("kve", "Ke"), kve_any, sc.kve, "K_v(x) e^x"),
    }


class _Lazy(dict):
    """FIXTURES without importing SciPy until a builder runs."""

    def __init__(self, names):
        super().__init__((n, None) for n in names)

    def __getitem__(self, k):
        return fixtures()[k]


FIXTURES = _Lazy(["j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1", "i0e", "i1e", "k0e", "k1e",
                  "jv", "yv", "jn", "yn", "iv", "kv", "ive", "kve"])


# ------------------------------------------------------------- tables (Go)

def dd(v):
    hi = float(v)
    lo = float(v - mpf(hi))
    return hi, lo


def tables():
    out = []
    w = out.append
    w("// Code generated by testdata/gen/fam_bessel.py tables; DO NOT EDIT.")
    w("")
    w("package special")
    w("")
    with mp.workprec(3000):
        f = 2 / mp.pi
        words = []
        for _ in range(24):
            f *= mpf(2) ** 64
            k = int(mp.floor(f))
            f -= k
            words.append(k)
    w("// besselTwoOverPiBits holds the binary digits of 2/pi after the point,")
    w("// 64 per word, most significant first (for Payne-Hanek reduction).")
    w("var besselTwoOverPiBits = [...]uint64{")
    for i in range(0, 24, 4):
        w("\t" + ", ".join("0x%016x" % x for x in words[i:i + 4]) + ",")
    w("}")
    w("")
    with mp.workprec(300):
        cs = mp.taylor(mp.rgamma, 0, 32)
    w("// besselRGammaTaylor[k-1] = c_k, the Taylor coefficients of 1/Gamma(z) =")
    w("// sum c_k z^k (Abramowitz & Stegun 6.1.34), to double precision.")
    w("var besselRGammaTaylor = [...]float64{")
    for k in range(1, 33):
        w("\t%r," % float(cs[k]))
    w("}")
    w("")
    with mp.workprec(300):
        for kind, n in (("J", 0), ("J", 1), ("Y", 0), ("Y", 1)):
            w("// bessel%s%dZeros: the zeros z of %s_%d in [5, 104] and the derivative" % (kind, n, kind, n))
            w("// d there, each as a double-double: {z.hi, z.lo, d.hi, d.lo}.")
            w("var bessel%s%dZeros = [...][4]float64{" % (kind, n))
            k = 1
            while True:
                z = mp.besseljzero(n, k) if kind == "J" else mp.besselyzero(n, k)
                if z > 104:
                    break
                if z >= 5:
                    d = mp.besselj(n, z, derivative=1) if kind == "J" else mp.bessely(n, z, derivative=1)
                    zh, zl = dd(z)
                    dh, dl = dd(d)
                    w("\t{%r, %r, %r, %r}," % (zh, zl, dh, dl))
                k += 1
            w("}")
            w("")
    sys.stdout.write("\n".join(out))


if __name__ == "__main__" and sys.argv[1:] == ["tables"]:
    tables()
