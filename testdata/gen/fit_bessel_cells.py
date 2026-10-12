"""J0, J1, Y0, Y1 between the power series near 0 and x = 100, as one
polynomial per cell.

    uv run --with mpmath==1.4.1 python testdata/gen/fit_bessel_cells.py > zz_besselCells_coeffs.go

Cells. The expansion points z are the zeros of the function, the zeros of
its derivative (Harrison, "Fast and accurate Bessel function computation",
ARITH 19, 2009) and the midpoints between consecutive ones; a cell runs
from the midpoint with its left neighbour to the midpoint with its right
one. A cell where the fit below does not reach the tolerance is split by
adding the midpoints of its two halves as points (this happens only for Y
near its logarithmic singularity at 0).

Polynomial. With h = x - z,

    f(z + h) = a0 + a1 h + h^2 R(h),

a0 = f(z) and a1 = f'(z) as double-doubles (a0 = 0 at a zero of f, whose
z is then a double-double too so that h = x - z keeps full relative
accuracy next to the zero), and R interpolated at Chebyshev nodes on the
cell at 200 bits with the fixed degree DEG, re-expanded in powers of h. The
tolerance is on the function, |h^2 dR(h)| <= TOL |f(z + h)|, checked on 81
points per cell; the worst ratio over all cells is printed in the header.
"""

import mpmath
from mpmath import mp, mpf

mp.prec = 200
DEG = 11
TOL = mpf(2) ** -62
XMAX = 100


def fns(kind, n):
    if kind == "J":
        f = lambda x: mp.besselj(n, x)
        df = lambda x: mp.besselj(n, x, derivative=1)
        zf = lambda k: mp.besseljzero(n, k)
        zd = lambda k: mp.besseljzero(n, k, derivative=1)
    else:
        f = lambda x: mp.bessely(n, x)
        df = lambda x: mp.bessely(n, x, derivative=1)
        zf = lambda k: mp.besselyzero(n, k)
        zd = lambda k: mp.besselyzero(n, k, derivative=1)
    return f, df, zf, zd


def special_points(zf, zd):
    pts = []
    for gen, isz in ((zf, True), (zd, False)):
        k = 1
        while True:
            z = gen(k)
            if z > XMAX + 4:
                break
            pts.append((z, isz))
            k += 1
    pts.sort()
    return pts


def d2(f, df, n, x):
    # Bessel's equation: f'' = -f'/x - (1 - n^2/x^2) f
    return -df(x) / x - (1 - mpf(n) ** 2 / x**2) * f(x)


def cheb_h(R, hl, hr, deg):
    """Chebyshev interpolant of R on [hl, hr], as monomials in h."""
    c, w = (hl + hr) / 2, (hr - hl) / 2
    N = deg + 1
    nodes = [mp.cos(mp.pi * (k + mpf(1) / 2) / N) for k in range(N)]
    vals = [R(c + w * t) for t in nodes]
    cc = []
    for j in range(N):
        s = mp.fsum(vals[k] * mp.cos(mp.pi * j * (k + mpf(1) / 2) / N) for k in range(N))
        cc.append(2 * s / N)
    cc[0] /= 2
    # Chebyshev -> monomials in t = (h - c)/w
    mono = [mpf(0)] * N
    Tp, Tc = [mpf(1)], [mpf(0), mpf(1)]
    for j in range(N):
        if j == 0:
            T = Tp
        elif j == 1:
            T = Tc
        else:
            T = [mpf(0)] + [2 * v for v in Tc]
            for i, v in enumerate(Tp):
                T[i] -= v
            Tp, Tc = Tc, T
        for i, v in enumerate(T):
            mono[i] += cc[j] * v
    # t = (h - c)/w: sum mono[i] ((h - c)/w)^i, expanded in powers of h
    out = [mpf(0)] * N
    for i, v in enumerate(mono):
        for k in range(i + 1):
            out[k] += v * mp.binomial(i, k) * (-c) ** (i - k) / w**i
    return out


def cell(f, df, n, z, isz, l, r):
    a0 = mpf(0) if isz else f(z)
    a1 = df(z)
    half = (mp.diff(f, z, 2) if z == 0 else d2(f, df, n, z)) / 2

    def R(h):
        if h == 0:
            return half
        with mp.workprec(mp.prec + 100):
            return (f(z + h) - a0 - a1 * h) / h**2

    co = cheb_h(R, l - z, r - z, DEG)
    worst = mpf(0)
    for k in range(81):
        x = l + (r - l) * k / 80
        h = x - z
        if h == 0:
            continue
        p = mp.fsum(co[i] * h**i for i in range(len(co)))
        fv = f(x)
        if fv == 0:
            continue
        worst = max(worst, abs(h * h * (p - R(h)) / fv))
    return a0, a1, co, worst


def build(kind, n, xmin):
    f, df, zf, zd = fns(kind, n)
    # zeros of f keep their full precision (z is emitted as a
    # double-double); every other point is a float64, so that h = x - z is
    # exact and no low part of z has to be carried through the derivative.
    sp = [(z if isz else mpf(float(z)), isz) for z, isz in special_points(zf, zd)]
    if kind == "J":
        sp = [(mpf(0), n == 1)] + sp  # J0'(0) = 0, J1(0) = 0
    pts = []
    for (a, ia), (b, ib) in zip(sp, sp[1:]):
        pts.append((a, ia))
        pts.append((mpf(float((a + b) / 2)), False))
    pts.append(sp[-1])
    cache = {}
    while True:
        out, split = [], None
        for i, (z, isz) in enumerate(pts):
            l = mpf(xmin) if i == 0 else mpf(float((pts[i - 1][0] + z) / 2))
            r = mpf(float((z + pts[i + 1][0]) / 2)) if i + 1 < len(pts) else mpf(XMAX)
            l, r = max(l, mpf(xmin)), min(r, mpf(XMAX))
            if r <= xmin or l >= XMAX:
                continue
            assert z == 0 or (l >= z / 2 and r <= 2 * z), (z, l, r)  # h = x - z exact
            key = (z, l, r)
            if key not in cache:
                cache[key] = cell(f, df, n, z, isz, l, r)
            res = cache[key]
            if res[3] > TOL:
                split = (i, z, isz, l, r)
                break
            out.append((z, isz, l, r) + res)
        if split is None:
            return out
        i, z, isz, l, r = split
        pl, pr = mpf(float((l + z) / 2)), mpf(float((z + r) / 2))
        pts[i:i + 1] = ([(pl, False)] if pl > l else []) + [(z, isz)] + ([(pr, False)] if pr < r else [])


def ddsplit(v):
    hi = float(v)
    return hi, float(v - mpf(hi))


def emit(kind, n, xmin):
    cells = build(kind, n, xmin)
    name = "bessel%s%dCells" % (kind, n)
    worst = max(c[7] for c in cells)
    print("// %s: %d cells on [%r, %d]; worst |h^2 dR|/|f| = %s." % (name, len(cells), xmin, XMAX, mpmath.nstr(worst, 3)))
    print("var %s = [...]besselCell{" % name)
    for z, isz, l, r, a0, a1, co, w in cells:
        zh, zl = ddsplit(z)
        a0h, a0l = ddsplit(a0)
        a1h, a1l = ddsplit(a1)
        print("\t{lo: %r, z: %r, zlo: %r, a0: %r, a0lo: %r, a1: %r, a1lo: %r, r: [...]float64{" % (
            float(l), zh, zl, a0h, a0l, a1h, a1l))
        print("\t\t" + ", ".join(repr(float(v)) for v in reversed(co)) + ",")
        print("\t}},")
    print("}")
    print()
    # uniform lookup: besselXnLUT[k] = first cell whose right edge > k/LUTSCALE
    edges = [float(c[2]) for c in cells] + [float(XMAX)]
    lut = []
    k = 0
    while k / LUTSCALE < XMAX:
        x = k / LUTSCALE
        j = 0
        while j + 1 < len(cells) and edges[j + 1] <= max(x, xmin):
            j += 1
        lut.append(j)
        k += 1
    print("var %sLUT = [...]uint16{" % name)
    for i in range(0, len(lut), 16):
        print("\t" + ", ".join(str(v) for v in lut[i:i + 16]) + ",")
    print("}")
    print()


LUTSCALE = 8

if __name__ == "__main__":
    print("// Code generated by testdata/gen/fit_bessel_cells.py; DO NOT EDIT.")
    print("//")
    print("// Degree-%d remainders R of f(z+h) = a0 + a1 h + h^2 R(h) per cell, fitted" % DEG)
    print("// to mpmath %s at 200 bits; lookup tables with %d entries per unit of x." % (mpmath.__version__, LUTSCALE))
    print()
    print("package special")
    print()
    emit("J", 0, 0.0)
    emit("J", 1, 0.0)
    emit("Y", 0, 0.6)
    emit("Y", 1, 1.5)
