package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Constants as double-doubles (checked against 60 digits in log_test.go).
var (
	ln2DD    = dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}
	log10eDD = dd.DD{Hi: 0.4342944819032518, Lo: 1.098319650216765e-17}
	log2eDD  = dd.DD{Hi: 1.4426950408889634, Lo: 2.0355273740931033e-17}
)

// atan2 returns arg(x + iy) in [-π, π] as a double-double, to about
// 2^-64: the quotient of the smaller part by the larger in double-double,
// its arctangent by atanDD, and the multiple of π/2 added in double-double,
// so that the caller rounds once. It also repairs math.Atan2(y, x) for x < 0 and y/x below the
// smallest subnormal: Go computes Atan(y/x) = Atan(+0) for y = -5e-324,
// x = -1e130 and returns +π instead of -π (the sign of y is lost with the
// underflowed quotient). Non-finite and zero arguments: math.Atan2 with
// the sign of y restored (its special cases already follow Annex G).
func atan2(y, x float64) dd.DD {
	pi := dd.DD{Hi: 2 * pio2a, Lo: 2 * pio2b}
	switch {
	case !finite(x) || !finite(y):
		return dd.F(math.Copysign(math.Abs(math.Atan2(math.Abs(y), x)), y))
	case y == 0: // ±0 or ±π, exactly as double-doubles
		if math.Signbit(x) {
			pi.Hi, pi.Lo = math.Copysign(pi.Hi, y), math.Copysign(pi.Lo, y)
			return pi
		}
		return dd.F(y)
	case x == 0:
		return dd.DD{Hi: math.Copysign(pio2a, y), Lo: math.Copysign(pio2b, y)}
	}
	ay, ax := math.Abs(y), math.Abs(x)
	var r dd.DD
	if ay <= ax {
		r = atanDD(dd.Div(dd.F(ay), dd.F(ax)))
	} else {
		r = dd.Sub(dd.DD{Hi: pio2a, Lo: pio2b}, atanDD(dd.Div(dd.F(ax), dd.F(ay))))
	}
	if x < 0 {
		r = dd.Sub(pi, r)
	}
	if y < 0 {
		r = dd.Neg(r)
	}
	return r
}

// arg returns atan2 rounded to float64.
func arg(y, x float64) float64 {
	r := atan2(y, x)
	if r.Hi == 0 || !finite(r.Hi) {
		return r.Hi // keeps the sign of a zero (-0 + 0 would be +0)
	}
	return r.Hi + r.Lo
}

// normSq returns |x+iy|² as n·4^e: the parts are scaled by 2^-e, the
// exponent of the larger, so that n = x'² + y'² in [1, 8) is formed from
// exact squares and neither overflows nor underflows (barring a part
// below 2^-1022 of the other, whose square is negligible). x, y finite,
// not both zero.
func normSq(x, y float64) (n dd.DD, e int) {
	e = math.Ilogb(math.Max(math.Abs(x), math.Abs(y)))
	xs, ys := math.Ldexp(x, -e), math.Ldexp(y, -e)
	return dd.Add(dd.MulFF(xs, xs), dd.MulFF(ys, ys)), e
}

// log1pDD returns log(1 + t) for a double-double t > -1, to about 2^-70
// relative: by its series below 2^-20, otherwise as log(u.Hi) + u.Lo/u.Hi
// with u = 1 + t (the dropped (u.Lo/u.Hi)²/2 is below 2^-106).
func log1pDD(t dd.DD) dd.DD {
	if math.Abs(t.Hi) < 0x1p-20 {
		h := t.Hi
		h2 := float64(h * h)
		p, q := float64(h*(1.0/3)), float64(h2*0.25)
		c := h2 * (-0.5 + p - q)
		return dd.AddF(t, c)
	}
	u := dd.AddF(t, 1)
	return dd.AddF(dd.Log(u.Hi), u.Lo/u.Hi)
}

// logAbs returns 2·log|x+iy| as a double-double for finite x, y not both
// zero. Where |z| is in [1/2, 2√2) the argument of the logarithm is
// t = x² + y² - 1, formed exactly from the squares (dd.TwoProd) and
// summed in double-double, so a |z| within 1e-20 of 1 keeps every digit
// (npymath's band uses (am-1)(am+1) + an², which rounds three times, and
// math/cmplx takes log(hypot), which loses them all). Elsewhere it is
// log n + e·2 ln 2 with |z|² = n·4^e.
func logAbs(x, y float64) dd.DD {
	n, e := normSq(x, y)
	if e == 0 || e == -1 {
		t := dd.Add(dd.AddF(dd.MulFF(x, x), -1), dd.MulFF(y, y))
		return log1pDD(t)
	}
	l := dd.AddF(dd.Log(n.Hi), n.Lo/n.Hi)
	return dd.Add(l, dd.MulF(ln2DD, float64(2*e)))
}

// Log returns the natural logarithm of x, principal branch: log|x| + i
// arg x, with arg in [-π, π] and the sign of a zero imaginary part
// choosing the side of the cut along the negative real axis. The real part
// is formed in double-double (see logAbs) and rounded once; the imaginary
// part is math.Atan2. Special values: C11 Annex G G.6.3.2.
//
// Algorithm: npymath's npy_clog (after CPython; NumPy, BSD-3-Clause) with
// its four cases (huge, subnormal, near the unit circle, zero) replaced by
// the exact scaling and the exact t = |z|² - 1 of logAbs.
func Log(x complex128) complex128 {
	re, im := real(x), imag(x)
	return complex(logRe(re, im, dd.F(1)), arg(im, re))
}

// logRe returns c·log|x+iy| rounded once, with Annex G's special values
// for the real part.
func logRe(x, y float64, c dd.DD) float64 {
	switch {
	case math.IsInf(x, 0) || math.IsInf(y, 0):
		return math.Inf(1)
	case math.IsNaN(x) || math.IsNaN(y):
		return math.NaN()
	case x == 0 && y == 0:
		return math.Inf(-1)
	}
	l := dd.Mul(logAbs(x, y), c)
	return round(l, -1, l.Hi)
}

// Log10 returns the decimal logarithm of x, Log(x)·log10(e), each part
// multiplied in double-double and rounded once.
func Log10(x complex128) complex128 {
	re, im := real(x), imag(x)
	return complex(logRe(re, im, log10eDD), mulC(atan2(im, re), log10eDD))
}

// Log2 returns the base-2 logarithm of x, Log(x)·log2(e), each part
// multiplied in double-double and rounded once.
func Log2(x complex128) complex128 {
	re, im := real(x), imag(x)
	return complex(logRe(re, im, log2eDD), mulC(atan2(im, re), log2eDD))
}

// mulC returns a·c rounded once (NaN and zeros as a.Hi·c.Hi).
func mulC(a, c dd.DD) float64 {
	if !finite(a.Hi) || a.Hi == 0 {
		return a.Hi * c.Hi
	}
	p := dd.Mul(a, c)
	return p.Hi + p.Lo
}

// Sqrt returns the square root of x, principal branch (non-negative real
// part; the sign of a zero imaginary part chooses the side of the cut
// along the negative real axis). Special values: C11 Annex G G.6.4.2.
//
// Algorithm: CACM Algorithm 312 (1967) with npymath's special cases
// (FreeBSD msun csqrt, BSD-2-Clause), carried in double-double:
// |z| = sqrt(x'² + y'²) on the parts scaled by an even power of two,
// t = sqrt((|x| + |z|)/2) and |y|/(2t), each part rounded once with the
// scaling.
func Sqrt(z complex128) complex128 {
	a, b := real(z), imag(z)
	switch {
	case a == 0 && b == 0:
		return complex(0, b)
	case math.IsInf(b, 0):
		return complex(math.Inf(1), b)
	case math.IsNaN(a):
		return complex(a, math.NaN())
	case math.IsInf(a, 0):
		// sqrt(+Inf + i NaN) = +Inf + i NaN, sqrt(+Inf + iy) = +Inf + i0,
		// sqrt(-Inf + i NaN) = NaN ± i Inf, sqrt(-Inf + iy) = +0 + i Inf.
		if a < 0 {
			return complex(math.Abs(b-b), math.Copysign(a, b))
		}
		return complex(a, math.Copysign(b-b, b))
	case math.IsNaN(b):
		return complex(math.NaN(), math.NaN())
	}
	e := math.Ilogb(math.Max(math.Abs(a), math.Abs(b))) &^ 1 // even
	as, bs := math.Ldexp(a, -e), math.Ldexp(b, -e)
	n := dd.Add(dd.MulFF(as, as), dd.MulFF(bs, bs))
	h := dd.Sqrt(n)
	t := dd.Sqrt(dd.Ldexp(dd.AddF(h, math.Abs(as)), -1))
	q := dd.Div(dd.F(math.Abs(bs)), dd.Ldexp(t, 1))
	m := e / 2
	if a >= 0 {
		return complex(round(t, m, 1), math.Copysign(round(q, m, 1), b))
	}
	return complex(round(q, m, 1), math.Copysign(round(t, m, 1), b))
}
