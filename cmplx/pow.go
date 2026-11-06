package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// ddc is a complex number with double-double parts (unexported; the
// design's internal/ddc is created with its first consumer outside this
// package).
type ddc struct{ re, im dd.DD }

func (p ddc) mul(q ddc) ddc {
	return ddc{
		dd.Sub(dd.Mul(p.re, q.re), dd.Mul(p.im, q.im)),
		dd.Add(dd.Mul(p.re, q.im), dd.Mul(p.im, q.re)),
	}
}

// inv returns 1/p = conj(p)/|p|² (|p|² in double-double).
func (p ddc) inv() ddc {
	n := dd.Add(dd.Sqr(p.re), dd.Sqr(p.im))
	return ddc{dd.Div(p.re, n), dd.Neg(dd.Div(p.im, n))}
}

// mulRoundDD returns m·2^k·t rounded once for double-doubles m, t (t split
// by its exponent first, as mulRound).
func mulRoundDD(m dd.DD, k int, t dd.DD) float64 {
	if t.Hi == 0 || math.IsInf(m.Hi, 0) || m.Hi == 0 {
		return m.Hi * t.Hi
	}
	e := math.Ilogb(t.Hi)
	return round(dd.Mul(m, dd.Ldexp(t, -e)), k+e, m.Hi*t.Hi)
}

// Pow returns x**y, the base-x exponential of y, principal branch.
//
//   - y = 0: 1 (also for x = 0 and for NaN x, as NumPy).
//   - x = 0: 0 for Re y > 0, NaN + i NaN otherwise (NumPy's npy_cpow;
//     math/cmplx returns Inf for Re y < 0).
//   - y an integer with |y| < 100 (checked before any conversion to int,
//     whose result Go leaves implementation-defined out of range): binary
//     powering, NumPy's path, here on parts scaled by a power of two and
//     carried in double-double, then the reciprocal for y < 0, rounded
//     once: exact where the exact result is representable (i² = -1,
//     (1+i)⁴ = -4), within an ulp otherwise.
//   - otherwise exp(y·log x) with log x (logAbs and atan2), the product and
//     the exponential (dd.ExpMant, sin and cos of a double-double
//     argument) in double-double, so the error does not grow with
//     |y·log x| as Exp(y·Log(x)) does.
//
// Non-finite arguments: Exp(Mul(y, Log(x))).
func Pow(x, y complex128) complex128 {
	a, b, c, d := real(x), imag(x), real(y), imag(y)
	switch {
	case c == 0 && d == 0:
		return 1
	case a == 0 && b == 0:
		if c > 0 {
			return 0
		}
		return complex(math.NaN(), math.NaN())
	case !finite(a) || !finite(b) || !finite(c) || !finite(d):
		return Exp(Mul(y, Log(x)))
	case d == 0 && c > -100 && c < 100 && c == math.Trunc(c):
		return powInt(a, b, int(c))
	}
	// w = y·log x in double-double.
	lr := dd.Ldexp(logAbs(a, b), -1)
	li := atan2(b, a)
	wr := dd.Sub(dd.MulF(lr, c), dd.MulF(li, d))
	wi := dd.Add(dd.MulF(li, c), dd.MulF(lr, d))
	if math.Abs(wr.Hi) > 1e7 || math.Abs(wi.Hi) > 0x1p1000 {
		return Exp(complex(wr.Hi, wi.Hi)) // beyond ExpMant's range: overflow or underflow
	}
	m, k := dd.ExpMant(wr)
	s, co := sincosW(wi)
	return complex(mulRoundDD(m, k, co), mulRoundDD(m, k, s))
}

// powInt returns (a+ib)^n for 0 < |n| < 100 by binary powering in
// double-double on (a+ib)·2^-e, rounded once with the scale 2^(n·e).
func powInt(a, b float64, n int) complex128 {
	if n == 1 {
		return complex(a, b)
	}
	// Scale only where |x|^n, or |x^n|² in the reciprocal, could leave the
	// double-double range: scaling would flush a subnormal part of x to
	// zero (5e-324 + 83i squared has an imaginary part of 8.2e-322).
	// The scale moves the larger part only as far as needed: to exponent
	// ±(450/|n| - 2), so that |p|^|n| < 2^(1.5 + 450/|n| - 2)·|n| ≤ 2^450
	// and its square in the reciprocal stays finite; the smaller part keeps
	// its bits unless it is below 2^-600 of the larger (and negligible).
	e := math.Ilogb(math.Max(math.Abs(a), math.Abs(b)))
	m0 := max(n, -n)
	switch lim := 450/m0 - 2; {
	case e > lim:
		e -= lim
	case e < -lim:
		e += lim
	default:
		e = 0
	}
	p := ddc{dd.F(math.Ldexp(a, -e)), dd.F(math.Ldexp(b, -e))}
	m := n
	if m < 0 {
		m = -m
	}
	acc := ddc{dd.F(1), dd.F(0)}
	for {
		if m&1 == 1 {
			acc = acc.mul(p)
		}
		m >>= 1
		if m == 0 {
			break
		}
		p = p.mul(p)
	}
	if n < 0 {
		acc = acc.inv()
	}
	k := n * e
	return complex(round(acc.re, k, acc.re.Hi), round(acc.im, k, acc.im.Hi))
}
