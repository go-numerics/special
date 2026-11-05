package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// π/2 as an unevaluated sum of three float64 (53 + 53 + 53 bits).
const (
	pio2a = 1.5707963267948966
	pio2b = 6.123233995736766e-17
	pio2c = -1.4973849048591698e-33
)

// kahan returns, for finite x, y with |x| < coshBig, the double-double
// pieces of Kahan's form of tanh(x+iy): t = tan y, B = 1 + t², S = sinh x,
// C = cosh x = sqrt(1 + S²). Then
//
//	tanh z = (B C S + i t) / (1 + B S²)
//	coth z = (B C S - i t) / (B S² + t²)
//
// (the second from (B C S)² + t² = (1 + B S²)(B S² + t²)). C and S come
// from coshSinh, B in double-double from the exact square of t; only
// math.Tan rounds.
func kahan(x, y float64) (t float64, b, c, s dd.DD) {
	t = tan(y)
	ch, sh, _ := coshSinh(math.Abs(x))
	if math.Signbit(x) {
		sh = dd.Neg(sh)
	}
	return t, dd.AddF(dd.MulFF(t, t), 1), ch, sh
}

// tanhBig returns ±1 and the imaginary part of tanh(x+iy) for |x| ≥ 22,
// where tanh z = sign x + i 2 sin 2y e^(-2|x|) to far below an ulp
// (msun's TANH_HUGE branch, 4 sin y cos y exp(-|x|)², with the exponential
// in double-double and rounded once with the product).
func tanhBig(x, y float64) (float64, float64) {
	m, k := dd.ExpMant(dd.F(-2 * math.Abs(x)))
	var im float64
	if math.Abs(y) < 0x1p1000 {
		s, _ := sincos(2 * y)
		im = mulRound(m, k+1, s)
	} else {
		// 2y would overflow: 4 sin y cos y, the product rounded once more.
		s, c := sincos(y)
		im = mulRound(m, k+2, s*c)
	}
	return math.Copysign(1, x), im
}

// Tanh returns the hyperbolic tangent of x. Each part is formed in
// double-double from tan im, sinh re and cosh re (Kahan's algorithm), so
// the only rounding before the last is that of math.Tan; for |re| ≥ 22 the
// real part is ±1 and the imaginary part 2 sin(2 im) e^(-2|re|), which
// math/cmplx turns into NaN (Inf/Inf) from |re| = 355. Special values: C11
// Annex G G.6.2.6.
//
// Algorithm: FreeBSD msun ctanh r226600 (Bruce D. Evans, after W. Kahan,
// "Branch cuts for complex elementary functions", 1987; BSD-2-Clause), as
// NumPy's npymath carries it, carried in double-double.
func Tanh(z complex128) complex128 {
	x, y := real(z), imag(z)
	if !finite(x) {
		if math.IsNaN(x) {
			if y == 0 {
				return complex(x, y) // tanh(NaN ± i0) = NaN ± i0
			}
			return complex(x, x*y)
		}
		// tanh(±Inf + i y) = ±1 + i0 sin 2y; tanh(±Inf + i Inf), tanh(±Inf +
		// i NaN) = ±1 ± i0 (sign of the zero unspecified).
		v := y
		if finite(y) {
			v = math.Sin(y) * math.Cos(y)
		}
		return complex(math.Copysign(1, x), math.Copysign(0, v))
	}
	if !finite(y) {
		return complex(y-y, y-y) // tanh(x + i Inf) = tanh(x + i NaN) = NaN + i NaN
	}
	if math.Abs(x) >= coshBig {
		re, im := tanhBig(x, y)
		return complex(re, im)
	}
	t, b, c, s := kahan(x, y)
	den := dd.AddF(dd.Mul(b, dd.Sqr(s)), 1)
	re := dd.Div(dd.Mul(dd.Mul(b, c), s), den)
	im := dd.Div(dd.F(t), den)
	return complex(round(re, 0, x), round(im, 0, y))
}

// Tan returns the tangent of x, -i Tanh(i x) (C11 G.6p7).
func Tan(x complex128) complex128 {
	w := Tanh(complex(-imag(x), real(x)))
	return complex(imag(w), -real(w))
}

// coth returns the hyperbolic cotangent of finite x + iy, not both zero,
// by the second form of kahan; tiny arguments are scaled so that
// B S² + t² does not underflow.
func coth(x, y float64) complex128 {
	if math.Abs(x) >= coshBig {
		re, im := tanhBig(x, y)
		return complex(re, -im)
	}
	t, b, c, s := kahan(x, y)
	// The denominator is formed from S and t scaled by 2^-e, the larger in
	// [1, 2), so that it neither underflows nor overflows; the numerators
	// stay unscaled (scaling the smaller of S and t down could make it
	// subnormal) and the quotients are scaled by 2^-2e in their rounding.
	e := math.Ilogb(math.Max(math.Abs(s.Hi), math.Abs(t)))
	ss, ts := dd.Ldexp(s, -e), math.Ldexp(t, -e)
	den := dd.Add(dd.Mul(b, dd.Sqr(ss)), dd.MulFF(ts, ts))
	re := dd.Div(dd.Mul(dd.Mul(b, c), s), den)
	im := dd.Div(dd.F(-t), den)
	return complex(round(re, -2*e, x), round(im, -2*e, -y))
}

// Cot returns the cotangent of x, i coth(i x). Away from 0 each part is
// formed in double-double (the coth form of Kahan's algorithm); at 0 and
// for non-finite arguments it is Div(1, Tan(x)) (math/cmplx returns NaN
// for |x| below about 1e-154, where 1/x is finite).
func Cot(x complex128) complex128 {
	re, im := real(x), imag(x)
	if !finite(re) || !finite(im) || x == 0 {
		return Div(1, Tan(x))
	}
	w := coth(-im, re)
	return complex(-imag(w), real(w))
}
