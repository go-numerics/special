package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Log1p returns log(1 + x), principal branch, accurate for small x where
// Log(1+x) loses the digits of x that 1 + x rounds away (NumPy's
// nc_log1p is that naive formula).
//
// Real part, ½ log|1+z|²: where |z| < 2^26 and Re z is outside
// [-2, -1/2], |1+z|² - 1 = 2x + x² + y² is formed exactly (the squares by
// dd.TwoProd) and summed in double-double, then log1p of it in
// double-double; this is xsf's log1p (SciPy, BSD-3-Clause), whose
// double-double branch covers x < 0 with y² ≈ -2x, here applied
// throughout. Where Re z is in [-2, -1/2], 1 + x is exact (Sterbenz) and
// |1+z| comes from Log's logAbs; beyond |z| = 2^26 the rounding of 1 + x
// is far below an ulp of the result. Imaginary part: atan2 of y and the
// exact double-double 1 + x. Non-finite arguments: Log(1 + x).
func Log1p(x complex128) complex128 {
	re, im := real(x), imag(x)
	switch {
	case !finite(re) || !finite(im):
		return Log(complex(1+re, im))
	case re == 0 && im == 0:
		return x // log1p(±0 ± i0) = ±0 ± i0, as math.Log1p
	}
	s, e := dd.TwoSum(1, re)
	onePlus := dd.DD{Hi: s, Lo: e}
	var l dd.DD
	switch {
	case s == 0 && im == 0:
		return complex(math.Inf(-1), arg(im, s)) // log(0): -Inf + i(0 or π)
	case (re >= -2 && re <= -0.5) || math.Max(math.Abs(re), math.Abs(im)) >= 0x1p26:
		l = logAbs(s, im) // 1 + x exact in the first range, harmless in the second
	default:
		t := dd.Add(dd.AddF(dd.MulFF(re, re), float64(2*re)), dd.MulFF(im, im))
		l = log1pDD(t)
	}
	a := atan2X(im, onePlus)
	ia := a.Hi
	if ia != 0 {
		ia += a.Lo
	}
	return complex(round(l, -1, l.Hi), ia)
}

// expm1DD returns e^x - 1 as a double-double for |x| < 709: by its series
// below 2^-20 (the part past x in float64, below 2^-21 of it; truncation
// below 2^-100), otherwise as e^x from dd.ExpMant minus 1 (relative error
// about 2^-76/|x| ≤ 2^-56).
func expm1DD(x float64) dd.DD {
	if math.Abs(x) < 0x1p-20 {
		// x + x²/2 in double-double, x³/6 + x⁴/24 in float64.
		x2 := float64(x * x)
		c := float64(x2 * (float64(x*(1.0/6)) + float64(x2*(1.0/24))))
		return dd.AddF(dd.Add(dd.F(x), dd.Ldexp(dd.MulFF(x, x), -1)), c)
	}
	m, k := dd.ExpMant(dd.F(x))
	return dd.AddF(dd.Ldexp(m, k), -1)
}

// Expm1 returns e^x - 1, accurate for small x where Exp(x) - 1 cancels
// (NumPy's nc_expm1 is that formula, its real part naive):
//
//	Re = expm1(re) cos(im) - 2 sin²(im/2),  Im = e^re sin(im),
//
// with expm1 in double-double and both products and the sum carried in
// double-double, rounded once. Beyond re = 708 the real part is Exp's (the
// 1 is far below an ulp). Non-finite arguments: Exp(x) - 1.
//
// Limit: along the curve e^re cos(im) = 1 with |re| ≥ 2^-20 the real part
// rests on dd.ExpMant (2^-76 relative), so where it cancels by more than
// about 2^20 its error can exceed 2 ulp.
func Expm1(x complex128) complex128 {
	re, im := real(x), imag(x)
	if !finite(re) || !finite(im) {
		w := Exp(x)
		return complex(real(w)-1, imag(w))
	}
	if re == 0 && im == 0 {
		return x // expm1(±0 ± i0) = ±0 ± i0, as math.Expm1
	}
	if re > 708 {
		w := Exp(x)
		return complex(real(w), imag(w))
	}
	s, c := sincosDD(im)
	sh, _ := sincosDD(im / 2)
	m, k := dd.ExpMant(dd.F(re))
	r := dd.Sub(dd.Mul(expm1DD(re), c), dd.Ldexp(dd.Sqr(sh), 1))
	ri := r.Hi
	if ri != 0 {
		ri = round(r, 0, r.Hi)
	}
	if s.Hi == 0 {
		return complex(ri, float64(m.Hi*s.Hi)) // e^x · (±0)
	}
	es := math.Ilogb(s.Hi)
	return complex(ri, round(dd.Mul(m, dd.Ldexp(s, -es)), k+es, s.Hi))
}
