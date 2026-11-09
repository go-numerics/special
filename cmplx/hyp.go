package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// coshBig (msun CCOSH_BIG): above it cosh(x) and sinh(x) both equal e^x/2
// to far below an ulp (e^-2x < 2^-63).
const coshBig = 22.0

// coshSinh returns cosh(ax) and sinh(ax) for ax ≥ 0 as double-double
// mantissas times 2^k, accurate to about 2^-70 relative: e^ax from
// dd.ExpMant, e^-ax as its double-double reciprocal; sinh by its Taylor
// series below 2^-16, where e^ax - e^-ax would cancel the bits ExpMant
// carries (the truncation, ax^6/5040, is below 2^-108 relative there).
func coshSinh(ax float64) (ch, sh dd.DD, k int) {
	if ax >= coshBig {
		m, k := dd.ExpMant(dd.F(ax))
		return m, m, k - 1
	}
	if ax < 0x1p-16 {
		t := float64(ax * ax)
		c, ce := dd.FastTwoSum(1, t/2+(t*t)/24)
		s, se := dd.FastTwoSum(ax, float64(ax*(t/6+(t*t)/120)))
		return dd.DD{Hi: c, Lo: ce}, dd.DD{Hi: s, Lo: se}, 0
	}
	m, k := dd.ExpMant(dd.F(ax))
	e := dd.Ldexp(m, k)
	ei := dd.Div(dd.F(1), e)
	return dd.Ldexp(dd.Add(e, ei), -1), dd.Ldexp(dd.Sub(e, ei), -1), 0
}

// Cosh returns the hyperbolic cosine of x: cosh re cos im + i sinh re sin im.
// cosh and sinh of the real part are double-double mantissas times a power
// of two, so each part is rounded once and is finite whenever its exact
// value is (math/cmplx overflows both parts from |re| = 709.8). Special
// values: C11 Annex G G.6.2.4.
//
// Algorithm: FreeBSD msun ccosh r226599 (Bruce D. Evans, David Schultz,
// BSD-2-Clause), as NumPy's npymath carries it, with cosh, sinh and the
// scaled exponential of the finite cases in double-double.
func Cosh(z complex128) complex128 {
	x, y := real(z), imag(z)
	xfinite, yfinite := finite(x), finite(y)
	if xfinite && yfinite {
		ch, sh, k := coshSinh(math.Abs(x))
		if y == 0 {
			return complex(mulRound(ch, k, 1), float64(x*y))
		}
		s, c := sincos(y)
		return complex(mulRound(ch, k, c), mulRound(sh, k, float64(math.Copysign(1, x)*s)))
	}
	switch {
	case x == 0 && !yfinite:
		// cosh(±0 ± i Inf) = NaN ± i0, cosh(±0 + i NaN) = NaN ± i0: the sign
		// of the zero is unspecified.
		return complex(y-y, math.Copysign(0, x*(y-y)))
	case y == 0 && !xfinite:
		// cosh(±Inf ± i0) = +Inf ± i0, cosh(NaN ± i0) = NaN ± i0.
		return complex(float64(x*x), float64(math.Copysign(0, x)*y))
	case xfinite && !yfinite:
		return complex(y-y, float64(x*(y-y)))
	case math.IsInf(x, 0):
		if !yfinite {
			// cosh(±Inf ± i Inf) = +Inf + i NaN, cosh(±Inf + i NaN) = +Inf + i NaN.
			return complex(float64(x*x), float64(x*(y-y)))
		}
		s, c := sincos(y)
		return complex(float64((x*x)*c), float64(x*s))
	}
	// cosh(NaN + i y) = NaN + i NaN for y ≠ 0.
	return complex(float64((x*x)*(y-y)), float64((x+x)*(y-y)))
}

// Sinh returns the hyperbolic sine of x: sinh re cos im + i cosh re sin im,
// evaluated as Cosh. Special values: C11 Annex G G.6.2.5.
//
// Algorithm: FreeBSD msun csinh r226599 (Bruce D. Evans, David Schultz,
// BSD-2-Clause), as NumPy's npymath carries it, with cosh, sinh and the
// scaled exponential of the finite cases in double-double.
func Sinh(z complex128) complex128 {
	x, y := real(z), imag(z)
	xfinite, yfinite := finite(x), finite(y)
	if xfinite && yfinite {
		ch, sh, k := coshSinh(math.Abs(x))
		sx := math.Copysign(1, x)
		if y == 0 {
			return complex(mulRound(sh, k, sx), y)
		}
		s, c := sincos(y)
		return complex(mulRound(sh, k, float64(sx*c)), mulRound(ch, k, s))
	}
	switch {
	case x == 0 && !yfinite:
		// sinh(±0 ± i Inf) = ±0 + i NaN, sinh(±0 + i NaN) = ±0 + i NaN: the
		// sign of the zero is unspecified.
		return complex(math.Copysign(0, x*(y-y)), y-y)
	case y == 0 && !xfinite:
		if math.IsNaN(x) {
			return z // sinh(NaN ± i0) = NaN ± i0
		}
		return complex(x, math.Copysign(0, y)) // sinh(±Inf ± i0) = ±Inf ± i0
	case xfinite && !yfinite:
		return complex(y-y, float64(x*(y-y)))
	case math.IsInf(x, 0):
		if !yfinite {
			// sinh(±Inf ± i Inf) = ±Inf + i NaN (sign of the real part
			// unspecified), sinh(±Inf + i NaN) likewise.
			return complex(float64(x*x), float64(x*(y-y)))
		}
		s, c := sincos(y)
		return complex(float64(x*c), float64(math.Inf(1)*s))
	}
	return complex(float64((x*x)*(y-y)), float64((x+x)*(y-y)))
}

// Cos returns the cosine of x, Cosh(i x) (C11 G.6p7).
func Cos(x complex128) complex128 {
	return Cosh(complex(-imag(x), real(x)))
}

// Sin returns the sine of x, -i Sinh(i x) (C11 G.6p7).
func Sin(x complex128) complex128 {
	w := Sinh(complex(-imag(x), real(x)))
	return complex(imag(w), -real(w))
}
