package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// mulRound returns m·2^k·t rounded once. t is split by math.Frexp first,
// so that the double-double product neither overflows nor underflows
// (a subnormal t would lose the low half of the exact product) and the
// whole scaling happens in the final rounding.
func mulRound(m dd.DD, k int, t float64) float64 {
	if math.IsInf(m.Hi, 0) || m.Hi == 0 {
		// dd.ExpMant beyond its range (|x| > 1e7): +Inf or 0.
		return m.Hi * t
	}
	ft, et := math.Frexp(t)
	return round(dd.MulF(m, ft), k+et, m.Hi*t)
}

// scaledCis returns e^x·c·2^shift and e^x·s·2^shift, each rounded once:
// e^x is a double-double mantissa times a power of two (dd.ExpMant,
// relative error about 2^-76), so neither the exponential nor the products
// overflow or underflow before the final scaling, and the only errors left
// are those of c and s and the last rounding. This is msun's __ldexp_cexp
// carried in double-double (npymath's _npy_scaled_cexp): the msun form,
// exp(x - k ln2) with k ln2 rounded to float64, is off by up to about
// 2^-53·k ln2 relative, hundreds of ulps at k = 1799.
func scaledCis(x, c, s float64, shift int) (float64, float64) {
	m, k := dd.ExpMant(dd.F(x))
	return mulRound(m, k+shift, c), mulRound(m, k+shift, s)
}

// Exp returns e^x, the base-e exponential of x: e^re (cos im + i sin im).
// The exponential is a double-double mantissa and a power of two, so each
// part is rounded once and is finite whenever its exact value is (up to
// re = 1454.9 for a tiny imaginary part, where math/cmplx returns Inf
// from re = 709.8). Special values: C11 Annex G G.6.3.1.
//
// Algorithm: FreeBSD msun cexp (David Schultz, BSD-2-Clause), as NumPy's
// npymath carries it, with the scaled exponential of every finite case
// taken from dd.ExpMant.
func Exp(x complex128) complex128 {
	r, i := real(x), imag(x)
	switch {
	case finite(r) && finite(i):
		if i == 0 {
			m, k := dd.ExpMant(dd.F(r))
			return complex(mulRound(m, k, 1), i)
		}
		s, c := sincos(i)
		re, im := scaledCis(r, c, s, 0)
		return complex(re, im)
	case math.IsNaN(r):
		if i == 0 {
			return x // exp(NaN ± i0) = NaN ± i0
		}
		return complex(r, math.Copysign(math.NaN(), i))
	case finite(r): // finite real part, imaginary part Inf or NaN
		return complex(math.NaN(), math.NaN())
	case r > 0: // +Inf
		switch {
		case i == 0:
			return x
		case finite(i):
			s, c := sincos(i)
			return complex(r*c, r*s)
		}
		return complex(r, math.NaN()) // ±Inf + iNaN: sign of the real part unspecified
	default: // -Inf
		if finite(i) {
			s, c := sincos(i)
			return complex(0*c, 0*s)
		}
		return complex(0, 0) // ±0 ± i0: signs unspecified
	}
}
