package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// polevl evaluates the polynomial c[0]*x^n + c[1]*x^(n-1) + ... + c[n]
// (Cephes polevl: coefficients in descending order).
func polevl(x float64, c []float64) float64 {
	ans := c[0]
	for _, v := range c[1:] {
		ans = ans*x + v
	}
	return ans
}

// p1evl is polevl with an implicit leading coefficient of 1:
// x^n + c[0]*x^(n-1) + ... + c[n-1] (Cephes p1evl).
func p1evl(x float64, c []float64) float64 {
	ans := x + c[0]
	for _, v := range c[1:] {
		ans = ans*x + v
	}
	return ans
}

// chbevl evaluates the Chebyshev series sum' c[i] T_i(x/2) with the
// coefficients in reverse order, as Cephes chbevl does (Clenshaw recurrence).
func chbevl(x float64, c []float64) float64 {
	b0 := c[0]
	b1, b2 := 0.0, 0.0
	for _, v := range c[1:] {
		b2 = b1
		b1 = b0
		b0 = x*b1 - b2 + v
	}
	return 0.5 * (b0 - b2)
}

// pi = piHi + piLo to about 107 bits.
const (
	piHi = 3.141592653589793116
	piLo = 1.2246467991473532e-16
)

// sinPi returns sin(pi*x) with the argument reduced exactly (math.Mod and
// the reflections used below are exact in binary floating point), so the
// result keeps full relative accuracy near the zeros at the integers, where
// math.Sin(math.Pi*x) loses it.
func sinPi(x float64) float64 {
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return math.NaN()
	}
	neg := x < 0
	r := math.Mod(math.Abs(x), 2) // exact, in [0, 2)
	if r >= 1 {
		r -= 1 // exact
		neg = !neg
	}
	// sin(pi r), r in [0, 1)
	if r > 0.5 {
		r = 1 - r // exact
	}
	var s float64
	switch {
	case r == 0:
		s = 0
	case r <= 0.25:
		s = sinPiReduced(r)
	default:
		s = cosPiReduced(0.5 - r) // exact for r in (0.25, 0.5]
	}
	if neg {
		s = -s
	}
	return s
}

// cosPi returns cos(pi*x) with exact argument reduction.
func cosPi(x float64) float64 {
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return math.NaN()
	}
	r := math.Mod(math.Abs(x), 2) // exact, in [0, 2)
	if r > 1 {
		r = 2 - r // exact
	}
	// cos(pi r), r in [0, 1]
	neg := false
	if r > 0.5 {
		r = 1 - r
		neg = true
	}
	var c float64
	if r <= 0.25 {
		c = cosPiReduced(r)
	} else {
		c = sinPiReduced(0.5 - r) // exact for r in (0.25, 0.5]
	}
	if neg {
		c = -c
	}
	return c
}

// sinPiReduced returns sin(pi*r), 0 <= r <= 0.25, as
// r*(π + r²G(r²)) with π split in two (π = piHi + piLo) and r*piHi formed
// exactly: the polynomial term is below 10% of the result, so its rounding
// error is scaled down accordingly and the result is within about 0.6 ulp.
// G is fitted to mpmath (zz_sinPiPieces_coeffs.go, testdata/gen/fit.py).
func sinPiReduced(r float64) float64 {
	u := r * r
	h := float64(r * piHi)
	e := math.FMA(r, piHi, -h)
	return h + (e + r*(piLo+u*polevl(u-sinPiPieces[0].c, sinPiPieces[0].coef)))
}

// cosPiReduced returns cos(pi*r), 0 <= r <= 0.25, as 1 + r²H(r²) with r²
// carried exactly (H fitted to mpmath, zz_cosPiPieces_coeffs.go).
func cosPiReduced(r float64) float64 {
	uh := float64(r * r)
	ul := math.FMA(r, r, -uh)
	h := polevl(uh-cosPiPieces[0].c, cosPiPieces[0].coef)
	return math.FMA(uh, h, 1) + ul*h
}

// polyPiece is one piece of a piecewise polynomial approximation generated
// by testdata/gen/fit.py: on [lo, hi] the function is
// coef[0]*s^n + ... + coef[n] with s = t - c (coefficients descending).
type polyPiece struct {
	lo, hi, c float64
	c0lo      float64 // low part of the constant term coef[n] as a double-double
	coef      []float64
}

// evalPieces evaluates a piecewise polynomial whose pieces have equal width
// and are contiguous; t outside [p[0].lo, p[n-1].hi] uses the nearest piece.
func evalPieces(p []polyPiece, t float64) float64 {
	q := &p[pieceIndex(p, t)]
	return polevl(t-q.c, q.coef)
}

// evalPiecesDD evaluates the piecewise polynomial as a double-double:
// c0 + s*q(s) with c0 = coef[n] + c0lo exact and only s*q(s) (a small
// fraction of the result near the piece centre) rounded, which removes the
// ulp the final Horner step would otherwise cost.
func evalPiecesDD(p []polyPiece, t float64) dd.DD {
	q := &p[pieceIndex(p, t)]
	n := len(q.coef) - 1
	s := t - q.c
	r := s * polevl(s, q.coef[:n])
	h, l := dd.TwoSum(q.coef[n], r)
	return dd.DD{Hi: h, Lo: l + q.c0lo}
}

func pieceIndex(p []polyPiece, t float64) int {
	w := p[0].hi - p[0].lo
	i := int((t - p[0].lo) / w)
	if i < 0 {
		i = 0
	} else if i >= len(p) {
		i = len(p) - 1
	}
	return i
}
