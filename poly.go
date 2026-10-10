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

// pi as a double-double.
var piDD = dd.DD{Hi: piHi, Lo: piLo}

// sinPi returns sin(pi*x) with the argument reduced exactly (math.Mod and
// the reflections used below are exact in binary floating point), so the
// result keeps full relative accuracy near the zeros at the integers, where
// math.Sin(math.Pi*x) loses it. sinPi(±Inf) and sinPi(NaN) are NaN.
func sinPi(x float64) float64 { return sinPiDD(x).Float() }

// cosPi returns cos(pi*x) with exact argument reduction.
func cosPi(x float64) float64 { return cosPiDD(x).Float() }

// sinPiDD returns sin(pi*x) as a double-double (relative error about 2^-60).
func sinPiDD(x float64) dd.DD {
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return dd.F(math.NaN())
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
	var s dd.DD
	switch {
	case r == 0:
		s = dd.DD{}
	case r <= 0.25:
		s = sinPiReduced(r)
	default:
		s = cosPiReduced(0.5 - r) // exact for r in (0.25, 0.5]
	}
	if neg {
		s = dd.Neg(s)
	}
	return s
}

// cosPiDD returns cos(pi*x) as a double-double with exact argument
// reduction.
func cosPiDD(x float64) dd.DD {
	if math.IsInf(x, 0) || math.IsNaN(x) {
		return dd.F(math.NaN())
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
	var c dd.DD
	if r <= 0.25 {
		c = cosPiReduced(r)
	} else {
		c = sinPiReduced(0.5 - r) // exact for r in (0.25, 0.5]
	}
	if neg {
		c = dd.Neg(c)
	}
	return c
}

// sinPiReduced returns sin(pi*r), 0 <= r <= 0.25, as r(π + r²G(r²)) in
// double-double: π r exactly, G (fitted to mpmath,
// zz_sinPiPieces_coeffs.go) with its constant term in double-double; the
// remaining float64 rounding is scaled down by |r²G|/π < 0.1, about 2^-60
// relative in all.
func sinPiReduced(r float64) dd.DD {
	u := dd.MulFF(r, r)
	g := evalPiecesDD(sinPiPieces[:], u.Hi)
	return dd.MulF(dd.Add(piDD, dd.Mul(u, g)), r)
}

// cosPiReduced returns cos(pi*r), 0 <= r <= 0.25, as 1 + r²H(r²) in
// double-double (H fitted to mpmath, zz_cosPiPieces_coeffs.go).
func cosPiReduced(r float64) dd.DD {
	u := dd.MulFF(r, r)
	h := evalPiecesDD(cosPiPieces[:], u.Hi)
	return dd.AddF(dd.Mul(u, h), 1)
}

// polyPiece is one piece of a piecewise polynomial approximation generated
// by testdata/gen/fit.py: on [lo, hi] the function is
// coef[0]*s^n + ... + coef[n] with s = t - c (coefficients descending).
type polyPiece struct {
	lo, hi, c float64
	c0lo      float64 // low part of the constant term coef[n] as a double-double
	coef      []float64
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
	// Callers pass t in [p[0].lo, p[n-1].hi]; the top end maps to the last
	// piece.
	w := p[0].hi - p[0].lo
	return min(int((t-p[0].lo)/w), len(p)-1)
}

// logf is math.Log made correct for subnormal arguments on every
// architecture: Go's amd64 assembly implementation of math.Log (log_amd64.s,
// Go 1.27) splits the exponent without normalising subnormals first, so it
// returns about -709.09 for every subnormal x (log(5e-324) is -744.44).
// Scaling by 2^54 into the normal range first sidesteps it.
func logf(x float64) float64 {
	if x > 0 && x < 0x1p-1022 {
		return math.Log(x*0x1p54) - 54*math.Ln2
	}
	return math.Log(x)
}
