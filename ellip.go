package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// The complete elliptic integrals are evaluated from fitted polynomials,
// using the structure of their logarithmic singularity at m = 1 (DLMF
// 19.12.1-2; it is the form of Cephes' ellpk/ellpe and of Hastings'
// approximations, A&S 17.3.34/36): with p = 1 - m,
//
//	K(1-p) = A_K(p) - log(p) K(p)/π
//	E(1-p) = A_E(p) - log(p) (K(p) - E(p))/π
//
// where A_K, A_E are analytic on [0, 1]. K and E on [-1, 1/2], and A_K, A_E
// and D(p) = (K(p) - E(p))/p (which removes the cancellation) on [0, 1/2], are piecewise Chebyshev interpolants fitted to mpmath
// (zz_ellip*_coeffs.go, testdata/gen/fit.py) with the constant terms in
// double-double; m < -1 is mapped into (1/2, 1) by the imaginary-modulus
// transformation K(m) = K(m/(m-1))/sqrt(1-m), E(m) = sqrt(1-m) E(m/(m-1))
// (DLMF 19.7.5).

// 1/π as a double-double (mpmath).
var invPi = dd.DD{Hi: 0.3183098861837907, Lo: -1.9678676675182486e-17}

// ellipOctPiece is one piece of zz_ellipOct_coeffs.go: on it the function
// is coef[0] s^11 + ... + coef[11] with s = m - c, the constant term
// carrying c0lo as its low part.
type ellipOctPiece struct {
	c, c0lo float64
	coef    [12]float64
}

// ellipOctIndex returns the piece of the octave tables holding m, from the
// exponent and the top three mantissa bits of p = 1 - m, or -1 when p is
// outside [2^-10, 2^11) (m near 1 or m < -2047), where the double-double
// path below takes over. p is rounded for m < -1; a piece is fitted
// slightly beyond its edges, so a rounding into the neighbouring piece is
// harmless.
func ellipOctIndex(m float64) int { return ellipOctIndexP(1 - m) }

// ellipOctIndexP is ellipOctIndex given p = 1 - m.
func ellipOctIndexP(p float64) int {
	b := math.Float64bits(p)
	e := int(b>>52) - 1023 // p > 0 here: m < 1
	if e < -10 || e > 10 {
		return -1
	}
	return (e+10)<<3 | int(b>>49&7)
}

// ellipOctEval evaluates one piece at s = m - c: c0 + (s q(s) + c0lo)
// rounds once, since |s q(s)| is below 2% of c0 on every piece. s is exact
// (Sterbenz) wherever the octave of p is below 1/2 or above 2, and its
// relative rounding elsewhere is scaled down by the same 2%.
func ellipOctEval(q *ellipOctPiece, s float64) float64 {
	c := &q.coef
	r := c[0]
	r = r*s + c[1]
	r = r*s + c[2]
	r = r*s + c[3]
	r = r*s + c[4]
	r = r*s + c[5]
	r = r*s + c[6]
	r = r*s + c[7]
	r = r*s + c[8]
	r = r*s + c[9]
	r = r*s + c[10]
	return c[11] + (r*s + q.c0lo)
}

// ellipKE returns K and E at m = 1 - p for a double-double p in (0, 1/2]
// (that is m in [1/2, 1)).
func ellipKE(p dd.DD) (k, e dd.DD) {
	lp := logDD(p)
	kp := evalPiecesDD(ellipKPieces[:], p.Hi)
	// K(p) - E(p) = p D(p), fitted directly: the difference would cancel.
	kme := dd.Mul(p, evalPiecesDD(ellipDPieces[:], p.Hi))
	k = dd.Sub(evalPiecesDD(ellipKLogPieces[:], p.Hi), dd.Mul(dd.Mul(lp, kp), invPi))
	e = dd.Sub(evalPiecesDD(ellipELogPieces[:], p.Hi), dd.Mul(dd.Mul(lp, kme), invPi))
	return k, e
}

// ellipAt returns K(m) and E(m) for m < 1 outside the octave tables (m
// near 1 or m < -2047) given p = 1 - m exactly.
func ellipAt(m float64, p dd.DD) (k, e dd.DD) {
	if m > 0.5 {
		return ellipKE(p)
	}
	// m < -1: m/(m-1) = 1 - 1/(1-m), so the transformed p is 1/(1-m).
	sq := dd.Sqrt(p)
	k, e = ellipKE(dd.Div(dd.F(1), p))
	return dd.Div(k, sq), dd.Mul(e, sq)
}

// EllipK returns the complete elliptic integral of the first kind
// K(m) = ∫₀^{π/2} dθ / sqrt(1 - m sin²θ), parameter m = k² (SciPy's ellipk
// convention), for m <= 1.
//
// Special values: K(1) = +Inf, K(-Inf) = 0, NaN for m > 1 or NaN. Near
// m = 1 use EllipKm1 with p = 1-m to keep the digits m itself cannot hold.
func EllipK(m float64) float64 {
	switch {
	case math.IsNaN(m) || m > 1:
		return math.NaN()
	case m == 1:
		return math.Inf(1)
	case math.IsInf(m, -1):
		return 0
	}
	if i := ellipOctIndex(m); i >= 0 {
		return ellipOctEval(&ellipKOct[i], m-ellipKOct[i].c)
	}
	k, _ := ellipAt(m, sumDD(1, -m))
	return k.Float()
}

// EllipKm1 returns K(1 - p) for p >= 0 (SciPy's ellipkm1): the form to use
// near m = 1, where p carries digits that 1-p would lose.
func EllipKm1(p float64) float64 {
	switch {
	case math.IsNaN(p) || p < 0:
		return math.NaN()
	case p == 0:
		return math.Inf(1)
	case math.IsInf(p, 1):
		return 0
	}
	if i := ellipOctIndexP(p); i >= 0 {
		// s = m - c = (1 - c) - p: 1 - c is the piece centre in p (exact)
		// and the difference is exact within the octave (Sterbenz).
		q := &ellipKOct[i]
		return ellipOctEval(q, (1-q.c)-p)
	}
	k, _ := ellipAt(1-p, dd.F(p))
	return k.Float()
}

// EllipE returns the complete elliptic integral of the second kind
// E(m) = ∫₀^{π/2} sqrt(1 - m sin²θ) dθ, parameter m = k², for m <= 1.
//
// Special values: E(1) = 1, E(-Inf) = +Inf, NaN for m > 1 or NaN.
func EllipE(m float64) float64 {
	switch {
	case math.IsNaN(m) || m > 1:
		return math.NaN()
	case m == 1:
		return 1
	case math.IsInf(m, -1):
		return math.Inf(1)
	}
	if i := ellipOctIndex(m); i >= 0 {
		return ellipOctEval(&ellipEOct[i], m-ellipEOct[i].c)
	}
	_, e := ellipAt(m, sumDD(1, -m))
	return e.Float()
}
