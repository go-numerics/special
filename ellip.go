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

// ellipAt returns K(m) and E(m) for m < 1 given p = 1 - m exactly.
func ellipAt(m float64, p dd.DD) (k, e dd.DD) {
	switch {
	case m <= 0.5 && m >= -1:
		return evalPiecesDD(ellipKPieces[:], m), evalPiecesDD(ellipEPieces[:], m)
	case m > 0.5:
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
	_, e := ellipAt(m, sumDD(1, -m))
	return e.Float()
}
