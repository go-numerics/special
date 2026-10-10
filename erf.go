package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Double-double constants (mpmath, 300 bits).
var (
	twoOverSqrtPi = dd.DD{Hi: 1.1283791670955126, Lo: 1.533545961316588e-17}
	invSqrt2      = dd.DD{Hi: 0.7071067811865476, Lo: -4.833646656726457e-17}
	sqrtPiOver2   = dd.DD{Hi: 1.2533141373155003, Lo: -9.164289990229583e-17}
	sqrtPiHalf    = dd.DD{Hi: 0.886226925452758, Lo: -3.8332932499128993e-17}
	invSqrt2Pi    = dd.DD{Hi: 0.3989422804014327, Lo: -2.49232720227773e-17}
	ln2DD         = dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}
)

// erfSmallDD returns erf(x) for |x| <= 1/2 as x E(x²), E fitted to mpmath
// (zz_erfSmallPieces_coeffs.go) with its constant term in double-double.
func erfSmallDD(x float64) dd.DD {
	return dd.MulF(evalPiecesDD(erfSmallPieces[:], x*x), x)
}

// erfcxDD returns erfcx(x) = exp(x²) erfc(x) for x >= 1/2 (finite) as a
// double-double whose error is about 0.1 ulp: on [1/2, 8] a piecewise
// Chebyshev interpolant of erfcx itself; beyond, x erfcx(x) as a function
// of u = 1/x² (smooth, tending to 1/sqrt(pi)), both fitted to mpmath
// (zz_erfcxPieces_coeffs.go, zz_erfcxTailPieces_coeffs.go). Fitting erfcx
// rather than erfc removes the Gaussian factor from the approximation; it is
// reapplied exactly by the callers (the classical device of Cody, "Rational
// Chebyshev approximations for the error function", Math. Comp. 23, 1969).
func erfcxDD(x float64) dd.DD {
	if x < 8 {
		return evalPiecesDD(erfcxPieces[:], x)
	}
	g := evalPiecesDD(erfcxTailPieces[:], 1/(x*x))
	return dd.DivF(g, x)
}

// erfcxCorrected returns erfcx(z) for a double-double z with z.Hi >= 1/2,
// correcting the evaluation at z.Hi to first order through the logarithmic
// derivative erfcx'/erfcx = 2x - 2/(sqrt(pi) erfcx(x)), which tends to
// -1/x + 1/x³ for large x (taken from the asymptotic series there, since the
// difference cancels). The correction is relative and below 2^-52, so a few
// correct digits of the derivative are ample.
func erfcxCorrected(z dd.DD) dd.DD {
	e := erfcxDD(z.Hi)
	x := z.Hi
	var ld float64
	if x < 8 {
		ld = 2*x - twoOverSqrtPi.Hi/e.Hi
	} else {
		ld = -1/x + 1/(x*x*x)
	}
	return dd.AddF(e, e.Hi*z.Lo*ld)
}

// erfcMant returns m, k with erfc(z) = m * 2^k for a double-double
// z >= 1/2: exp(-z²) is formed from z² in double-double (dd.ExpMant) and
// multiplied by erfcx(z) before the single final rounding, so that the
// Gaussian factor does not amplify the rounding of z² (the error a naive
// exp(-x*x) incurs grows like 2x² ulp).
func erfcMant(z dd.DD) (dd.DD, int) {
	return erfcMantWith(z, dd.ExpMant)
}

// erfcMantPrecise is erfcMant with the Gaussian factor to about 2^-100
// (dd.ExpMantPrecise), for the residuals of the inverse functions.
func erfcMantPrecise(z dd.DD) (dd.DD, int) {
	return erfcMantWith(z, dd.ExpMantPrecise)
}

func erfcMantWith(z dd.DD, exp func(dd.DD) (dd.DD, int)) (dd.DD, int) {
	em, k := exp(dd.Neg(dd.Mul(z, z)))
	if em.Hi == 0 {
		return em, 0
	}
	return dd.Mul(em, erfcxCorrected(z)), k
}

// erfcDD returns erfc(z) as a double-double for finite z.
func erfcDD(z dd.DD) dd.DD {
	switch {
	case z.Hi >= 0.5:
		m, k := erfcMant(z)
		return dd.Ldexp(m, k)
	case z.Hi <= -0.5:
		return dd.Sub(dd.F(2), erfcDD(dd.Neg(z)))
	}
	// 1 - erf(z), erf(z.Hi + z.Lo) = erf(z.Hi) + erf'(z.Hi) z.Lo
	e := erfSmallDD(z.Hi)
	e = dd.AddF(e, z.Lo*twoOverSqrtPi.Hi*math.Exp(-z.Hi*z.Hi))
	return dd.Sub(dd.F(1), e)
}

// Erf returns the error function erf(x) = 2/sqrt(pi) ∫₀ˣ exp(-t²) dt.
//
// Special values: erf(±0) = ±0, erf(±Inf) = ±1, erf(NaN) = NaN.
//
// Algorithm: for |x| <= 1/2, x E(x²) with E fitted to mpmath; beyond,
// 1 - erfc(|x|) in double-double with erfc from erfcMant.
func Erf(x float64) float64 {
	ax := math.Abs(x)
	switch {
	case math.IsNaN(x):
		return x
	case ax <= 0.5:
		return erfSmallDD(x).Float()
	case ax >= 6:
		// erfc(6) < 2.2e-17 < ulp(1)/4: the result rounds to ±1.
		return math.Copysign(1, x)
	}
	m, k := erfcMant(dd.F(ax))
	return math.Copysign(dd.Sub(dd.F(1), dd.Ldexp(m, k)).Float(), x)
}

// Erfc returns the complementary error function erfc(x) = 1 - erf(x).
//
// Special values: erfc(+Inf) = 0, erfc(-Inf) = 2, erfc(NaN) = NaN; erfc
// underflows to 0 for x > 27.2.
//
// Algorithm: for x >= 1/2, exp(-x²) erfcx(x) with x² in double-double and
// erfcx fitted to mpmath, multiplied before the single rounding; for
// |x| < 1/2, 1 - erf(x) in double-double; for x <= -1/2, 2 - erfc(-x).
func Erfc(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x >= 0.5:
		if x > 27.3 {
			return 0
		}
		m, k := erfcMant(dd.F(x))
		return math.Ldexp(m.Float(), k)
	case x <= -6:
		// erfc(-6) = 2 - 2.2e-17: rounds to 2.
		return 2
	}
	return erfcDD(dd.F(x)).Float()
}

// Erfcx returns the scaled complementary error function
// erfcx(x) = exp(x²) erfc(x).
//
// Special values: erfcx(+Inf) = 0, erfcx(-Inf) = +Inf, erfcx(NaN) = NaN;
// erfcx overflows to +Inf for x < -26.628.
//
// Algorithm: for x >= 1/2 the fitted erfcx directly (no exponential at
// all); for |x| < 1/2, exp(x²)(1 - erf(x)) in double-double; for
// x <= -1/2, 2 exp(x²) - erfcx(-x).
func Erfcx(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x >= 0.5:
		if math.IsInf(x, 1) {
			return 0
		}
		return erfcxDD(x).Float()
	case x < -26.7:
		return math.Inf(1)
	}
	em, k := dd.ExpMant(dd.MulFF(x, x))
	if x > -0.5 {
		v := dd.Mul(em, dd.Sub(dd.F(1), erfSmallDD(x)))
		return math.Ldexp(v.Float(), k)
	}
	e2 := dd.Ldexp(dd.MulF(em, 2), k)
	if math.IsInf(e2.Hi, 0) {
		return e2.Hi
	}
	return dd.Sub(e2, erfcxDD(-x)).Float()
}

// ndtrZ returns z = -x/sqrt(2) as a double-double.
func ndtrZ(x float64) dd.DD {
	return dd.MulF(invSqrt2, -x)
}

// Ndtr returns the standard normal cumulative distribution function
// Φ(x) = (1/sqrt(2π)) ∫_{-∞}^x exp(-t²/2) dt = erfc(-x/sqrt 2)/2.
//
// Special values: Φ(-Inf) = 0, Φ(+Inf) = 1, Φ(NaN) = NaN.
//
// Algorithm: the argument -x/sqrt(2) is carried in double-double into
// erfc (erfcMant), so the lower tail keeps full relative accuracy down to
// the subnormal range; the rounding of x/sqrt(2) that limits
// implementations computing erfc(-x*M_SQRT1_2) (an error growing like
// x² ulp) does not occur.
func Ndtr(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		// Φ(-40) < 4e-350: Φ(x) rounds to 1 (and the tail to 0 below).
		return 1
	case x < -40:
		return 0
	}
	z := ndtrZ(x)
	if z.Hi >= 0.5 {
		m, k := erfcMant(z)
		return math.Ldexp(m.Float(), k-1)
	}
	return dd.MulF(erfcDD(z), 0.5).Float()
}

// LogNdtr returns log Φ(x), the logarithm of the standard normal cdf,
// accurate in both tails: for x -> -Inf it is -x²/2 - log(-x) - ... with no
// underflow, for large x it is -Φ(-x) with full relative accuracy.
//
// Special values: LogNdtr(-Inf) = -Inf, LogNdtr(+Inf) = 0,
// LogNdtr(NaN) = NaN.
//
// Algorithm: for x <= -1/sqrt(2), log erfcx(z) - z² - log 2 with
// z = -x/sqrt(2) in double-double; for x >= 1/sqrt(2), log1p(-Φ(-x)) with
// Φ(-x) in double-double; in between, the logarithm of Φ in double-double.
func LogNdtr(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		// log Φ(x) = -Φ(-x) + ..., and Φ(-40) < 4e-350 rounds to -0.
		return math.Copysign(0, -1)
	case x < -1e150:
		// log Φ(x) = -x²/2 - log(-x) - log sqrt(2π) + O(x^-2); the first
		// term alone is the correctly rounded result here.
		return -0.5 * x * x
	}
	z := ndtrZ(x)
	switch {
	case z.Hi >= 0.5:
		l := dd.Sub(logDD(erfcxCorrected(z)), dd.Mul(z, z))
		return dd.Sub(l, ln2DD).Float()
	case z.Hi <= -0.5:
		m, k := erfcMant(dd.Neg(z))
		q := dd.Ldexp(m, k-1)
		return log1pDD(dd.Neg(q)).Float()
	}
	return logDD(dd.MulF(erfcDD(z), 0.5)).Float()
}
