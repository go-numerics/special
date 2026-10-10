package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// sumDD returns a + b exactly as a double-double.
func sumDD(a, b float64) dd.DD {
	h, l := dd.TwoSum(a, b)
	return dd.DD{Hi: h, Lo: l}
}

// log1pmxDD returns log(1+x) - x for a double-double x > -1. For
// |x| <= 1/4 it sums the series -x²/2 + x³/3 - x⁴/4 + ... with the first
// two terms in double-double (the rest is below 1/3 of the result and
// summed in float64), which keeps full relative accuracy as x -> 0; beyond,
// log1p(x) - x in double-double has no cancellation to speak of.
func log1pmxDD(x dd.DD) dd.DD {
	if math.Abs(x.Hi) > 0.25 {
		return dd.Sub(log1pDD(x), x)
	}
	x2 := dd.Mul(x, x)
	r := dd.Add(dd.MulF(x2, -0.5), dd.DivF(dd.Mul(x2, x), 3))
	// tail: Σ_{k>=4} (-1)^(k+1) x^k / k
	xh := x.Hi
	s, p := 0.0, xh*xh*xh
	for k := 4.0; k < 120; k++ {
		p *= -xh
		t := p / k
		s += t
		if math.Abs(t) <= 0x1p-60*math.Abs(r.Hi) {
			break
		}
	}
	return dd.AddF(r, s)
}

// logGammaDiff returns log Γ(b) - log Γ(a+b) for b >= 10 and 0 < a <= b
// as a double-double, without forming either logarithm (they cancel for
// b >> a). With x = a/b, L = log1p(x) and Stirling's series S:
//
//	log Γ(b) - log Γ(b+a) = -a log b - b (L - x) - (a - 1/2) L + S(b) - S(a+b)
//
// (from DLMF 5.11.1); every term is well conditioned.
func logGammaDiff(a, b float64) dd.DD {
	x := dd.Div(dd.F(a), dd.F(b))
	L := log1pDD(x)
	t := dd.Neg(dd.MulF(dd.Log(b), a))
	t = dd.Sub(t, dd.MulF(log1pmxDD(x), b))
	t = dd.Sub(t, dd.Mul(sumDD(a, -0.5), L))
	s := sumDD(a, b)
	return dd.AddF(t, stirlingSeries(b)-stirlingSeries(s.Hi))
}

// logBetaPos returns log B(a, b) for 0 < a <= b (finite) as a double-double.
func logBetaPos(a, b float64) dd.DD {
	if b >= 10 {
		return dd.Add(logGammaPos(a), logGammaDiff(a, b))
	}
	// Both below 10: log Γ(a) + log Γ(b) - log Γ(a+b), with a+b = s + e
	// exactly and log Γ(s+e) = log Γ(s) + ψ(s) e to first order.
	s := sumDD(a, b)
	lg := dd.Add(logGammaPos(a), logGammaPos(b))
	ls := dd.AddF(logGammaPos(s.Hi), Digamma(s.Hi)*s.Lo)
	return dd.Sub(lg, ls)
}

// LogBeta returns log|B(a, b)|, the logarithm of the absolute value of the
// beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
//
// Special values: NaN if either argument is NaN; +Inf when a or b is 0 or a
// negative integer (and a+b is not); -Inf when B(a, b) = 0 (a+b a
// non-positive integer, a and b not).
//
// Algorithm: for positive arguments with max(a, b) >= 10, log Γ(b) -
// log Γ(a+b) is evaluated as a single cancellation-free expression from
// Stirling's series (logGammaDiff) and added to log Γ(a) in double-double;
// otherwise the three log-gammas are combined in double-double with the
// rounding of a+b corrected to first order. Negative arguments go through
// LogGamma with the same correction.
func LogBeta(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	if a > b {
		a, b = b, a
	}
	if a > 0 {
		if math.IsInf(b, 1) {
			return math.Inf(-1)
		}
		return logBetaPos(a, b).Float()
	}
	return logBetaGeneral(a, b)
}

// logBetaGeneral handles a <= 0 or b <= 0 (a <= b).
func logBetaGeneral(a, b float64) float64 {
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.NaN()
	}
	s := sumDD(a, b)
	pa, pb := a == math.Trunc(a) && a <= 0, b == math.Trunc(b) && b <= 0
	ps := s.Lo == 0 && s.Hi == math.Trunc(s.Hi) && s.Hi <= 0
	switch {
	case pa || pb:
		if ps && pa && pb {
			return math.NaN()
		}
		return math.Inf(1)
	case ps:
		return math.Inf(-1)
	}
	return logBetaNegDD(a, b, s).Float()
}

// logBetaNegDD returns log|B(a,b)| in double-double for arguments with no
// pole involved (s = a+b exactly).
func logBetaNegDD(a, b float64, s dd.DD) dd.DD {
	l := dd.Add(logGammaAbsDD(a), logGammaAbsDD(b))
	ls := dd.AddF(logGammaAbsDD(s.Hi), Digamma(s.Hi)*s.Lo)
	return dd.Sub(l, ls)
}

// Beta returns the beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
//
// Special values: NaN if either argument is NaN; +Inf when a or b is 0 or a
// negative integer and a+b is not a non-positive integer; 0 when a+b is a
// non-positive integer and neither a nor b is; overflow to ±Inf and
// underflow to 0 as the true value dictates.
//
// Algorithm: for positive arguments below 10 the three gamma values come
// from Gamma's double-double core and the rounding of a+b is corrected to
// first order (Γ(s+e) = Γ(s)(1 + ψ(s)e)); otherwise B = exp(log B) with
// log B from LogBeta's cancellation-free double-double evaluation.
func Beta(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	if a > b {
		a, b = b, a
	}
	if a > 0 {
		if math.IsInf(b, 1) {
			return 0
		}
		if b < 10 && a > 0x1p-56 {
			s := sumDD(a, b)
			gs := dd.MulF(gammaPos(s.Hi), 1+Digamma(s.Hi)*s.Lo)
			return dd.Div(dd.Mul(gammaPos(a), gammaPos(b)), gs).Float()
		}
		return dd.Exp(logBetaPos(a, b))
	}
	l := logBetaGeneral(a, b)
	if math.IsNaN(l) || math.IsInf(l, 0) {
		if math.IsInf(l, -1) {
			return 0
		}
		return l
	}
	s := sumDD(a, b)
	sign := GammaSign(a) * GammaSign(b) * GammaSign(s.Hi)
	return sign * dd.Exp(logBetaNegDD(a, b, s))
}
