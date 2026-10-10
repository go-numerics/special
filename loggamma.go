package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// logGammaMax: log Γ(x) overflows float64 for x above this value.
const logGammaMax = 2.556348e305

// log1pDD returns log(1 + u) for a double-double u > -1 as a double-double
// accurate to about 2^-90 relative. For |u| < 2^-20 it is the series
// u - u²/2 + u³/3 - ... (first two terms in double-double); otherwise
// w = 1 + u is formed exactly and dd.Log keeps its relative accuracy as w
// approaches 1 (its reduced argument m - 1 is exact). Forming 1 + u for a
// tiny u would leave a low part comparable to the result, whose first-order
// correction is then not enough.
func log1pDD(u dd.DD) dd.DD {
	if math.Abs(u.Hi) < 0x1p-20 {
		x := u.Hi
		tail := x * x * x * (1.0/3 - x*(1.0/4-x*(1.0/5-x/6)))
		return dd.AddF(dd.Sub(u, dd.MulF(dd.Mul(u, u), 0.5)), tail)
	}
	h, l := dd.TwoSum(1, u.Hi)
	w := dd.AddF(dd.DD{Hi: h, Lo: l}, u.Lo)
	return logDD(w)
}

// logDD returns log(g) for a double-double g > 0.
func logDD(g dd.DD) dd.DD {
	return dd.AddF(dd.Log(g.Hi), g.Lo/g.Hi)
}

// LogGamma returns log|Γ(x)|, the natural logarithm of the absolute value
// of the gamma function.
//
// Special values: LogGamma(±Inf) = +Inf; LogGamma(x) = +Inf at the poles
// x = 0, -1, -2, ...; NaN for NaN; +Inf for x > 2.556348e305 (overflow).
//
// Algorithm: near the zeros at x = 1 and x = 2 the result is log1p of
// Γ(1+t)-1 computed from the same fitted core as Gamma (gamma1pm1), so the
// relative accuracy holds as the result goes to zero; on (2.5, 10) it is the
// logarithm of Gamma's double-double value; from 10 Stirling's series
// (DLMF 5.11.1) is summed in double-double. For x < 0 it is
// log|Γ(1+t)| - log|t Π(x+k)| with exact factors for x > -34, and the
// reflection formula log π - log|x sin πx| - log Γ(-x) below.
func LogGamma(x float64) float64 {
	v, _ := LogGammaSign(x)
	return v
}

// LogGammaSign returns log|Γ(x)| and the sign of Γ(x) (+1 or -1), like
// math.Lgamma. At a pole the sign is +1, except for -0 where it is -1.
func LogGammaSign(x float64) (float64, int) {
	switch {
	case math.IsNaN(x):
		return x, 1
	case math.IsInf(x, 0):
		return math.Inf(1), 1
	case x == 0:
		if math.Signbit(x) {
			return math.Inf(1), -1
		}
		return math.Inf(1), 1
	}
	if x > 0 {
		if x > logGammaMax {
			return math.Inf(1), 1
		}
		return logGammaPos(x).Float(), 1
	}
	if x == math.Trunc(x) {
		return math.Inf(1), 1
	}
	sign := 1
	if gammaSignNeg(x) < 0 {
		sign = -1
	}
	return logGammaNeg(x).Float(), sign
}

// logGammaPos returns log Γ(x) for finite 0 < x <= logGammaMax as a
// double-double.
func logGammaPos(x float64) dd.DD {
	switch {
	case x < 0x1p-56:
		// log Γ(x) = -log x - γx + O(x²)
		return dd.AddF(dd.Neg(dd.Log(x)), -euler*x)
	case x < 0.5:
		// log(Γ(1+x)/x)
		return dd.Sub(log1pDD(gamma1pm1(x)), dd.Log(x))
	case x < 1.5:
		return log1pDD(gamma1pm1(x - 1)) // x-1 exact
	case x < 2.5:
		// log((1+t)Γ(1+t)), t = x-2 exact; (1+t)(1+u) - 1 = t + u + tu.
		t := x - 2
		u := gamma1pm1(t)
		w := dd.Add(dd.AddF(u, t), dd.MulF(u, t))
		return log1pDD(w)
	case x < 10:
		return logDD(gammaPos(x))
	}
	return logGammaStirling(x)
}

// logGammaNeg returns log|Γ(x)| for non-integer x < 0 as a double-double.
func logGammaNeg(x float64) dd.DD {
	switch {
	case x > -0x1p-56:
		return dd.AddF(dd.Neg(dd.Log(-x)), -euler*x)
	case x > -0.5:
		return dd.Sub(log1pDD(gamma1pm1(x)), dd.Log(-x))
	case x > -34:
		// |Γ(x)| = Γ(1+t) / |t Π_{k=0}^{n-1} (x+k)|, t = x+n in [0, 1),
		// with exact factors (see gammaNeg).
		n := math.Ceil(-x)
		t := x + n
		den := dd.F(t)
		for k := 0.0; k < n; k++ {
			den = dd.MulF(den, x+k)
		}
		if den.Hi < 0 {
			den = dd.Neg(den)
		}
		return dd.Sub(log1pDD(gamma1pm1(t)), logDD(den))
	}
	// log|Γ(x)| = log π - log|x sin πx| - log Γ(-x)
	s := math.Abs(sinPi(x))
	l := dd.Sub(lnPi, dd.Add(dd.Log(-x), dd.Log(s)))
	return dd.Sub(l, logGammaStirling(-x))
}

// logGammaAbsDD returns log|Γ(x)| as a double-double for finite x that is
// not a pole, x <= logGammaMax.
func logGammaAbsDD(x float64) dd.DD {
	if x > 0 {
		return logGammaPos(x)
	}
	return logGammaNeg(x)
}

// GammaSign returns the sign of Γ(x): +1 or -1, ±1 at ±0 (following the
// sign of the infinity Γ returns there), and NaN at the poles x = -1, -2,
// ..., at -Inf and for NaN.
func GammaSign(x float64) float64 {
	switch {
	case math.IsNaN(x) || math.IsInf(x, -1):
		return math.NaN()
	case x > 0:
		return 1
	case x == 0:
		return math.Copysign(1, x)
	case x == math.Trunc(x):
		return math.NaN()
	}
	return gammaSignNeg(x)
}
