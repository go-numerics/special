package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Inverses of the regularised incomplete gamma functions.
//
// Algorithm. The starting value is DiDonato & Morris' (ACM TOMS 12(4),
// 1986, eqs. 21-36) as implemented in Boost.Math's find_inverse_gamma
// (boost/math/special_functions/detail/igamma_inverse.hpp, BSL-1.0, ported
// below). Boost then runs Halley's method on P(x) - p; here the iteration
// is Newton's method on log T(x) - log t in the variable log x, where T is
// whichever of P and Q is the smaller at the root and t its exact target
// (p, or 1 - p, which is exact when p >= 1/2). In that variable both log P
// and log Q are concave (the log-gamma density is log-concave), so the
// iteration converges monotonically after its first step, and it is kept
// inside a bracket regardless. Each step is applied as x + x·expm1(-Δ), so
// x itself is never recomputed from its logarithm. For a < 1 and x < 1.1
// log P is evaluated in double-double from its series (and log Q from
// -expm1(log P)): there d log P/d log x is about a, so locating x to one ulp
// needs log P to about a·2^-53 absolutely, which no float64 evaluation of P
// provides when a is small.

// GammaPInv returns the inverse of the regularised lower incomplete gamma
// function: the x >= 0 with P(a, x) = p.
//
// Domain: a >= 0, 0 <= p <= 1. Special values: GammaPInv(a, 0) = 0,
// GammaPInv(a, 1) = +Inf; for a = 0 or a = +Inf only those two are
// defined (NaN otherwise); NaN for a < 0, p outside [0, 1] or a NaN
// argument. A root below the smallest subnormal returns 0.
//
// Algorithm: DiDonato & Morris' initial approximation (via Boost.Math's
// find_inverse_gamma) refined by a safeguarded Newton iteration in log x;
// see the comment at the top of incgamma_inv.go.
func GammaPInv(a, p float64) float64 {
	switch {
	case math.IsNaN(a) || math.IsNaN(p) || a < 0 || p < 0 || p > 1:
		return math.NaN()
	case p == 0:
		return 0
	case p == 1:
		return math.Inf(1)
	case a == 0 || math.IsInf(a, 1):
		return math.NaN()
	}
	q := 1 - p
	g := igamInvGuess(a, p, q)
	if p <= 0.5 {
		return igamInvSolve(a, p, false, g)
	}
	return igamInvSolve(a, q, true, g) // q = 1 - p exactly
}

// GammaQInv returns the inverse of the regularised upper incomplete gamma
// function: the x >= 0 with Q(a, x) = q.
//
// Domain: a >= 0, 0 <= q <= 1. Special values: GammaQInv(a, 0) = +Inf,
// GammaQInv(a, 1) = 0; for a = 0 or a = +Inf only those two are defined
// (NaN otherwise); NaN for a < 0, q outside [0, 1] or a NaN argument.
//
// Algorithm: as GammaPInv, matching Q(x) = q directly when q <= 1/2 (so
// small q keep full relative accuracy in x).
func GammaQInv(a, q float64) float64 {
	switch {
	case math.IsNaN(a) || math.IsNaN(q) || a < 0 || q < 0 || q > 1:
		return math.NaN()
	case q == 0:
		return math.Inf(1)
	case q == 1:
		return 0
	case a == 0 || math.IsInf(a, 1):
		return math.NaN()
	}
	p := 1 - q
	g := igamInvGuess(a, p, q)
	if q <= 0.5 {
		return igamInvSolve(a, q, true, g)
	}
	return igamInvSolve(a, p, false, g) // p = 1 - q exactly
}

// igamLogTail returns log T(a, x), T = Q if upper else P, as a
// double-double, and r = x·(d/dx)T / T in absolute value (the derivative
// of log T with respect to log x), for finite a > 0, x > 0.
func igamLogTail(a, x float64, upper bool) (lt dd.DD, r float64) {
	lpre := igamLogPrefix(a, x) // log(x · x^(a-1) e^(-x) / Γ(a))
	switch {
	case a < 1 && x < 1.1:
		// Q from DLMF 8.7.3 with its leading difference exact; log P from
		// it unless P is small, where the series gives log P directly.
		qd := igamSmallQDD(a, x)
		switch {
		case upper:
			lt = igamLogDD(qd)
		case qd.Hi > 0.99:
			lt = igamLogPSeries(a, x, dd.LogAccurate(x))
		default:
			lt = igamLog1pDD(dd.Neg(qd))
		}
	default:
		p, q := igamPQ(a, x)
		v := p
		if upper {
			v = q
		}
		if v == 0 {
			return dd.F(math.Inf(-1)), math.Inf(1)
		}
		lt = igamLogDD(dd.F(v))
	}
	return lt, math.Exp(dd.Sub(lpre, lt).Hi)
}

// igamLogPSeries returns log P(a, x) for 0 < a < 1 and 0 <= x < 1.1 as a
// double-double with an absolute error of about 2^-100:
//
//	log P = a log x - x - log Γ(1+a) + log1p(S - 1),
//	S = Σ_{n>=0} x^n / ((a+1)...(a+n))      (DLMF 8.7.1),
//
// with log Γ(1+a) = log1p(Γ(1+a) - 1) (no cancellation against log a).
// lx is log x (passed so that x = 2^-1075, below the float64 range, can
// be probed as x = 0 with its true logarithm).
func igamLogPSeries(a, x float64, lx dd.DD) dd.DD {
	l := dd.MulF(lx, a)
	l = dd.AddF(l, -x)
	l = dd.Sub(l, igamLog1pDD(gamma1pm1(a)))
	return dd.Add(l, igamLog1pDD(igamLowerSeriesRest(a, x)))
}

// igamInvSolve returns the x with T(a, x) = t (T = Q if upper, else P),
// for 0 < t <= 1/2, from the starting value x0.
func igamInvSolve(a, t float64, upper bool, x0 float64) float64 {
	lt := dd.LogAccurate(t)
	// g(x) = ±(log T(x) - log t) increases with x; dg/dlog x = r > 0.
	eval := func(x float64) (g, r float64) {
		var l dd.DD
		l, r = igamLogTail(a, x, upper)
		if math.IsInf(l.Hi, -1) {
			g = math.Inf(-1) // T underflowed: far below the root's T
		} else {
			g = dd.Sub(l, lt).Float()
		}
		if upper {
			g = -g
		}
		return g, r
	}
	if !(x0 > 0x1p-1022) || math.IsInf(x0, 1) || math.IsNaN(x0) {
		// The guess underflowed (tiny a) or failed: the root may be below
		// the float64 range altogether.
		if g, _ := eval(0x1p-1074); g >= 0 {
			return igamInvBelowSubnormal(a, t, upper, lt)
		}
		if !(x0 > 0) || math.IsInf(x0, 1) {
			x0 = 1 // no usable guess: start in the middle of the range
		}
	}
	lo, hi := 0.0, math.Inf(1)
	x := x0
	for i := 0; i < 400; i++ {
		g, r := eval(x)
		if g == 0 {
			return x
		}
		if g < 0 {
			lo = x
		} else {
			hi = x
		}
		step := -g / r
		nx := math.NaN()
		if !math.IsInf(step, 0) && !math.IsNaN(step) && step < 700 {
			nx = float64(x + x*math.Expm1(step))
		}
		if nx == x || math.Abs(step) < 0x1p-52 {
			return nx // converged: the step is below the resolution of x
		}
		if !(nx > lo && nx < hi) {
			switch {
			case math.IsInf(hi, 1):
				nx = math.Min(x*16, math.MaxFloat64)
			case lo == 0:
				nx = x / 16
			default:
				nx = float64(math.Sqrt(lo) * math.Sqrt(hi))
			}
			x = nx
			continue
		}
		x = nx
	}
	return x
}

// igamInvBelowSubnormal decides between 0 and the smallest subnormal for a
// root at or below 2^-1074: it is 2^-1074 when the root is at least
// 2^-1075 (the halfway point), i.e. when g(2^-1075) <= 0. Only a < 1 can
// put the root there; T is then evaluated from the series at x = 2^-1075
// (x itself is below the float64 range, log x is exact).
func igamInvBelowSubnormal(a, t float64, upper bool, lt dd.DD) float64 {
	if a >= 1 {
		return 0x1p-1074
	}
	lp := igamLogPSeries(a, 0, dd.MulF(ln2DD, -1075))
	var g float64
	if upper {
		g = -dd.Sub(igamLogDD(dd.Neg(igamExpm1DD(lp))), lt).Hi
	} else {
		g = dd.Sub(lp, lt).Hi
	}
	if g > 0 {
		return 0
	}
	return 0x1p-1074
}

// igamInvGuess returns DiDonato & Morris' approximation to the x with
// P(a, x) = p, Q(a, x) = q (ACM TOMS 12(4), 1986, eqs. 21-36), a port of
// Boost.Math's find_inverse_gamma (detail/igamma_inverse.hpp, BSL-1.0).
func igamInvGuess(a, p, q float64) float64 {
	switch {
	case a == 1:
		return -igamLog(q)
	case a < 1:
		g := Gamma(a)
		b := q * g
		switch {
		case b > 0.6 || (b >= 0.45 && a >= 0.3):
			// DiDonato and Morris Eq 21
			var u float64
			if b*q > 1e-8 && q > 1e-5 {
				u = math.Pow(p*g*a, 1/a)
			} else {
				u = math.Exp(-q/a - euler)
			}
			return u / (1 - u/(a+1))
		case a < 0.3 && b >= 0.35:
			// DiDonato and Morris Eq 22
			t := math.Exp(-euler - b)
			u := t * math.Exp(t)
			return t * math.Exp(u)
		case b > 0.15 || a >= 0.3:
			// DiDonato and Morris Eq 23
			y := -igamLog(b)
			u := y - (1-a)*math.Log(y)
			return y - (1-a)*math.Log(u) - math.Log(1+(1-a)/(1+u))
		case b > 0.1:
			// DiDonato and Morris Eq 24
			y := -igamLog(b)
			u := y - (1-a)*math.Log(y)
			return y - (1-a)*math.Log(u) - math.Log((u*u+2*(3-a)*u+(2-a)*(3-a))/(u*u+(5-a)*u+2))
		}
		// DiDonato and Morris Eq 25
		return igamInvEq25(a, -igamLog(b))
	}
	// DiDonato and Morris Eq 31
	s := igamInvS(p, q)
	s2 := s * s
	s3 := s2 * s
	s4 := s2 * s2
	s5 := s4 * s
	ra := math.Sqrt(a)
	w := a + s*ra + (s*s-1)/3
	w += (s3 - 7*s) / (36 * ra)
	w -= (3*s4 + 7*s2 - 16) / (810 * a)
	w += (9*s5 + 256*s3 - 433*s) / (38880 * a * ra)
	switch {
	case a >= 500 && math.Abs(1-w/a) < 1e-6:
		return w
	case p > 0.5:
		if w < 3*a {
			return w
		}
		d := math.Max(2, a*(a-1))
		lb := igamLog(q) + LogGamma(a)
		if lb < -d*2.3 {
			// DiDonato and Morris Eq 25
			return igamInvEq25(a, -lb)
		}
		// DiDonato and Morris Eq 33
		u := -lb + (a-1)*math.Log(w) - math.Log(1+(1-a)/(1+w))
		return -lb + (a-1)*math.Log(u) - math.Log(1+(1-a)/(1+u))
	}
	z := w
	ap1 := a + 1
	ap2 := a + 2
	if w < 0.15*ap1 {
		// DiDonato and Morris Eq 35
		v := igamLog(p) + LogGamma(ap1)
		z = math.Exp((v + w) / a)
		s = math.Log1p(z / ap1 * (1 + z/ap2))
		z = math.Exp((v + z - s) / a)
		s = math.Log1p(z / ap1 * (1 + z/ap2))
		z = math.Exp((v + z - s) / a)
		s = math.Log1p(z / ap1 * (1 + z/ap2*(1+z/(a+3))))
		z = math.Exp((v + z - s) / a)
	}
	if z <= 0.01*ap1 || z > 0.7*ap1 {
		return z
	}
	// DiDonato and Morris Eq 36
	ls := math.Log(igamInvSN(a, z, 100, 1e-4))
	v := igamLog(p) + LogGamma(ap1)
	z = math.Exp((v + z - ls) / a)
	return z * (1 - (a*math.Log(z)-z-v+ls)/(a-z))
}

// igamInvEq25 is DiDonato and Morris' Eq 25 (asymptotic inversion for
// small q), with y = -log(q Γ(a)).
func igamInvEq25(a, y float64) float64 {
	c1 := (a - 1) * math.Log(y)
	c12 := c1 * c1
	c13 := c12 * c1
	c14 := c12 * c12
	a2 := a * a
	a3 := a2 * a
	c2 := (a - 1) * (1 + c1)
	c3 := (a - 1) * (-(c12 / 2) + (a-2)*c1 + (3*a-5)/2)
	c4 := (a - 1) * ((c13 / 3) - (3*a-5)*c12/2 + (a2-6*a+7)*c1 + (11*a2-46*a+47)/6)
	c5 := (a - 1) * (-(c14 / 4) +
		(11*a-17)*c13/6 +
		(-3*a2+13*a-13)*c12 +
		(2*a3-25*a2+72*a-61)*c1/2 +
		(25*a3-195*a2+477*a-379)/12)
	y2 := y * y
	y3 := y2 * y
	y4 := y2 * y2
	return y + c1 + (c2 / y) + (c3 / y2) + (c4 / y3) + (c5 / y4)
}

// igamInvS is DiDonato and Morris' Eq 32: the normal deviate s with
// Φ(s) ~ p, from a rational approximation in t = sqrt(-2 log min(p, q)).
func igamInvS(p, q float64) float64 {
	var t float64
	if p < 0.5 {
		t = math.Sqrt(-2 * igamLog(p))
	} else {
		t = math.Sqrt(-2 * igamLog(q))
	}
	num := 3.31125922108741 + t*(11.6616720288968+t*(4.28342155967104+t*0.213623493715853))
	den := 1 + t*(6.61053765625462+t*(6.40691597760039+t*(1.27364489782223+t*0.3611708101884203e-1)))
	s := t - num/den
	if p < 0.5 {
		s = -s
	}
	return s
}

// igamInvSN is DiDonato and Morris' Eq 34: the partial sum
// 1 + Σ_{i=1}^{N} x^i / ((a+1)...(a+i)), stopped once a term falls below
// tolerance.
func igamInvSN(a, x float64, n int, tolerance float64) float64 {
	sum := 1.0
	partial := 1.0
	for i := 1; i <= n; i++ {
		partial *= x / (a + float64(i))
		sum += partial
		if partial < tolerance {
			break
		}
	}
	return sum
}
