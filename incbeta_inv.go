package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Inverses of the regularised incomplete beta function.
//
// Algorithm. The root is first placed on one side of 1/2 by comparing the
// target with I_(1/2)(a, b); it is then found in the variable that is small
// there: u = x, or u = 1 - x with (a, b) exchanged (I_x(a,b) = 1 -
// I_(1-x)(b,a)), so that a root near 1 keeps the full relative accuracy of
// 1 - x and the result is a single rounding of 1 - u. On that side the
// smaller of the two tails is matched against its exact target (the input
// probability, or its complement when that is the smaller and hence exact),
// by Newton's method in log u on log T(u) - log t, kept inside a bracket.
// The starting value is the normal approximation for large a + b and the
// small-u power law u^a/(a B(a, b)) otherwise (as in the first steps of
// Boost.Math's ibeta_inv, boost/math/special_functions/detail/
// ibeta_inverse.hpp, after DiDonato & Morris and Temme, "Asymptotic
// inversion of the incomplete beta function", J. Comput. Appl. Math. 41,
// 1992); the iteration does the rest. Where d log T/d log u is small (a < 1,
// lower tail) the logarithm of the tail is taken in double-double from the
// power series (TOMS 708's BPSER), since the root then depends on log T to
// within a factor 1/a.
//
// Limitation: when a parameter is far below 1e-3 (a or b ~ 1e-20 down to
// 1e-300), I_x(a, b) is flat to within 2^-100 over most of (0, 1) — it is
// b/(a+b) up to factors x^a and (1-x)^b — and the float64 root can depend
// on digits beyond double-double: the result may then be far from the
// exact root (SciPy's is too, usually returning 0.5 or the smallest normal).

// BetaIncInv returns the inverse of the regularised incomplete beta
// function: the x in [0, 1] with I_x(a, b) = p.
//
// Domain: a > 0, b > 0, 0 <= p <= 1. Special values: BetaIncInv(a, b, 0) =
// 0 and BetaIncInv(a, b, 1) = 1 for any a, b >= 0 that are not NaN, with
// a = 0 or b = 0 giving NaN (SciPy's conventions); NaN for p outside
// [0, 1], a or b negative, infinite (for 0 < p < 1) or NaN. A root closer
// to 0 than the smallest subnormal returns 0, one closer to 1 than half an
// ulp of 1 returns 1.
//
// Algorithm: safeguarded Newton iteration in log x or log(1-x) on the
// smaller tail; see the comment at the top of incbeta_inv.go.
func BetaIncInv(a, b, p float64) float64 {
	switch {
	case math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(p) || a <= 0 || b <= 0 || p < 0 || p > 1:
		return math.NaN()
	case p == 0:
		return 0
	case p == 1:
		return 1
	case math.IsInf(a, 1) || math.IsInf(b, 1):
		return math.NaN()
	}
	return ibetaInv(a, b, p, 1-p)
}

// BetaIncCInv returns the inverse of the complement of the regularised
// incomplete beta function: the x in [0, 1] with 1 - I_x(a, b) = q.
//
// Domain and special values: as BetaIncInv, with BetaIncCInv(a, b, 0) = 1
// and BetaIncCInv(a, b, 1) = 0.
//
// Algorithm: as BetaIncInv, with q as the exact target of the upper tail
// (so a small q keeps full relative accuracy in 1 - x).
func BetaIncCInv(a, b, q float64) float64 {
	switch {
	case math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(q) || a <= 0 || b <= 0 || q < 0 || q > 1:
		return math.NaN()
	case q == 0:
		return 1
	case q == 1:
		return 0
	case math.IsInf(a, 1) || math.IsInf(b, 1):
		return math.NaN()
	}
	return ibetaInv(a, b, 1-q, q)
}

// ibetaInv returns the x with I_x(a, b) = p, 1 - I_x(a, b) = q, where
// p + q = 1 and the smaller of the two is exact.
func ibetaInv(a, b, p, q float64) float64 {
	if a == b && p == q {
		return 0.5 // symmetry
	}
	lb := ibetaLogBeta(a, b)
	if a < 1e-3 || b < 1e-3 {
		// Which side of 1/2: the sign of the x-side residual at 1/2, from
		// the same (double-double where needed) evaluation as the
		// iteration, so a nearly flat I (tiny a and b) is not decided by
		// a rounded I_(1/2).
		switch g0, _ := ibetaInvEval(a, b, p, q, lb)(0.5); {
		case g0 == 0:
			return 0.5
		case g0 > 0:
			u, _ := ibetaInvSide(a, b, p, q, lb, ibetaInvGuess(a, b, p, lb), true)
			return u
		}
		// I_x(a,b) = p  <=>  I_(1-x)(b,a) = q
		u, _ := ibetaInvSide(b, a, q, p, lb, ibetaInvGuess(b, a, q, lb), true)
		return 1 - u
	}
	// Elsewhere the side is taken from a starting value and confirmed by
	// the iteration's own evaluations (ibetaInvSide), which saves the
	// evaluation at 1/2 whenever the guess lies on the root's side of it.
	if x0 := ibetaInvGuessX(a, b, p, q, lb); x0 <= 0.5 {
		if u, ok := ibetaInvSide(a, b, p, q, lb, x0, false); ok {
			return u
		}
		u, _ := ibetaInvSide(b, a, q, p, lb, 0.5, true)
		return 1 - u
	} else {
		if u, ok := ibetaInvSide(b, a, q, p, lb, 1-x0, false); ok {
			return 1 - u
		}
		u, _ := ibetaInvSide(a, b, p, q, lb, 0.5, true)
		return u
	}
}

// ibetaInvGuessX returns a starting value for the x in (0, 1) with
// I_x(a, b) = p (1 - I = q), for a, b >= 1e-3: the normal approximation
// when a, b >= 1 and a + b >= 10, else the power law of whichever tail
// the root is nearer (u^a/(a B) = p from below, (1-x)^b/(b B) = q from
// above).
func ibetaInvGuessX(a, b, p, q float64, lb dd.DD) float64 {
	if a >= 1 && b >= 1 && a+b >= 10 {
		s := a + b
		x := a/s + Ndtri(p)*math.Sqrt(a*b/(s*s*(s+1)))
		if x > 0 && x < 1 {
			return x
		}
	}
	lu := (igamLog(p) + math.Log(a) + lb.Hi) / a
	lv := (igamLog(q) + math.Log(b) + lb.Hi) / b
	if lu <= lv {
		return math.Exp(lu)
	}
	return -math.Expm1(lv)
}

// ibetaInvEval returns g(u) = ±(log T(u) - log t), increasing in u, and
// r = dg/d log u > 0, where T is the smaller-target tail of I_u(a, b)
// (lower with target p, or upper with target q).
func ibetaInvEval(a, b, p, q float64, lb dd.DD) func(u float64) (g, r float64) {
	upper := q < p
	t := p
	if upper {
		t = q
	}
	// log t to 2^-104 where the series branch of ibetaInvTail (a < 1)
	// compares double-double logarithms; elsewhere only far from the root
	var lt dd.DD
	if a < 1 {
		lt = dd.LogAccurate(t)
	} else {
		lt = dd.Log(t)
	}
	lbf := lb.Hi + lb.Lo
	return func(u float64) (g, r float64) {
		g, r = ibetaInvTail(a, b, u, t, upper, lt, lb, lbf)
		if upper {
			g = -g
		}
		return g, r
	}
}

// ibetaInvSide returns the u (at most about 1/2) with I_u(a, b) = p and
// 1 - I_u(a, b) = q, matching the smaller of the two, from the starting
// value u0, and true. Unless sure, the root is first confirmed to lie
// below 1/2 (u0 <= 1/2): at once when it lies below u0, else by the sign
// at 1/2; a root above 1/2 returns false (the caller then takes the other
// side, where it is computed as 1 - (a root below 1/2)).
//
// Halley's method in log u: with r = d log T/d log u, the second
// derivative of g is r (a - (b-1) u/(1-u) ∓ r) (- for the lower tail),
// falling back to Newton's step far from the root. The iteration runs to
// the computed root (a step below the resolution of u): the forward
// function carries errors of a few ulps here, which vary from one u to
// the next, so a step extrapolated from a nearby u lands up to that much
// further from the true root.
func ibetaInvSide(a, b, p, q float64, lb dd.DD, u0 float64, sure bool) (float64, bool) {
	eval := ibetaInvEval(a, b, p, q, lb)
	u := u0
	// A root below the float64 range altogether: the power-law guess says
	// so (it is then near or below the normal range), except for tiny
	// parameters, where I is too flat for the guess to be trusted.
	if u < 0x1p-1000 || a < 1e-3 || b < 1e-3 {
		if g, _ := eval(0x1p-1074); g >= 0 {
			return 0, true
		}
	}
	sg := -1.0
	if q < p {
		sg = 1
	}
	lo, hi := 0.0, 0.75
	for i := 0; i < 400; i++ {
		g, r := eval(u)
		if g == 0 {
			return u, true
		}
		if g < 0 {
			lo = u
		} else {
			hi = u
		}
		if !sure {
			sure = true
			if g < 0 {
				// the root is above u0: below 1/2?
				if gh, _ := eval(0.5); gh < 0 {
					return 0, false
				}
				hi = 0.5
			}
		}
		step := -g / r
		curv := a - (b-1)*u/(1-u) + sg*r
		if h := step * curv / 2; math.Abs(h) < 0.5 {
			step /= 1 + h
		}
		nu := math.NaN()
		if !math.IsInf(step, 0) && !math.IsNaN(step) && step < 700 {
			nu = float64(u + u*math.Expm1(step))
		}
		if nu == u || math.Abs(step) < 0x1p-52 {
			return nu, true
		}
		if !(nu > lo && nu < hi) {
			if lo == 0 {
				nu = u / 16
			} else {
				nu = float64(math.Sqrt(lo) * math.Sqrt(hi))
			}
		}
		u = nu
	}
	return u, true
}

// ibetaInvTail returns g = log T(u) - log t, T = 1 - I_u(a, b) if upper
// else I_u(a, b), and r = |d log T / d log u| = u^a (1-u)^(b-1) /
// (B(a, b) T); lt = log t, lb = log B(a, b) (lbf in float64).
//
// Where TOMS 708's BPSER applies with a < 1, log T is a double-double from
// that series (the root depends on log T to within a factor 1/a there).
// Elsewhere T is the float64 value of ibetaRatio, which is what the root
// can be located to, and g = log1p((T - t)/t) once T is within a factor 2
// of t (T - t is then exact); r only steers the step and is formed in
// float64.
func ibetaInvTail(a, b, u, t float64, upper bool, lt, lb dd.DD, lbf float64) (g, r float64) {
	bpser := a < 1 && (b <= 1 || b*u <= 0.7)
	if bpser {
		v := ibetaNewXY(u, math.Inf(1))
		ld := dd.Sub(dd.Add(dd.MulF(v.lx, a), dd.MulF(v.ly, b-1)), lb)
		var l dd.DD
		switch {
		case !upper:
			l = ibetaLogBpser(a, b, v)
		case ibetaUpperFromLog(ibetaLogBpser(a, b, v), &l):
		default:
			bpser = false
		}
		if bpser {
			return dd.Sub(l, lt).Float(), math.Exp(dd.Sub(ld, l).Hi)
		}
	}
	w, w1 := ibetaRatio(a, b, ibetaNewXY(u, a+b))
	v := w
	if upper {
		v = w1
	}
	if v <= 0 {
		return math.Inf(-1), math.Inf(1)
	}
	lv := logf(v)
	if v > 0.5*t && v < 2*t {
		g = math.Log1p((v - t) / t)
	} else {
		g = lv - lt.Hi
	}
	ld := a*logf(u) + (b-1)*math.Log1p(-u) - lbf
	return g, math.Exp(ld - lv)
}

// ibetaUpperFromLog sets *lt = log(1 - I) from log I in double-double and
// reports success, unless 1 - I is too small for the double-double
// difference to keep its relative accuracy (then the upper tail is well
// conditioned and its float64 evaluation is used instead).
func ibetaUpperFromLog(li dd.DD, lt *dd.DD) bool {
	q := dd.Neg(igamExpm1DD(li)) // 1 - exp(log I)
	// log I is good to about 2^-100 absolutely; below 2^-45 the float64
	// evaluation of the (then well conditioned) upper tail is better.
	if !(q.Hi > 0x1p-45) {
		return false
	}
	*lt = igamLogDD(q)
	return true
}

// ibetaLogBpser returns log I_x(a, b) as a double-double from the power
// series of ibetaBpser (b <= 1 or b·x <= 0.7):
// log(x^a / (a B(a, b))) + log1p(a Σ).
func ibetaLogBpser(a, b float64, v ibetaXY) dd.DD {
	l := dd.Sub(dd.MulF(v.lx, a), ibetaLogABeta(a, b))
	sum := dd.F(0)
	c := dd.F(1)
	for n := 1.0; n < 100000; n++ {
		c = dd.Mul(c, v.xd)
		c = dd.Sub(c, igamDivF(dd.MulF(c, b), n))
		w := dd.Div(c, sumDD(a, n))
		sum = dd.Add(sum, w)
		if !(math.Abs(w.Hi) > 0x1p-110*math.Max(math.Abs(sum.Hi), 1/a)) {
			break
		}
	}
	return dd.Add(l, igamLog1pDD(dd.MulF(sum, a)))
}

// ibetaInvGuess returns a starting value for the u <= ~1/2 with
// I_u(a, b) = p where a or b is below 1e-3: the power law
// I_u ~ u^a/(a B(a, b)).
func ibetaInvGuess(a, b, p float64, lb dd.DD) float64 {
	lu := (igamLog(p) + math.Log(a) + lb.Hi) / a
	if lu > math.Log(0.5) {
		return 0.5
	}
	if lu < -744 {
		return 0x1p-1074
	}
	return math.Exp(lu)
}
