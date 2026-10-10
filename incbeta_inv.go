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
	// Which side of 1/2: the sign of the x-side residual at 1/2, from the
	// same (double-double where needed) evaluation as the iteration, so a
	// nearly flat I (tiny a and b) is not decided by a rounded I_(1/2).
	if a == b && p == q {
		return 0.5 // symmetry
	}
	g := ibetaInvEval(a, b, p, q)
	switch g0, _ := g(0.5); {
	case g0 == 0:
		return 0.5
	case g0 > 0:
		return ibetaInvSide(a, b, p, q)
	}
	// I_x(a,b) = p  <=>  I_(1-x)(b,a) = q
	u := ibetaInvSide(b, a, q, p)
	return 1 - u
}

// ibetaInvEval returns g(u) = ±(log T(u) - log t), increasing in u, and
// r = dg/d log u > 0, where T is the smaller-target tail of I_u(a, b)
// (lower with target p, or upper with target q).
func ibetaInvEval(a, b, p, q float64) func(u float64) (g, r float64) {
	upper := q < p
	t := p
	if upper {
		t = q
	}
	lt := dd.Log(t)
	lb := ibetaLogBeta(a, b)
	return func(u float64) (g, r float64) {
		var l dd.DD
		l, r = ibetaInvTail(a, b, u, upper, lb)
		if math.IsInf(l.Hi, -1) {
			g = math.Inf(-1)
		} else {
			g = dd.Sub(l, lt).Float()
		}
		if upper {
			g = -g
		}
		return g, r
	}
}

// ibetaInvSide returns the u (at most about 1/2) with I_u(a, b) = p and
// 1 - I_u(a, b) = q, matching the smaller of the two.
func ibetaInvSide(a, b, p, q float64) float64 {
	eval := ibetaInvEval(a, b, p, q)
	if g, _ := eval(0x1p-1074); g >= 0 {
		return 0 // the root is below the float64 range
	}
	u := ibetaInvGuess(a, b, p, ibetaLogBeta(a, b))
	lo, hi := 0.0, 0.75
	for i := 0; i < 400; i++ {
		g, r := eval(u)
		if g == 0 {
			return u
		}
		if g < 0 {
			lo = u
		} else {
			hi = u
		}
		step := -g / r
		nu := math.NaN()
		if !math.IsInf(step, 0) && !math.IsNaN(step) && step < 700 {
			nu = float64(u + u*math.Expm1(step))
		}
		if nu == u || math.Abs(step) < 0x1p-52 {
			return nu
		}
		if !(nu > lo && nu < hi) {
			if lo == 0 {
				nu = u / 16
			} else {
				nu = float64(math.Sqrt(lo) * math.Sqrt(hi))
			}
			if nu <= lo || nu >= hi {
				return ibetaInvClosest(eval, lo, hi)
			}
		}
		u = nu
	}
	return u
}

// ibetaInvClosest picks, between two adjacent floats lo < hi bracketing
// the root, the one whose residual is smaller.
func ibetaInvClosest(eval func(u float64) (float64, float64), lo, hi float64) float64 {
	if lo == 0 {
		return hi
	}
	gl, _ := eval(lo)
	gh, _ := eval(hi)
	if -gl <= gh {
		return lo
	}
	return hi
}

// ibetaInvTail returns log T(u) as a double-double, T = 1 - I_u(a, b) if
// upper else I_u(a, b), and r = |d log T / d log u| = u^a (1-u)^(b-1) /
// (B(a, b) T); lb = log B(a, b).
func ibetaInvTail(a, b, u float64, upper bool, lb dd.DD) (lt dd.DD, r float64) {
	v := ibetaNewXY(u)
	ld := dd.Sub(dd.Add(dd.MulF(v.lx, a), dd.MulF(v.ly, b-1)), lb)
	bpser := a < 1 && (b <= 1 || b*u <= 0.7)
	switch {
	case bpser && !upper:
		lt = ibetaLogBpser(a, b, v)
	case bpser && ibetaUpperFromLog(ibetaLogBpser(a, b, v), &lt):
	default:
		w, w1 := ibetaRatio(a, b, v)
		t := w
		if upper {
			t = w1
		}
		if t <= 0 {
			return dd.F(math.Inf(-1)), math.Inf(1)
		}
		lt = logDD(dd.F(t))
	}
	return lt, math.Exp(dd.Sub(ld, lt).Hi)
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
	*lt = logDD(q)
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
		c = dd.Sub(c, dd.DivF(dd.MulF(c, b), n))
		w := dd.Div(c, sumDD(a, n))
		sum = dd.Add(sum, w)
		if !(math.Abs(w.Hi) > 0x1p-110*math.Max(math.Abs(sum.Hi), 1/a)) {
			break
		}
	}
	return dd.Add(l, log1pDD(dd.MulF(sum, a)))
}

// ibetaInvGuess returns a starting value for the u <= ~1/2 with
// I_u(a, b) = p: the normal approximation (mean a/(a+b), variance
// ab/((a+b)²(a+b+1))) when a and b are both at least 1 and a+b >= 10,
// else the power law I_u ~ u^a/(a B(a, b)).
func ibetaInvGuess(a, b, p float64, lb dd.DD) float64 {
	if a >= 1 && b >= 1 && a+b >= 10 {
		s := a + b
		m := a / s
		sd := math.Sqrt(a*b/(s*s*(s+1))) / 1
		u := m + Ndtri(p)*sd
		if u > 0 && u <= 0.75 {
			return u
		}
	}
	lu := (igamLog(p) + math.Log(a) + lb.Hi) / a
	if lu > math.Log(0.5) {
		return 0.5
	}
	if lu < -744 {
		return 0x1p-1074
	}
	return math.Exp(lu)
}
