package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// The last step of the incomplete beta and gamma inverses: correct
// rounding.
//
// The iterations (ibetaInvIterate, igamInvIterate) run on the float64
// forward functions, whose errors (half an ulp of rounding and up to a few
// ulps of method: BFRAC, BGRAT, BASYM and Temme's expansion are float64
// evaluations) exceed the half-gap in p between two neighbouring floats
// wherever r = d log T/d log x is below a few units: the iteration then
// stops on whichever float the noise favours, up to 1.2 ulp from the root
// (on 300 random points with a, b in [e^-1, e^6], p in [0.01, 0.99], 7 to
// 11% of the roots were above half an ulp).
//
// At the iterate the step below evaluates the target tail T again, as a
// double-double from series whose terms are all positive (no cancellation
// can amplify their roundings): for the beta case TOMS 708's BPSER in
// Euler's form (ibetaPosSum) in whichever of x and 1 - x gives T directly,
// for the gamma case the series of P. The power prefix comes from an
// accurate log Γ (logGammaAcc: Stirling's series from 30 on, its first two
// terms in double-double) and an accurate exponential (dd.ExpMantAccurate).
// T is then good to about 2^-84 relative (measured against the betainc and
// gammainc fixtures: 2^-84 to 2^-87 per region). One Newton step in log x
// on log T - log t follows; it is a few ulps at most, so its linearisation
// error is far below an ulp, and x·(1 + expm1(step)), rounded once, is the
// float nearest the root unless the root lies within err(T)/r of the
// midpoint between two floats (relative to x).
//
// Limits, where the iteration's own result stands (within an ulp or so):
//
//   - a target tail that only a complement 1 - (the other tail) gives, an
//     upper gamma tail Q or a beta tail whose own series does not converge
//     in roundMaxTerms terms: its error is 2^-85/T, so a tail below 2^-30 is
//     not used, and between 2^-30 and 2^-17 the rounding is settled except
//     within 2^-85/(T r) of a midpoint;
//   - err(T)/r above 2^-60, where the step would be less certain than the
//     iterate (tiny a: r ~ a/T; the iterations already work in
//     double-double there);
//   - a + b (beta) or x (gamma) beyond 1e5, where the series need many
//     thousands of terms (r is of order sqrt(a) there, and the float64
//     forward functions settle most roundings);
//   - results near the subnormal range.

// roundSplit is the fraction of the sum below which the series' terms
// are summed in float64 rather than in pairs: their relative errors grow
// by about 3·2^-53 per term, so the float64 part stays below 2^-90 of the
// sum for some thousands of terms. (The series are summed to 2^-100.)
const roundSplit = 0x1p-50

// roundMaxTerms bounds the terms of the rounding step's beta series.
const roundMaxTerms = 1 << 14

// roundMaxParam bounds the parameters (a + b for the beta, x for the gamma
// case) for which the rounding step sums its series.
const roundMaxParam = 1e5

// roundMinTail is the smallest tail the rounding step forms as the
// complement of the other one.
const roundMinTail = 0x1p-30

// roundDirectErr is the relative error of a tail from its own series,
// and roundComplementErr the absolute error of a complement 1 - (the
// other tail), both with a margin over what the fixtures show.
const (
	roundDirectErr     = 0x1p-80
	roundComplementErr = 0x1p-84
)

// roundMaxUncertainty is the largest err(T)/r, the uncertainty of the step
// relative to x, at which the rounding step is taken.
const roundMaxUncertainty = 0x1p-60

// stirlingC1 and stirlingC2 are 1/12 and -1/360 as double-doubles.
var (
	stirlingC1 = dd.Div(dd.F(1), dd.F(12))
	stirlingC2 = dd.Div(dd.F(-1), dd.F(360))
)

// stirlingAccMin is the x from which the accurate log Γ takes Stirling's
// series directly: its nine terms stop at the tenth, 1.4/x^19, which is
// 2^-62.6 at x = 10 (enough for stirlingSeries' float64 uses) and below
// 2^-97 from x = 30.
const stirlingAccMin = 30

// stirlingSeriesDD returns Stirling's correction
// log Γ(x) - [(x-1/2) log x - x + log sqrt(2π)] for x >= stirlingAccMin
// as a double-double: the first two terms, 1/(12x) - 1/(360x³), in
// double-double, the rest (below 1/(1260x⁵), so its float64 error is
// below 2^-88) in float64.
func stirlingSeriesDD(x float64) dd.DD {
	if x > 1e10 {
		return dd.DivF(stirlingC1, x) // the rest is below 1/(30x²) of it
	}
	zd := dd.Div(dd.F(1), dd.MulFF(x, x))
	z := zd.Hi
	r := z * z * (stirC3 + z*(stirC4+z*(stirC5+z*(stirC6+z*(stirC7+z*(stirC8+z*stirC9))))))
	return dd.DivF(dd.AddF(dd.Add(stirlingC1, dd.Mul(stirlingC2, zd)), r), x)
}

// logGammaAcc returns log Γ(x) for finite x > 0 as a double-double, to
// about 2^-86 absolutely below 1e5 (measured against the loggamma
// fixture): Stirling's series with everything in double-double for
// x >= stirlingAccMin; below, log Γ(x+n) - log(x(x+1)...(x+n-1)) with
// x+n >= stirlingAccMin (the product in double-double, x+n as a pair whose
// low part enters through ψ to first order).
func logGammaAcc(x float64) dd.DD {
	l, p := logGammaParts(x)
	if p.Hi == 1 && p.Lo == 0 {
		return l
	}
	return dd.Sub(l, igamLogDD(p))
}

// logGammaParts returns L and P with log Γ(x) = L - log P: Stirling's
// form and P = 1 from stirlingAccMin on; below, L = log Γ(x+n) with
// x+n >= stirlingAccMin (its low part through ψ) and P = x(x+1)...(x+n-1)
// in light pairs (each factor's error from TwoSum and TwoProd, one
// renormalisation at the end: about 2^-100 for the 30 factors at most).
// A caller combining several log Γ takes one logarithm for all their P.
func logGammaParts(x float64) (l, p dd.DD) {
	if x >= stirlingAccMin {
		t := dd.Mul(sumDD(x, -0.5), dd.LogAccurate(x))
		t = dd.AddF(t, -x)
		t = dd.Add(t, lnSqrt2Pi)
		return dd.Add(t, stirlingSeriesDD(x)), dd.F(1)
	}
	n := math.Ceil(stirlingAccMin - x)
	ph, pl := x, 0.0
	for k := 1.0; k < n; k++ {
		sh, sl := dd.TwoSum(x, k)
		h := ph * sh
		pl = math.FMA(ph, sh, -h) + ph*sl + pl*sh
		ph = h
	}
	ph, pl = dd.FastTwoSum(ph, pl)
	return logGammaAccDD(sumDD(x, n)), dd.DD{Hi: ph, Lo: pl}
}

// logGammaAccDD is logGammaAcc of a double-double argument, its low part
// to first order (ψ(s) s.Lo).
func logGammaAccDD(s dd.DD) dd.DD {
	return dd.AddF(logGammaAcc(s.Hi), Digamma(s.Hi)*s.Lo)
}

// logBetaAcc returns log B(a, b) for finite a, b > 0 as a double-double
// to about 2^-84 absolutely, and false where it cannot: for a + b <=
// roundMaxParam, log Γ(a) + log Γ(b) - log Γ(a+b) (each term to
// 2^-104·(a+b)·log(a+b)); beyond, for a <= b with x = a/b small enough
// that a·x³ <= 2^-30, the cancellation-free form of logGammaDiff,
//
//	log Γ(a) - a log b - b log1pmx(x) - (a - 1/2) log1p(x) + S(b) - S(a+b),
//
// every term in double-double (the series of log1pmx(x) and log1p(x)
// leave float64 tails of order x⁴ and x³, whose errors times b and a are
// below a·x³·2^-53).
func logBetaAcc(a, b float64) (dd.DD, bool) {
	if a > b {
		a, b = b, a
	}
	if a+b <= roundMaxParam {
		// two logarithms for the three shift products: log(Pa/Ps) + log Pb
		// (Pa/Ps is of order 1 for a <= b; Pa·Pb would underflow for tiny
		// a and b)
		la, pa := logGammaParts(a)
		lb, pb := logGammaParts(b)
		s := sumDD(a, b)
		ls, ps := logGammaParts(s.Hi)
		ls = dd.AddF(ls, Digamma(s.Hi)*s.Lo)
		l := dd.Sub(dd.Add(la, lb), ls)
		l = dd.Sub(l, igamLogDD(dd.Div(pa, ps)))
		if pb.Hi != 1 || pb.Lo != 0 {
			l = dd.Sub(l, igamLogDD(pb))
		}
		return l, true
	}
	xh := a / b
	if !(a*xh*xh*xh <= 0x1p-30) || math.IsInf(b, 0) {
		return dd.DD{}, false
	}
	x := dd.DD{Hi: xh, Lo: math.FMA(-xh, b, a) / b}
	t := dd.Neg(dd.MulF(dd.LogAccurate(b), a))
	t = dd.Sub(t, dd.MulF(igamLog1pmxDD(x), b))
	t = dd.Sub(t, dd.Mul(sumDD(a, -0.5), igamLog1pDD(x)))
	// S(a+b) with a+b = s.Hi + s.Lo: S'(s) = -1/(12s²) to first order
	s := sumDD(a, b)
	ss := dd.AddF(stirlingSeriesDD(s.Hi), -s.Lo/(12*s.Hi*s.Hi))
	t = dd.Add(t, dd.Sub(stirlingSeriesDD(b), ss))
	return dd.Add(logGammaAcc(a), t), true
}

// ibetaLowerAcc returns I_x(a, b) as a double-double from the series of
// ibetaBpserDD summed to 2^-100, for x = xd (exact as a double-double)
// with logarithms lx, ly of x and 1 - x to 2^-104 and lb = log B(a, b); ok
// is false where the result nears the subnormal range or the series does
// not converge within roundMaxTerms terms.
func ibetaLowerAcc(a, b float64, xd, lx, ly, lb dd.DD) (v dd.DD, ok bool) {
	x := xd.Hi
	k0 := 0
	if r := ((a+b-1)*x - a) / (1 - x); r >= 1 {
		if r > roundMaxTerms {
			return dd.DD{}, false
		}
		k0 = int(r)
	}
	// The exponent cancels terms of size a|log x|, b|log(1-x)| and |log B|,
	// each good to about 2^-102 relative (measured: up to 2^-98 in all):
	// up to 2^17 their sum's absolute error, the prefix's relative one,
	// stays within roundDirectErr.
	if !(math.Abs(a*lx.Hi)+math.Abs(b*ly.Hi)+math.Abs(lb.Hi) <= 0x1p17) {
		return dd.DD{}, false
	}
	l := dd.Sub(dd.Add(dd.MulF(lx, a), dd.MulF(ly, b)), lb)
	m, k := dd.ExpMantAccurate(l)
	if k < -960 {
		return dd.DD{}, false
	}
	s, ok := ibetaPosSumSplit(a, b, xd, k0, roundMaxTerms, 0x1p-92, roundSplit)
	if !ok {
		return dd.DD{}, false
	}
	return dd.Ldexp(dd.Mul(igamDivF(m, a), s), k), true
}

// complementAcc returns 1 - v and its relative error, for a tail v from
// a series (absolute error at most roundComplementErr), or false where
// 1 - v is below roundMinTail.
func complementAcc(v dd.DD) (dd.DD, float64, bool) {
	c := dd.Sub(dd.F(1), v)
	if !(c.Hi >= roundMinTail) {
		return dd.DD{}, 0, false
	}
	return c, roundComplementErr / c.Hi, true
}

// ibetaTailAcc returns the tail T of I_u(a, b) (the upper one, 1 - I, if
// upper) as a double-double for the rounding step, with its relative
// error, a + b <= roundMaxParam: from the series of that tail itself (in u
// for the lower, in 1 - u with a and b exchanged for the upper) where its
// variable is at most 0.95 and it converges within roundMaxTerms terms,
// else as the complement of the other tail's series.
func ibetaTailAcc(a, b, u float64, upper bool, lb dd.DD) (dd.DD, float64, bool) {
	var xd, yd, lx, ly dd.DD
	if u <= 0.5 {
		xd, yd = dd.F(u), sumDD(1, -u)
		lx, ly = dd.LogAccurate(u), igamLog1pDD(dd.F(-u))
	} else {
		y := 1 - u // exact
		xd, yd = dd.F(u), dd.F(y)
		lx, ly = igamLog1pDD(dd.F(-y)), dd.LogAccurate(y)
	}
	// own: the series whose sum is T; other: the one whose sum is 1 - T
	own := func() (dd.DD, bool) { return ibetaLowerAcc(a, b, xd, lx, ly, lb) }
	other := func() (dd.DD, bool) { return ibetaLowerAcc(b, a, yd, ly, lx, lb) }
	ownVar, nOwn, nOther := u, ibetaSeriesTerms(a, b, u), ibetaSeriesTerms(b, a, yd.Hi)
	if upper {
		own, other, ownVar, nOwn, nOther = other, own, yd.Hi, nOther, nOwn
	}
	// the tail's own series first, unless it would take many more terms
	// than the complement's (a series that runs to roundMaxTerms without
	// converging costs as much as dozens of evaluations)
	ownFirst := ownVar <= 0.95 && nOwn <= max(2000, 4*nOther)
	if ownFirst {
		if v, ok := own(); ok {
			return v, roundDirectErr, true
		}
	}
	if v, ok := other(); ok {
		if c, err, ok := complementAcc(v); ok {
			return c, err, true
		}
	}
	if !ownFirst && ownVar <= 0.95 {
		if v, ok := own(); ok {
			return v, roundDirectErr, true
		}
	}
	return dd.DD{}, 0, false
}

// ibetaSeriesTerms estimates the terms ibetaLowerAcc sums for I_x(a, b):
// up to the largest, k0 = ((a+b-1)x - a)/(1-x), then the decrease, about
// 12 sqrt((a+1)/(1-x)) near the mean and 70/log(1/x) for the geometric
// tail (to 2^-100). It only orders the two series of ibetaTailAcc.
func ibetaSeriesTerms(a, b, x float64) float64 {
	k0 := max(0, ((a+b-1)*x-a)/(1-x))
	return k0 + 12*math.Sqrt((a+1)/(1-x)) + 70/-math.Log(x)
}

// igamTailAcc returns P(a, x) (Q(a, x) if upper) as a double-double for
// the rounding step, with its relative error, x <= roundMaxParam and
// x - a <= 9000 (the series rises for x - a terms before it falls; the
// sum stops at 10000): P from the series of igamSeriesPMant summed to
// 2^-100 with the prefix from logGammaAcc, Q as its complement.
func igamTailAcc(a, x float64, upper bool) (dd.DD, float64, bool) {
	if x > roundMaxParam || x-a > 9000 {
		return dd.DD{}, 0, false
	}
	var l dd.DD
	if a >= stirlingAccMin && x > 0x1p-20*a {
		// a·log1pmx((x-a)/a) + log(a)/2 - log sqrt(2π) - S(a), the form
		// of igamLogPrefix that does not cancel for x near a
		d := igamDivF(sumDD(x, -a), a)
		l = dd.MulF(igamLog1pmxDD(d), a)
		l = dd.Add(l, dd.MulF(dd.LogAccurate(a), 0.5))
		l = dd.Sub(l, lnSqrt2Pi)
		l = dd.Sub(l, stirlingSeriesDD(a))
	} else {
		l = dd.AddF(dd.MulF(dd.LogAccurate(x), a), -x)
		l = dd.Sub(l, logGammaAcc(a))
	}
	m, k := dd.ExpMantAccurate(l)
	if k < -960 {
		return dd.DD{}, 0, false
	}
	s, ok := igamLowerSeriesRestTol(a, x, roundSplit, 0x1p-100)
	if !ok {
		return dd.DD{}, 0, false
	}
	v := dd.Ldexp(igamDivF(dd.Mul(m, dd.AddF(s, 1)), a), k)
	if upper {
		return complementAcc(v)
	}
	return v, roundDirectErr, true
}

// ibetaInvRound returns the float nearest the x with I_x(a, b) = p
// (1 - I_x = q), from the iterate x, given lb = logBetaAcc(a, b) (see the
// comment at the top).
func ibetaInvRound(a, b, p, q, x float64, lb dd.DD) float64 {
	if !(x > 0 && x < 1) {
		return x
	}
	upper := q < p
	t, sg := p, 1.0
	if upper {
		t, sg = q, -1
	}
	tx, err, ok := ibetaTailAcc(a, b, x, upper, lb)
	if !ok {
		return x
	}
	// r = x·x^(a-1)(1-x)^(b-1)/(B T), 1 - x exact where x >= 1/2
	ly := math.Log1p(-x)
	if x >= 0.5 {
		ly = math.Log(1 - x)
	}
	r := math.Exp(a*math.Log(x) + (b-1)*ly - lb.Hi - math.Log(tx.Hi))
	return roundStep(x, tx, t, r, sg, err)
}

// igamInvRound returns the float nearest the x with T(a, x) = t (T = Q if
// upper, else P), from the iterate x (see the comment at the top).
func igamInvRound(a, t float64, upper bool, x float64) float64 {
	if !(x > 0x1p-1022) {
		return x
	}
	tx, err, ok := igamTailAcc(a, x, upper)
	if !ok {
		return x
	}
	sg := 1.0
	if upper {
		sg = -1
	}
	r := math.Exp(a*math.Log(x) - x - LogGamma(a) - math.Log(tx.Hi))
	return roundStep(x, tx, t, r, sg, err)
}

// roundStep returns the float nearest the root of log T(x) = log t from x
// (a converged iterate, within a few ulps of the root), given T(x) as a
// double-double with relative error err and r = d log T/d log x (in
// float64: it scales a step of a few ulps): one Newton step in log x,
// x·(1 + expm1(step)) rounded once. sg is +1 if T increases with x, -1 if
// it decreases. The iterate is returned unchanged where the step would be
// less certain than roundMaxUncertainty or is not a correction of a few
// ulps.
func roundStep(x float64, tx dd.DD, t, r, sg, err float64) float64 {
	if !(err <= roundMaxUncertainty*r) {
		return x
	}
	g := dd.AddF(tx, -t).Float() / t // T/t - 1, the difference exact to 2^-104
	if !(math.Abs(g) < 0x1p-30) {
		// T is not within a few ulps of t: one float away from the
		// iterate it changes by more than the linearisation can follow
		// (T ~ (1-x)^b, b small, next to x = 1), or the iterate is not
		// converged; either way the step is not a rounding correction.
		return x
	}
	step := -sg * math.Log1p(g) / r
	if !(math.Abs(step) < 0x1p-40) {
		return x
	}
	return x + x*math.Expm1(step)
}
