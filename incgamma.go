package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// The regularised incomplete gamma functions
//
//	P(a, x) = γ(a, x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt,
//	Q(a, x) = Γ(a, x)/Γ(a) = 1 - P(a, x),
//
// for a > 0, x >= 0 (DLMF 8.2.4).
//
// Algorithm. The choice of method follows Boost.Math's
// gamma_incomplete_imp (boost/math/special_functions/gamma.hpp, BSL-1.0,
// after DiDonato & Morris, "Computation of the incomplete gamma function
// ratios and their inverse", ACM TOMS 12(4), 1986) and SciPy's Cephes
// igam/igamc (xsf/cephes/igam.h, BSD-3-Clause), which both compute the
// smaller of P and Q directly and the other as its complement:
//
//   - a > 20 and |x-a| < 0.3a: Temme's uniform asymptotic expansion
//     (DLMF 8.12.3-8.12.4, coefficients from Cephes' igam_asymp_coeff.h);
//   - otherwise, P by the power series (DLMF 8.11.4 / 8.7.1) when x is
//     below the changeover of DiDonato & Morris, and Q by Legendre's
//     continued fraction (DLMF 8.9.2, modified Lentz) above it, or for small
//     a and x < 1.1 by the series for Γ(a, x) built on Γ(1+a) - 1 and
//     x^a - 1 (Boost's tgamma_small_upper_part, DLMF 8.7.3).
//
// What is done differently, for accuracy: the factor x^a e^(-x)/Γ(a)
// that multiplies the series and the fraction (where most of the error of
// the classical implementations comes from) is formed as a logarithm in
// double-double and exponentiated once (dd.ExpMant): for a >= 10 through
// a·log1pmx((x-a)/a) and Stirling's series, which do not cancel however
// large a is, instead of the Lanczos form with float64 pow. In Temme's
// expansion the argument η sqrt(a/2) of erfc and the Gaussian factor
// exp(-aη²/2) come from the same double-double -a·log1pmx((x-a)/a), so
// the result keeps its relative accuracy in the tails (the float64
// evaluation loses about 2z² ulp there, z = η sqrt(a/2)).

// GammaP returns the regularised lower incomplete gamma function
// P(a, x) = γ(a, x)/Γ(a) = (1/Γ(a)) ∫₀ˣ t^(a-1) e^(-t) dt (DLMF 8.2.4).
//
// Domain: a >= 0, x >= 0. Special values: P(a, 0) = 0 for a > 0;
// P(a, +Inf) = 1; P(0, x) = 1 for x > 0; P(+Inf, x) = 0 for finite x;
// NaN for a < 0, x < 0, a = x = 0, a = x = +Inf, or a NaN argument.
//
// Algorithm: see the comment at the top of incgamma.go (DiDonato & Morris
// / Boost.Math method selection, Temme's expansion, double-double prefix).
func GammaP(a, x float64) float64 {
	if p, _, done := igamEdges(a, x); done {
		return p
	}
	p, _ := igamPQ(a, x)
	return p
}

// GammaQ returns the regularised upper incomplete gamma function
// Q(a, x) = Γ(a, x)/Γ(a) = 1 - P(a, x) (DLMF 8.2.4), computed directly (not
// as 1 - P) where it is the smaller of the two, so it keeps its relative
// accuracy in the upper tail down to the subnormal range.
//
// Domain and special values: as GammaP, with Q = 1 - P at the edges
// (Q(a, 0) = 1, Q(a, +Inf) = 0, Q(0, x) = 0, Q(+Inf, x) = 1).
//
// Algorithm: as GammaP.
func GammaQ(a, x float64) float64 {
	if _, q, done := igamEdges(a, x); done {
		return q
	}
	_, q := igamPQ(a, x)
	return q
}

// igamEdges handles the special values of P and Q (SciPy's conventions).
func igamEdges(a, x float64) (p, q float64, done bool) {
	nan := math.NaN()
	switch {
	case math.IsNaN(a) || math.IsNaN(x) || a < 0 || x < 0:
		return nan, nan, true
	case a == 0:
		if x == 0 {
			return nan, nan, true
		}
		return 1, 0, true
	case math.IsInf(a, 1):
		if math.IsInf(x, 1) {
			return nan, nan, true
		}
		return 0, 1, true
	case x == 0:
		return 0, 1, true
	case math.IsInf(x, 1):
		return 1, 0, true
	}
	return 0, 0, false
}

// igamPQ returns P(a, x) and Q(a, x) for finite a > 0, x > 0.
func igamPQ(a, x float64) (p, q float64) {
	if a > 20 && math.Abs(x-a) < 0.3*a {
		return igamTemme(a, x)
	}
	usePSeries := false
	switch {
	case x < 0.5:
		usePSeries = -0.4/igamLog(x) < a
	case x < 1.1:
		usePSeries = 0.75*x < a
	default:
		usePSeries = x-1/(3*x) < a
	}
	switch {
	case usePSeries:
		return igamPair(igamSeriesPMant(a, x))
	case x < 1.1:
		q, p = igamPair(igamSmallQDD(a, x), 0)
	default:
		q, p = igamPair(igamFractionQMant(a, x))
	}
	return p, q
}

// igamLog returns log(x) for x > 0, subnormal x included (math.Log is
// wrong for subnormal arguments on some platforms; logf normalises). Its
// uses are branch decisions and starting values.
func igamLog(x float64) float64 {
	return logf(x)
}

// igamLogPrefix returns log(x^a e^(-x) / Γ(a)) as a double-double, for
// finite a > 0, x > 0.
//
// For a >= 10 and x not far below a, Stirling's series for log Γ(a)
// (DLMF 5.11.1) turns it into
//
//	a·log1pmx(d) + log(a)/2 - log sqrt(2π) - S(a),   d = (x-a)/a,
//
// where every term is small when x ≈ a: the direct form
// a log x - x - log Γ(a) cancels to the last bits of terms of size a log a.
func igamLogPrefix(a, x float64) dd.DD {
	if a >= 10 && x > 0x1p-20*a {
		d := igamDivF(sumDD(x, -a), a) // x-a exact, then one division
		l := dd.MulF(igamLog1pmxDDScaled(d, a), a)
		l = dd.Add(l, dd.MulF(igamLogScaled(a, 0.5), 0.5))
		l = dd.Sub(l, lnSqrt2Pi)
		return dd.AddF(l, -stirlingSeries(a))
	}
	l := dd.MulF(igamLogScaled(x, a), a)
	l = dd.AddF(l, -x)
	return dd.Sub(l, logGammaPos(a))
}

// igamSeriesPMant returns P(a, x) by the series
// P = x^a e^(-x)/Γ(a+1) Σ_{n>=0} x^n / ((a+1)(a+2)...(a+n)) (DLMF 8.7.1),
// whose terms are all positive, as v 2^k.
func igamSeriesPMant(a, x float64) (v dd.DD, k int) {
	m, k := dd.ExpMant(igamLogPrefix(a, x))
	if m.Hi == 0 {
		return dd.DD{}, 0
	}
	s := igamLowerSeries(a, x)
	return igamDivF(dd.Mul(m, s), a), k
}

// igamLowerSeries returns S = Σ_{n>=0} x^n / ((a+1)...(a+n)) as a
// double-double (1 + igamLowerSeriesRest).
func igamLowerSeries(a, x float64) dd.DD {
	return dd.AddF(igamLowerSeriesRest(a, x), 1)
}

// igamLowerSeriesRest returns S - 1 = Σ_{n>=1} x^n / ((a+1)...(a+n)) as a
// double-double. The terms are positive. While they are above 2^-16 of
// the sum each is carried as an unrenormalised pair (the factor x/(a+n) to
// about 2^-104 from one division and its FMA remainder, a+n from TwoSum,
// the product with TwoProd's error, the sum with TwoSum's): their float64
// recurrence accumulated about two roundings per term, an ulp or two of S
// once a few terms count, and P near 1/2 passed that on to Q = 1 - P. The
// rest is summed in float64 (its weight keeps its error below 2^-60 of S).
func igamLowerSeriesRest(a, x float64) dd.DD {
	th, tl := 1.0, 0.0
	sh, sl := 0.0, 0.0
	n := 1.0
	for ; n < 10000; n++ {
		dh, de := dd.TwoSum(a, n)
		r := 1 / dh
		fh := x * r
		fl := (math.FMA(-fh, dh, x) - fh*de) * r
		ph := float64(th * fh) // rounded: it feeds the TwoSum below
		tl = math.FMA(th, fh, -ph) + (th*fl + tl*fh)
		th = ph
		var e float64
		sh, e = dd.TwoSum(sh, th)
		sl += e + tl
		if th <= 0x1p-16*sh {
			n++
			break
		}
	}
	tail := 0.0
	for ; n < 10000 && th > 0x1p-60*sh; n++ {
		th *= x / (a + n)
		tail += th
	}
	h, l := dd.FastTwoSum(sh, sl+tail)
	return dd.DD{Hi: h, Lo: l}
}

// igamFractionQMant returns Q(a, x) = v 2^k by Legendre's continued
// fraction Γ(a,x) = x^a e^(-x) / (x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a-
// ...))) (DLMF 8.9.2; igamFractionDenDD).
func igamFractionQMant(a, x float64) (v dd.DD, k int) {
	m, k := dd.ExpMant(igamLogPrefix(a, x))
	if m.Hi == 0 {
		return dd.DD{}, 0
	}
	return dd.Div(m, igamFractionDenDD(a, x)), k
}

// igamFractionDen returns the denominator x+1-a - 1(1-a)/(x+3-a - ...) of
// igamFractionQMant, for x >= 1.1 and x - 1/(3x) >= a: b_0 + a_1/(b_1 + a_2/(b_2
// + ...)) with b_k = x - a + 1 + 2k and a_k = k(a - k).
//
// Two passes, neither with a division per step. The forward pass runs the
// three-term recurrences of the convergents A_k/B_k (Wallis) only to find
// the depth: the step |A_k/B_k - A_(k-1)/B_(k-1)| = |D_k|/|B_k B_(k-1)|,
// D_k = Π a_j, is exact up to a few roundings (it does not come from a
// difference), and it is compared with the value every fourth step, the
// tolerance scaled by the observed rate of decrease (when the steps shrink
// by a ratio close to 1, what remains is many times the last one). The
// value is then summed backwards from beyond that depth, t_k = a_k/(b_k +
// t_(k+1)), which is stable for this fraction (forward products accumulate
// one rounding per step, several ulp near the transition x ~ a); t is
// carried as P/Q with P_k = a_k Q_(k+1), Q_k = b_k Q_(k+1) + P_(k+1), so the
// common rounding of Q cancels in the ratio and one division remains; the
// last step, whose rounding reaches the result undamped, is done with the
// partial denominator in double-double.
// Both passes rescale by powers of two (exact) to stay in range.
func igamFractionDen(a, x float64) float64 { return igamFractionDenDD(a, x).Float() }

// igamFractionDenDD is igamFractionDen as a double-double.
func igamFractionDenDD(a, x float64) dd.DD {
	z0 := dd.AddF(sumDD(x, -a), 1)
	// The equivalence transformation b_k -> b_k/s, a_k -> a_k/s², s = 2^e
	// the binade of b_0, keeps every term near 1 whatever the size of x and
	// a (exactly: s is a power of two), so the recurrences below grow by a
	// bounded factor per step.
	_, e := math.Frexp(z0.Hi)
	sc := math.Ldexp(1, -e)
	b0 := z0.Hi * sc
	a2, a1 := 1.0, b0 // A_(k-2), A_(k-1)
	b2, b1 := 0.0, 1.0
	dk := 1.0
	prev := 1.0
	n := 0.0
	sc2 := sc * sc
	step := func(k float64) {
		ak := k * (a - k) * sc2
		bk := b0 + 2*k*sc
		a2, a1 = a1, bk*a1+ak*a2
		b2, b1 = b1, bk*b1+ak*b2
		dk *= ak
	}
	// Four steps between checks. An integer a ends the fraction (a_k = 0 at
	// k = a): D_k is 0 from there on and the test below stops.
	for k := 1.0; k < 10000; k += 4 {
		step(k)
		step(k + 1)
		step(k + 2)
		step(k + 3)
		n = k + 3
		if m := math.Abs(b1); m > 0x1p256 || m < 0x1p-256 {
			f := igamUnscale(m)
			a1, a2, b1, b2, dk = a1*f, a2*f, b1*f, b2*f, dk*f*f
		}
		cur := math.Abs(dk / (a1 * b2))
		rho := cur / prev
		prev = cur
		// remaining <= cur/(1 - rho^(1/4)) <= 4 cur/(1 - rho)
		if lim := 1 - rho; cur <= 0x1p-58*max(lim*0.25, 1.0/256) {
			break
		}
	}
	// Summed backwards from beyond that depth (the truncated fraction is
	// exactly the convergent there), two steps between range checks.
	p, q := 0.0, 1.0
	for k := n + 3; k >= 3; k -= 2 {
		p, q = k*(a-k)*sc2*q, (b0+2*k*sc)*q+p
		k1 := k - 1
		p, q = k1*(a-k1)*sc2*q, (b0+2*k1*sc)*q+p
		if m := math.Abs(q); m > 0x1p256 || m < 0x1p-256 {
			f := igamUnscale(m)
			p, q = p*f, q*f
		}
	}
	// The first step, which the result feels undamped, in double-double:
	// den = z0 + (a-1)/(z0 + 2 + t2).
	w := dd.AddF(dd.AddF(z0, 2), p/q/sc)
	t1 := (a - 1) / w.Hi
	t1 += (math.FMA(-t1, w.Hi, a-1) - t1*w.Lo) / w.Hi
	return dd.AddF(z0, t1)
}

// igamUnscale returns the power of two that brings m > 0 into [1/2, 1)
// (an exact rescaling of a recurrence).
func igamUnscale(m float64) float64 {
	_, e := math.Frexp(m)
	return math.Ldexp(1, -e)
}

// igamSmallQDD returns Q(a, x) for x < 1.1 and the a below DiDonato &
// Morris' changeover (a < 0.83), from
//
//	Γ(a, x) = (Γ(1+a) - x^a)/a - x^a Σ_{n>=1} (-x)^n / (n! (a+n))
//
// (DLMF 8.7.3; Boost's tgamma_small_upper_part), multiplied by a/Γ(1+a):
//
//	Q = [ (Γ(1+a)-1) - (x^a-1) - a x^a Σ ] / Γ(1+a).
//
// Γ(1+a)-1 and x^a-1 are double-doubles, so their difference does not
// cancel and nothing overflows for tiny a. The result is a double-double
// accurate to a few units of 2^-53 times the alternating sum's share of
// the result, which is small: the leading difference is exact.
func igamSmallQDD(a, x float64) dd.DD {
	gm1 := gamma1pm1(a)
	pm1 := igamExpm1DD(dd.MulF(dd.LogAccurate(x), a)) // x^a - 1
	// Σ_{n>=1} (-x)^n / (n! (a+n)) in double-double: it partly cancels
	// against the leading difference (by a factor of about 4 near x = 1).
	s := dd.F(0)
	t := dd.F(1)
	for n := 1.0; n < 200; n++ {
		t = igamDivF(dd.MulF(t, -x), n)
		term := dd.Div(t, sumDD(a, n))
		s = dd.Add(s, term)
		if math.Abs(term.Hi) <= 0x1p-106*math.Abs(s.Hi) {
			break
		}
	}
	xa := dd.AddF(pm1, 1)
	num := dd.Sub(gm1, pm1)
	num = dd.Sub(num, dd.MulF(dd.Mul(xa, s), a))
	return dd.Div(num, dd.AddF(gm1, 1))
}

// igamExpm1DD returns exp(u) - 1 for a double-double u with |u| not large,
// as a double-double: expm1(u.Hi) + u.Lo·exp(u.Hi) with expm1(u.Hi) itself
// refined by one Newton step on log1p (relative error about 2^-100).
func igamExpm1DD(u dd.DD) dd.DD {
	if u.Hi < -30 {
		// e^u < 2^-43: -1 + e^u directly. (Below about -36.7, expm1 rounds
		// to -1 exactly and the Newton step below would take log1p(-1).)
		return dd.AddF(dd.F(-1), math.Exp(u.Hi)*(1+u.Lo))
	}
	e := math.Expm1(u.Hi)
	if e == 0 {
		return u
	}
	// Newton on f(e) = log1p(e) - u: e' = e - (log1p(e) - u)(1 + e).
	r := dd.Sub(igamLog1pDD(dd.F(e)), u)
	ed := dd.Sub(dd.F(e), dd.Mul(r, dd.AddF(dd.F(e), 1)))
	return ed
}

// igamTemme returns P(a, x) and Q(a, x) by Temme's uniform asymptotic
// expansion (DLMF 8.12.3-8.12.4, 8.12.8):
//
//	Q = erfc(z)/2 + R,  P = erfc(-z)/2 - R,  z = η sqrt(a/2),
//	η²/2 = λ - 1 - log λ, λ = x/a, sign(η) = sign(λ - 1),
//	R = exp(-z²)/sqrt(2πa) Σ_k C_k(η) a^(-k),
//
// with the C_k(η) from their Taylor coefficients in η (Cephes'
// asymptotic_series in igam.h; table igamTemmeCoef), summed by the
// generated straight-line igamTemmeSum. The smaller of P and Q
// is computed directly. z² = -a·log1pmx((x-a)/a) is a double-double, and
// for |z| >= 1/2 the result is exp(-z²)·(erfcx(|z|)/2 ± R'), so the
// Gaussian factor is applied once from the exact exponent.
func igamTemme(a, x float64) (p, q float64) {
	d := igamDivF(sumDD(x, -a), a)
	lpm := igamLog1pmxDDScaled(d, a) // <= 0
	z2 := dd.Neg(dd.MulF(lpm, a))
	eta := math.Sqrt(-2 * lpm.Hi)
	upper := x > a // compute Q directly; else P
	if !upper {
		eta = -eta
	}
	// Σ_k C_k(η) a^-k (zz_igamTemmeSum_gen.go: each row cut where |η| <=
	// 0.34 and a >= 20 allow, to 2^-56 of the sum: R is at most ~7% of the
	// result).
	sum := igamTemmeSum(eta, 1/a)
	// Q = erfc(z)/2 + R (η > 0); P = erfc(|z|)/2 - R (η < 0), where
	// R = exp(-z²) sum/sqrt(2πa) with sum computed at the signed η.
	r := sum / math.Sqrt(2*math.Pi*a)
	if !upper {
		r = -r
	}
	z := dd.Sqrt(z2)
	var w dd.DD
	k := 0
	if z.Hi >= 0.5 {
		var m dd.DD
		m, k = dd.ExpMant(dd.Neg(z2))
		base := dd.AddF(dd.MulF(erfcxCorrected(z), 0.5), r)
		w = dd.Mul(m, base)
	} else {
		e := dd.MulF(erfcDD(z), 0.5)
		w = dd.AddF(e, r*math.Exp(-z2.Hi))
	}
	v, c := igamPair(w, k)
	if upper {
		return c, v
	}
	return v, c
}

// igamPair returns v = w 2^k rounded and its complement 1 - v formed from
// the double-double (so the complement does not inherit the rounding of v,
// which is up to an ulp of the complement when v is above 1/2).
func igamPair(w dd.DD, k int) (v, c float64) {
	v = math.Ldexp(w.Hi+w.Lo, k)
	if k < -60 {
		return v, 1 - v
	}
	h := math.Ldexp(w.Hi, k)
	return v, (1 - h) - math.Ldexp(w.Lo, k)
}

// igamDivF returns x/b for a double-double x and a float64 b, to about
// 2^-104 relative (Joldes, Muller & Popescu's DWDivFP with the remainder
// x.Hi - q·b taken exactly by an FMA): a quarter of dd.DivF, which runs the
// general double-double division.
func igamDivF(x dd.DD, b float64) dd.DD {
	q := x.Hi / b
	l := (math.FMA(-q, b, x.Hi) + x.Lo) / b
	h, e := dd.FastTwoSum(q, l)
	return dd.DD{Hi: h, Lo: e}
}

// igamLogScaled returns log(x) for x > 0 as a double-double whose absolute
// error, once multiplied by scale, stays below about 2^-64: dd.Log (2^-72
// relative, about five times cheaper) where scale·|log x| < 180, read off
// the exponent of x, and dd.LogAccurate (2^-104) beyond or for subnormal x.
func igamLogScaled(x, scale float64) dd.DD {
	if e := int(math.Float64bits(x)>>52) - 1023; e > -1023 && scale*float64(max(e, -e)+1) < 256 {
		return dd.Log(x)
	}
	return dd.LogAccurate(x)
}

// igamLog1pDDScaled is igamLog1pDD with the precision chosen as in
// igamLogScaled: dd.Log where scale·|u| < 128 and u > -1/2, so that
// scale·|log1p(u)| < 180.
func igamLog1pDDScaled(u dd.DD, scale float64) dd.DD {
	if math.Abs(u.Hi) < 0x1p-20 || !(u.Hi > -0.5 && scale*math.Abs(u.Hi) < 128) {
		return igamLog1pDD(u)
	}
	h, l := dd.TwoSum(1, u.Hi)
	g := dd.AddF(dd.DD{Hi: h, Lo: l}, u.Lo)
	return dd.AddF(dd.Log(g.Hi), g.Lo/g.Hi)
}

// igamLog1pmxDDScaled is igamLog1pmxDD with the logarithm from
// igamLog1pDDScaled: the error of log1p(x) - x is that of log1p(x).
func igamLog1pmxDDScaled(x dd.DD, scale float64) dd.DD {
	if math.Abs(x.Hi) >= 0x1p-10 {
		return dd.Sub(igamLog1pDDScaled(x, scale), x)
	}
	return igamLog1pmxDD(x)
}

// igamLog1pDD returns log(1 + u) for a double-double u > -1, as log1pDD
// does, but with dd.LogAccurate (about 2^-104) for the logarithm: the
// incomplete functions multiply it by a or b, up to 1e15 and beyond.
func igamLog1pDD(u dd.DD) dd.DD {
	if math.Abs(u.Hi) < 0x1p-20 {
		x := u.Hi
		tail := x * x * x * (1.0/3 - x*(1.0/4-x*(1.0/5-x/6)))
		return dd.AddF(dd.Sub(u, dd.MulF(dd.Mul(u, u), 0.5)), tail)
	}
	h, l := dd.TwoSum(1, u.Hi)
	return igamLogDD(dd.AddF(dd.DD{Hi: h, Lo: l}, u.Lo))
}

// igamLogDD returns log(g) for a double-double g > 0 (dd.LogAccurate).
func igamLogDD(g dd.DD) dd.DD {
	return dd.AddF(dd.LogAccurate(g.Hi), g.Lo/g.Hi)
}

// igamLog1pmxDD returns log(1+x) - x for a double-double x > -1: the
// series -x²/2 + x³/3 - ... for |x| < 2^-10 (first two terms in
// double-double), else igamLog1pDD(x) - x.
func igamLog1pmxDD(x dd.DD) dd.DD {
	if math.Abs(x.Hi) >= 0x1p-10 {
		return dd.Sub(igamLog1pDD(x), x)
	}
	x2 := dd.Mul(x, x)
	r := dd.Add(dd.MulF(x2, -0.5), igamDivF(dd.Mul(x2, x), 3))
	xh := x.Hi
	t := 0.0
	for k := 11.0; k >= 4; k-- {
		t = 1/k - xh*t
	}
	return dd.AddF(r, -xh*xh*xh*xh*t)
}
