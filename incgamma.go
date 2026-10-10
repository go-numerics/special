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
		p = igamSeriesP(a, x)
		return p, 1 - p
	case x < 1.1:
		q = igamSmallQ(a, x)
	default:
		q = igamFractionQ(a, x)
	}
	return 1 - q, q
}

// igamLog returns log(x) for x > 0, subnormal x included (math.Log is
// wrong for subnormal arguments on some platforms; dd.Log normalises).
func igamLog(x float64) float64 {
	return dd.Log(x).Hi
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
		d := dd.DivF(sumDD(x, -a), a) // x-a exact, then one DD division
		l := dd.MulF(log1pmxDD(d), a)
		l = dd.Add(l, dd.MulF(dd.Log(a), 0.5))
		l = dd.Sub(l, lnSqrt2Pi)
		return dd.AddF(l, -stirlingSeries(a))
	}
	l := dd.MulF(dd.Log(x), a)
	l = dd.AddF(l, -x)
	return dd.Sub(l, logGammaPos(a))
}

// igamSeriesP returns P(a, x) by the series
// P = x^a e^(-x)/Γ(a+1) Σ_{n>=0} x^n / ((a+1)(a+2)...(a+n)) (DLMF 8.7.1),
// whose terms are all positive.
func igamSeriesP(a, x float64) float64 {
	m, k := dd.ExpMant(igamLogPrefix(a, x))
	if m.Hi == 0 {
		return 0
	}
	s := igamLowerSeries(a, x)
	v := dd.DivF(dd.Mul(m, s), a)
	return math.Ldexp(v.Hi+v.Lo, k)
}

// igamLowerSeries returns S = Σ_{n>=0} x^n / ((a+1)...(a+n)) as a
// double-double (1 + igamLowerSeriesRest).
func igamLowerSeries(a, x float64) dd.DD {
	return dd.AddF(igamLowerSeriesRest(a, x), 1)
}

// igamLowerSeriesRest returns S - 1 = Σ_{n>=1} x^n / ((a+1)...(a+n)) as a
// double-double: the terms are formed in float64 (their relative errors
// grow slowly with n, and they decrease geometrically wherever this series
// is used) and summed with Neumaier's compensation.
func igamLowerSeriesRest(a, x float64) dd.DD {
	t := 1.0
	s := 0.0
	c := 0.0
	for n := 1.0; n < 10000; n++ {
		t *= x / (a + n)
		y := s + t
		c += (s - y) + t
		s = y
		if t <= 0x1p-60*s {
			break
		}
	}
	h, l := dd.FastTwoSum(s, c)
	return dd.DD{Hi: h, Lo: l}
}

// igamFractionQ returns Q(a, x) by Legendre's continued fraction
// Γ(a,x) = x^a e^(-x) / (x+1-a- 1(1-a)/(x+3-a- 2(2-a)/(x+5-a- ...)))
// (DLMF 8.9.2), evaluated by the modified Lentz method as Boost's
// upper_gamma_fraction does.
func igamFractionQ(a, x float64) float64 {
	m, k := dd.ExpMant(igamLogPrefix(a, x))
	if m.Hi == 0 {
		return 0
	}
	v := dd.DivF(m, igamFractionDen(a, x))
	return math.Ldexp(v.Hi+v.Lo, k)
}

// igamFractionDen returns the denominator x+1-a - 1(1-a)/(x+3-a - ...) of
// igamFractionQ, for x >= 1.1 and x - 1/(3x) >= a.
//
// The modified Lentz method (Boost's continued_fraction_a) only serves to
// find how many terms are needed: its running product accumulates one
// rounding per step, which near the transition x ~ a (50 and more steps)
// costs several ulp. The value is then summed backwards from that depth,
// which is stable for this fraction.
func igamFractionDen(a, x float64) float64 {
	z0 := dd.AddF(sumDD(x, -a), 1)
	const tiny = 0x1p-1000
	b := z0.Hi
	f := b
	c := f
	d := 0.0
	prev := 1.0
	n := 0.0
	for k := 1.0; k < 10000; k++ {
		n = k
		ak := k * (a - k)
		if ak == 0 {
			break // integer a: the fraction terminates exactly
		}
		b += 2
		d = b + ak*d
		if d == 0 {
			d = tiny
		}
		c = b + ak/c
		if c == 0 {
			c = tiny
		}
		d = 1 / d
		delta := c * d
		f *= delta
		// Near the transition x ~ a the fraction converges slowly and the
		// corrections shrink by a ratio rho close to 1: what remains is
		// about |delta-1|·rho/(1-rho), so the test scales with 1-rho.
		cur := math.Abs(delta - 1)
		rho := cur / prev
		prev = cur
		if cur <= 0x1p-56*math.Max(1-rho, 1.0/64) {
			break
		}
	}
	// backward: t = a_k/(b_k + t), k = N..1, with a margin beyond the
	// Lentz depth (the terms vanish exactly past an integer a)
	t := 0.0
	for k := math.Ceil(n*1.25) + 4; k >= 1; k-- {
		ak := k * (a - k)
		t = ak / (z0.Hi + 2*k + t)
	}
	return dd.AddF(z0, t).Float()
}

// igamSmallQ returns Q(a, x) for x < 1.1 and the a below DiDonato &
// Morris' changeover (a < 0.83), from
//
//	Γ(a, x) = (Γ(1+a) - x^a)/a - x^a Σ_{n>=1} (-x)^n / (n! (a+n))
//
// (DLMF 8.7.3; Boost's tgamma_small_upper_part), multiplied by a/Γ(1+a):
//
//	Q = [ (Γ(1+a)-1) - (x^a-1) - a x^a Σ ] / Γ(1+a).
//
// Γ(1+a)-1 and x^a-1 are double-doubles, so their difference does not
// cancel and nothing overflows for tiny a.
func igamSmallQ(a, x float64) float64 {
	return igamSmallQDD(a, x).Float()
}

// igamSmallQDD is igamSmallQ as a double-double (accurate to a few units
// of 2^-53 times the alternating sum's share of the result, which is
// small: the leading difference is exact).
func igamSmallQDD(a, x float64) dd.DD {
	gm1 := gamma1pm1(a)
	pm1 := igamExpm1DD(dd.MulF(dd.Log(x), a)) // x^a - 1
	// Σ_{n>=1} (-x)^n / (n! (a+n))
	s := 0.0
	t := 1.0
	for n := 1.0; n < 200; n++ {
		t *= -x / n
		term := t / (a + n)
		s += term
		if math.Abs(term) <= 0x1p-60*math.Abs(s) {
			break
		}
	}
	xa := dd.AddF(pm1, 1)
	num := dd.Sub(gm1, pm1)
	num = dd.Sub(num, dd.MulF(dd.MulF(xa, a), s))
	return dd.Div(num, dd.AddF(gm1, 1))
}

// igamExpm1DD returns exp(u) - 1 for a double-double u with |u| not large,
// as a double-double: expm1(u.Hi) + u.Lo·exp(u.Hi) with expm1(u.Hi) itself
// refined by one Newton step on log1p (relative error about 2^-100).
func igamExpm1DD(u dd.DD) dd.DD {
	if u.Hi < -40 {
		return dd.AddF(dd.F(-1), math.Exp(u.Hi))
	}
	e := math.Expm1(u.Hi)
	if e == 0 {
		return u
	}
	// Newton on f(e) = log1p(e) - u: e' = e - (log1p(e) - u)(1 + e).
	r := dd.Sub(log1pDD(dd.F(e)), u)
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
// asymptotic_series in igam.h; table igamTemmeCoef). The smaller of P and Q
// is computed directly. z² = -a·log1pmx((x-a)/a) is a double-double, and
// for |z| >= 1/2 the result is exp(-z²)·(erfcx(|z|)/2 ± R'), so the
// Gaussian factor is applied once from the exact exponent.
func igamTemme(a, x float64) (p, q float64) {
	d := dd.DivF(sumDD(x, -a), a)
	lpm := log1pmxDD(d) // <= 0
	z2 := dd.Neg(dd.MulF(lpm, a))
	eta := math.Sqrt(-2 * lpm.Hi)
	upper := x > a // compute Q directly; else P
	if !upper {
		eta = -eta
	}
	// Σ_k C_k(η) a^-k, stopping when the asymptotic terms stop decreasing.
	sum := 0.0
	afac := 1.0
	old := math.Inf(1)
	for k := range igamTemmeCoef {
		row := &igamTemmeCoef[k]
		ck := row[0]
		ep := 1.0
		for n := 1; n < len(row); n++ {
			ep *= eta
			term := row[n] * ep
			ck += term
			if math.Abs(term) < 0x1p-56*math.Abs(ck) {
				break
			}
		}
		term := ck * afac
		at := math.Abs(term)
		if at > old {
			break
		}
		sum += term
		if at < 0x1p-56*math.Abs(sum) {
			break
		}
		old = at
		afac /= a
	}
	// Q = erfc(z)/2 + R (η > 0); P = erfc(|z|)/2 - R (η < 0), where
	// R = exp(-z²) sum/sqrt(2πa) with sum computed at the signed η.
	r := sum / math.Sqrt(2*math.Pi*a)
	if !upper {
		r = -r
	}
	z := dd.Sqrt(z2)
	var v float64
	if z.Hi >= 0.5 {
		m, k := dd.ExpMant(dd.Neg(z2))
		base := dd.AddF(dd.MulF(erfcxCorrected(z), 0.5), r)
		w := dd.Mul(m, base)
		v = math.Ldexp(w.Hi+w.Lo, k)
	} else {
		e := dd.MulF(erfcDD(z), 0.5)
		v = dd.AddF(e, r*math.Exp(-z2.Hi)).Float()
	}
	if upper {
		return 1 - v, v
	}
	return v, 1 - v
}
