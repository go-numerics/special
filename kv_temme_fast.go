package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Path 4 of kv_fast.go: K_v e^x for 0 < v < 32 and kvTemmeMinX <= x < kvCF2MinX,
// the last region that was answered by besselIK (up to 2.64 ulp at v = 0.3,
// x = 0.3). With n the nearest integer to v and u = v - n, |u| <= 1/2 (exact), the
// pair K_u and K_(u+1) from Temme's series (kvTemmeSeries), then the hi + lo
// upward recurrence of the other paths (kvChainUp), run on the unscaled values:
// K_v itself is the result, K_v e^x one double-double product more.
//
// Method. Temme (1975, "On the numerical evaluation of the modified Bessel
// function of the third kind", J. Comput. Phys. 19, 324-337, not reopened; the
// recurrences below are those of Boost's temme_ik in bessel_ik.go and of
// besselk_temme_series in Bessels.jl, src/BesselFunctions/besselk.jl at
// 009b615 (MIT), read 2026-11-15, which also takes g1, g2 as Taylor
// polynomials in v^2 and sinh(mu)/mu as a series in mu^2, but only for |v| <
// 1e-5 and in float64, here for every |u| <= 1/2 in double-double. DLMF 10.31,
// opened 2026-11-15, gives only the integer-order series (10.31.1) and refers
// to 10.25.2 and 10.27.4 for the rest: the formulas are held by the math/big
// oracle of kv_oracle_test.go, not by it). With
// L = ln(2/x), sigma = u L, c_k = (x^2/4)^k / k! and E = e^sigma,
//
//	K_u = sum_k c_k f_k,    (x/2) K_(u+1) = sum_k c_k h_k,
//	f_0 = [cosh(sigma) g1 + (sinh(sigma)/sigma) L g2] / (rp rm),
//	p_0 = E / (2 rp),   q_0 = 1 / (2 E rm),   h_0 = p_0,
//	f_k = (k f_(k-1) + p_(k-1) + q_(k-1)) / (k^2 - u^2),
//	p_k = p_(k-1) / (k - u),   q_k = q_(k-1) / (k + u),   h_k = p_k - k f_k,
//
// where rp = 1/Gamma(1+u), rm = 1/Gamma(1-u), g2 = (rp + rm)/2 and g1 = (rm -
// rp)/(2u) (Temme's Gamma_2 and Gamma_1), and rp rm = sin(pi u)/(pi u) (the
// reflection formula, so there is no sine to evaluate).
//
// The limits u -> 0. Nothing divides by u. g1 and g2 are even functions of u
// given by Taylor series in w = u^2 (zz_kvTemme_coeffs.go, from 1/Gamma(1+u) =
// sum B_m u^m: g2 = sum B_(2j) w^j, g1 = -sum B_(2j+1) w^j; the generator,
// testdata/gen/kvtemme, shares nothing with the package), so g1(0) = -gamma and
// g2(0) = 1 are the constant terms and "1/Gamma(1-u) - 1/Gamma(1+u)" is never
// formed. rp = g2 - u g1 and rm = g2 + u g1 add numbers of one sign (g2 >= 0.84,
// |u g1| <= 0.29). sinh(sigma)/sigma and cosh(sigma) are Taylor series in sigma^2
// for |sigma| < 2^-6 (the terms to sigma^8: below 2^-82), where (E - 1/E)/(2
// sigma) would lose 6 bits of E's 2^-76; sigma = 0 gives exactly 1 and 1. The
// order is not restricted to u != 0: u = 0 (integer v) is K_0 = L - gamma + ...
// and K_1 from the same code.
//
// Signs, for |u| <= 1/2, 0 < x < 1.5 (TestKvTemmeSigns also checks them on a
// grid): p_k, q_k > 0 for every k (k - |u| >= 1/2), and f_k > 0 for k >= 1.
// Proof (independent review of #83): with 1/Gamma(1 -/+ u) = g2 +/- u g1 and
// Gamma(1+u) Gamma(1-u) = pi u / sin(pi u), p_0 - q_0 = u f_0 exactly, and the
// recurrences p_k = p_(k-1)/(k-u), q_k = q_(k-1)/(k+u), f_k = (k f_(k-1) +
// p_(k-1) + q_(k-1))/(k^2 - u^2) keep f_k = (p_k - q_k)/u for every k. For u > 0,
// p_k/q_k = (2/x)^(2u) Gamma(k+1+u)/Gamma(k+1-u): (2/x)^(2u) > 1 since x < 2,
// and for k >= 1, k+1-u >= 3/2 > 1.4616 (the minimum of Gamma), where Gamma
// increases, so p_k > q_k and f_k > 0. f is even in u, and u = 0 is the limit
// (L + psi(k+1))/k! > 0 with L = ln(2/x) > 0. f_0 is the only term that can be
// negative: for u near 0 it is L - gamma, below zero for x above 2 e^-gamma =
// 1.12. So every term c_k f_k of K_u is positive for k >= 1,
// and the partial sums of K_u rise (h_k = p_k - k f_k has no sign; the series of
// (x/2) K_(u+1) loses a factor 8 at worst, sum c_k |h_k| / |sum c_k h_k| at u =
// -1/2, x = 1.5: TestKvTemmeCancellation, 3 bits of the double-double's 100 and
// counted in the weights below).
//
// Truncation, rigorous for the exact recurrences. For j > k >= 1, P_j = max(p_j,
// q_j) <= P_(j-1)/(j - 1/2), and f_j <= (j f_(j-1) + 2 P_(j-1))/(j^2 - 1/4)
// (p_(j-1) + q_(j-1) <= 2 P_(j-1)). With z_j = f_j + 4 P_j:
//
//	z_j <= z_(j-1) / k           (j > k)
//
// because j/(j^2 - 1/4) <= 1/k for j >= k + 1 (j (j - k) >= j > 1/4), and
// 2/(j^2 - 1/4) + 4/(j - 1/2) <= 4/k: the left side is at most 2/((k + 1/2)(k +
// 3/2)) + 4/(k + 1/2) (j >= k + 1), and 4/k - 4/(k + 1/2) = 2/(k (k + 1/2)) is
// at least the first term because 2/(k + 3/2) <= 2/k. So f_j, P_j <= z_k
// k^-(j-k) for j > k, and with q = x^2/4, c_(j+1)/c_j = q/(j+1):
//
//	sum_(j>k) c_j f_j     <= z_k c_(k+1) / (k (1 - rho)),     rho = q / (k (k + 1)),
//	sum_(j>k) c_j |h_j|   <= z_k c_(k+1) (k + 5/4) / (k (1 - rho)),
//
// (|h_j| <= p_j + j f_j; j c_j = q c_(j-1) turns the second sum into a geometric
// one). kvTemmeTail evaluates both from the float64 state after step k; the loop
// stops when the first is below 2^-64 of the partial sum of K_u and the second
// below 2^-64 of the partial sum of (x/2) K_(u+1) (a partial sum of K_u is a lower
// bound: its terms are positive after the first). The bound holds for the exact
// numbers; the computed state differs from them by a few 2^-53, which moves the
// bound by the same relative amount: nothing against the 4x margin between the
// stopping tolerance 2^-64 and the 2^-62 used below. TestKvTemmeTruncation
// compares the stopped pair with the loop run to 2^-90.
//
// What is NOT proven: the Taylor series of g1 and g2
// (truncated after 14 terms each, below 2^-89 at |u| = 1/2 by the computed
// coefficients, which decay faster than 2^-m beyond m = 12; the decay of B_m for
// m > 63 is not proved here; TestKvTemmeGammaSeries holds the sum against ln
// Gamma at 320 bits on a grid of u to 2^-100), and the first kvTemmeTerms terms'
// weights below (a grid).
//
// Precision. The first kvTemmeTerms(x) steps run in double-double (0 to 4: the
// sloppy additions of bessel_ik.go, 2^-104 each); the rest in float64. The terms
// after the double-double ones weigh at most kvTemmeTailWeight of the sums
// (TestKvTemmeTermWeights, on a grid of u and x), so their float64 rounding is
// below 2^-62. f_0, p_0 and q_0 are in double-double throughout: the cancellation
// of f_0 (cosh(sigma) g1 < 0 against L g2 > 0 near x = 1.12) is of the absolute
// size of its terms, 2^-100, against K >= K_0(1.5) = 0.21. The logarithm (dd.Log,
// 2^-72 relative) moves sigma by 2^-72 |sigma| <= 2^-63.5 (|sigma| <= 347 at x =
// 2^-1000); the exponentials (dd.ExpMant, 2^-76) and the Gamma polynomials
// (float64 beyond the 4th term, 2^-69) are smaller. Together the start pair is
// within 2^-60 relative of the exact (measured on 1.2e5 points,
// TestKvTemmePairOracle: K_u 2^-64.7, K_(u+1) 2^-62.0; Kv and Kve 0.500 ulp).
//
// Total: 0.5 ulp for the rounding, the start pair 2^-60 (0.008 ulp), the
// recurrence R_n <= R_0 + n 2^-91 (kn_fast.go's derivation, n <= 32), and 2^-76 for
// e^x of the scaled value: below 0.52 ulp.
//
// Range. x >= kvTemmeMinX = 2^-1000 so that 2/x and 2(u+k)/x (below 2^1006) are
// finite. K_w(x) increases with w >= 0 and the chain's values increase to the
// result (kvChainUp), so where the result is below kvTemmeBig = 2^1020 no
// intermediate overflows. Where the result, or K_(u+1) (needed for n >= 1), is not
// below kvTemmeBig, or any value is not finite (NaN fails the test too), the
// path declines and besselIK answers as before, bit for bit: it returns +Inf
// where the true value overflows. The band 2^1020 to MaxFloat64 is therefore
// never decided here, and the path never returns +Inf itself.
// There is no underflow: K_v(x) >= K_0(1.5) = 0.21 for v >= 0 and x < 1.5.
const (
	// kvTemmeMinX is the smallest x answered.
	kvTemmeMinX = 0x1p-1000
	// kvTemmeStop is the tolerance of the truncation bounds.
	kvTemmeStop = 0x1p-64
	// kvTemmeMaxIters caps the float64 loop (kvInv has more entries).
	kvTemmeMaxIters = 60
	// kvTemmeSmallSigma is where cosh and sinh(sigma)/sigma are Taylor series.
	kvTemmeSmallSigma = 0x1p-6
	// kvTemmeBig bounds the unscaled result the chain answers: below it the
	// scaled result (times e^x < 4.49 < 2^2.2) is finite too, and no
	// intermediate of kvChainUp (all below the result) is near overflow.
	kvTemmeBig = 0x1p1020
)

// kvTemmeTerms is the number of terms run in double-double at x.
func kvTemmeTerms(x float64) int {
	switch {
	case x < 0x1p-10:
		return 0
	case x < 0.125:
		return 1
	case x < 0.5:
		return 2
	case x < 1:
		return 3
	}
	return 4
}

// kvTemmeGamma returns g1(w) and g2(w) as double-doubles for w = u^2 <= 1/4:
// the Taylor series of zz_kvTemme_coeffs.go, the terms from kvTemmeGammaDD on
// in float64 (at most 2^-17 of the sum), the first ones in double-double.
func kvTemmeGamma(w dd.DD) (g1, g2 dd.DD) {
	n := kvTemmeGammaTerms
	t1, t2 := kvTemmeG1[n-1][0], kvTemmeG2[n-1][0]
	for j := n - 2; j >= kvTemmeGammaDD; j-- {
		t1 = math.FMA(t1, w.Hi, kvTemmeG1[j][0])
		t2 = math.FMA(t2, w.Hi, kvTemmeG2[j][0])
	}
	g1, g2 = dd.F(t1), dd.F(t2)
	for j := kvTemmeGammaDD - 1; j >= 0; j-- {
		g1 = besselAddS(dd.Mul(g1, w), dd.DD{Hi: kvTemmeG1[j][0], Lo: kvTemmeG1[j][1]})
		g2 = besselAddS(dd.Mul(g2, w), dd.DD{Hi: kvTemmeG2[j][0], Lo: kvTemmeG2[j][1]})
	}
	return g1, g2
}

// kvTemmeHyp returns cosh(sigma) and sinh(sigma)/sigma for sigma = sig, with
// E = e^sigma and Ei = 1/E: the Taylor series below kvTemmeSmallSigma (the
// terms past the first held in float64: at most 2^-13 of the value, 2^-66), the
// quotients of E above it.
func kvTemmeHyp(sig, E, Ei dd.DD) (ch, shc dd.DD) {
	if math.Abs(sig.Hi) >= kvTemmeSmallSigma {
		ch = dd.MulF(dd.Add(E, Ei), 0.5)
		shc = dd.Div(dd.Sub(E, Ei), dd.MulF(sig, 2))
		return ch, shc
	}
	s2, s2l := dd.TwoProd(sig.Hi, sig.Hi)
	s2l += float64(2 * float64(sig.Hi*sig.Lo))
	a, e := dd.TwoSum(1, float64(0.5*s2))
	r := float64(s2 * s2 * (1.0/24 + float64(s2*(1.0/720+float64(s2*(1.0/40320))))))
	ch = dd.DD{Hi: a, Lo: e}
	ch = besselAddFS(ch, float64(0.5*s2l)+r)
	a, e = dd.TwoSum(1, float64(s2*(1.0/6)))
	r = float64(s2 * s2 * (1.0/120 + float64(s2*(1.0/5040+float64(s2*(1.0/362880))))))
	shc = dd.DD{Hi: a, Lo: e}
	shc = besselAddFS(shc, float64(s2l*(1.0/6))+r)
	return ch, shc
}

// kvTemmeTail returns the bounds of the header on sum_(j>k) c_j f_j and on
// sum_(j>k) c_j |h_j|, from the float64 state after step k >= 1 (qh = x^2/4,
// cf = c_k).
func kvTemmeTail(k int, qh, cf, ff, pf, qf float64) (bf, bh float64) {
	z := ff + float64(4*math.Max(pf, qf))
	c1 := float64(float64(cf*qh) * kvInv[k+1])
	t := float64(float64(z*c1) * kvInv[k])
	rho := float64(float64(qh*kvInv[k]) * kvInv[k+1])
	den := 1 - rho
	if !(den > 0) {
		// outside the region (x^2/4 >= k (k + 1)): no bound
		return math.Inf(1), math.Inf(1)
	}
	return t / den, float64(t*(float64(k)+1.25)) / den
}

// kvTemmeSeries returns K_u(x) and, if needK1, K_(u+1)(x) (unscaled) as
// double-doubles for |u| <= 1/2 and kvTemmeMinX <= x < kvCF2MinX, ok false if
// the iteration cap is reached.
func kvTemmeSeries(u, x float64, needK1 bool) (k0, k1 dd.DD, ok bool) {
	return kvTemmeSeriesTol(u, x, needK1, kvTemmeStop)
}

// kvTemmeSeriesTol is kvTemmeSeries with the truncation tolerance as a
// parameter (the tests run it far below the production one).
func kvTemmeSeriesTol(u, x float64, needK1 bool, stop float64) (k0, k1 dd.DD, ok bool) {
	L := dd.Neg(dd.Log(float64(0.5 * x)))
	sig := dd.MulF(L, u)
	w := dd.MulFF(u, u)
	g1, g2 := kvTemmeGamma(w)
	ug := dd.MulF(g1, u)
	rp, rm := dd.Sub(g2, ug), dd.Add(g2, ug)
	mE, kE := dd.ExpMant(sig)
	E := dd.Ldexp(mE, kE)
	Ei := dd.Div(dd.F(1), E)
	ch, shc := kvTemmeHyp(sig, E, Ei)
	t := dd.Add(dd.Mul(ch, g1), dd.Mul(dd.Mul(shc, L), g2))
	f := dd.Mul(t, besselRecipDD(dd.Mul(rp, rm)))
	p := dd.MulF(dd.Mul(E, besselRecipDD(rp)), 0.5)
	q := dd.MulF(dd.Mul(Ei, besselRecipDD(rm)), 0.5)
	Sf, Sh := f, p
	c := dd.F(1)
	x2h, x2l := dd.TwoProd(x, x)
	qd := dd.DD{Hi: float64(0.25 * x2h), Lo: float64(0.25 * x2l)}
	m := kvTemmeTerms(x)
	k := 1
	for ; k <= m; k++ {
		fk := float64(k)
		den := besselAddFS(dd.Neg(w), float64(fk*fk))
		fn := dd.Mul(besselAddS(besselAddS(dd.MulF(f, fk), p), q), besselRecipDD(den))
		pk := dd.Mul(p, besselRecipDD(kvTemmePair(fk, -u)))
		qk := dd.Mul(q, besselRecipDD(kvTemmePair(fk, u)))
		f, p, q = fn, pk, qk
		h := dd.Sub(p, dd.MulF(f, fk))
		c = igamDivF(dd.Mul(c, qd), fk)
		Sf = besselAddS(Sf, dd.Mul(c, f))
		if needK1 {
			Sh = besselAddS(Sh, dd.Mul(c, h))
		}
	}
	// the rest in float64
	qh := qd.Hi
	ff, pf, qf, cf := f.Hi, p.Hi, q.Hi, c.Hi
	var tf, th float64
	converged := false
	if m >= 1 {
		bf, bh := kvTemmeTail(m, qh, cf, ff, pf, qf)
		converged = kvTemmeDone(bf, bh, Sf.Hi, Sh.Hi, needK1, stop)
	}
	for ; !converged && k < kvTemmeMaxIters; k++ {
		fk := float64(k)
		d := math.FMA(-u, u, float64(fk*fk))
		ff = (float64(fk*ff) + pf + qf) / d
		pf /= fk - u
		qf /= fk + u
		hf := pf - float64(fk*ff)
		cf = float64(float64(cf*qh) * kvInv[k])
		tf += float64(cf * ff)
		th += float64(cf * hf)
		bf, bh := kvTemmeTail(k, qh, cf, ff, pf, qf)
		converged = kvTemmeDone(bf, bh, Sf.Hi+tf, Sh.Hi+th, needK1, stop)
	}
	if !converged {
		return dd.DD{}, dd.DD{}, false
	}
	k0 = dd.AddF(Sf, tf)
	if needK1 {
		k1 = dd.DivF(dd.AddF(Sh, th), float64(0.5*x))
	}
	return k0, k1, true
}

// kvTemmePair returns a + b as a double-double (the exact sum).
func kvTemmePair(a, b float64) dd.DD {
	s, e := dd.TwoSum(a, b)
	return dd.DD{Hi: s, Lo: e}
}

// kvTemmeDone reports whether the tail bounds bf (of K_u) and bh (of (x/2)
// K_(u+1), if needK1) are at most stop times the partial sums sf and sh: both
// partial sums must be positive (sf is a lower bound of K_u; sh is within bh
// of the value).
func kvTemmeDone(bf, bh, sf, sh float64, needK1 bool, stop float64) bool {
	if !(sf > 0 && bf <= float64(stop*sf)) {
		return false
	}
	return !needK1 || (sh > 0 && bh <= float64(stop*sh))
}

// kvTemmeChain is Path 4: v in (0, 32), kvTemmeMinX <= x < kvCF2MinX. With n the
// nearest integer to v and u = v - n, the pair K_u, K_(u+1) by Temme's series,
// then kvChainUp on the unscaled values; ok false where the result is not below
// kvTemmeBig (or an intermediate is not finite).
func kvTemmeChain(v, x float64, scaled bool) (float64, bool) {
	n, u := kvReduce(v)
	k0, k1, ok := kvTemmeSeries(u, x, n >= 1)
	if !ok {
		return 0, false
	}
	h, l := kvChainUp(u, int(n), x, k0.Hi, k0.Lo, k1.Hi, k1.Lo)
	if !(h < kvTemmeBig) {
		return 0, false
	}
	if !scaled {
		return h + l, true
	}
	m, e := dd.ExpMant(dd.F(x))
	r := dd.Mul(dd.DD{Hi: h, Lo: l}, m)
	return math.Ldexp(r.Hi+r.Lo, e), true
}
