package special

import (
	"math"

	"github.com/go-numerics/special/dd"
	"github.com/go-numerics/special/internal/libm"
)

// Iv, In and Ive fast paths for non-negative order. The scheme is that of
// Bessels.jl's besseli (JuliaMath/Bessels.jl, MIT, src/BesselFunctions/
// besseli.jl, read 2026-11-14 at master; LICENSE opened): the large-argument
// expansion where it converges (DLMF 10.40.1), Debye's uniform expansion for
// large orders (DLMF 10.41.3), a power series otherwise. Its accuracy is not
// taken (Bessels.jl tolerates about 2e-11 relative error): every answer is
// built from pairs hi + lo that are good to 2^-58 or better and rounded once,
// and where no path applies the answer is besselIK as before, bit for bit.
//
// Everything is carried as I_v(x) e^-x (the scaled function), which neither
// overflows nor underflows on its own; Iv multiplies by e^x from dd.ExpMant at
// the end (ivFinish).
//
// Path 1, v in [32, 1100], x < 1e6 (ivDebye). With S = sqrt(v^2 + x^2),
// p = v/S and v eta = S + v ln(x/(v+S)) (DLMF 10.41.7),
//
//	I_v(x) = e^(v eta) / sqrt(2 pi S) (1 + sum_{k>=1} U_k(p)/v^k)
//
// (10.41.3, with (1+z^2)^(-1/4) (2 pi v)^(-1/2) = (2 pi S)^(-1/2)). The
// exponent, the prefactor and the stopping rule are those of kvDebye
// (kv_fast.go), which carries their analysis: S in double-double, the
// exponent within v 3 2^-72 |ln x| absolute (below 2^-56), the sum of the
// terms k >= 1 bounded by 1/(12 v) so that its float64 rounding is below
// 2^-57, and the loop stops on max |U_{k+1}|/v^(k+1) < 2^-64 over p in [0, 1]
// (kvDebyeMaxU), never on a computed term: U_k(p) has roots in (0, 1) and a
// term can vanish while the next is large (the defect found in the review of
// #78). Declined when 16 terms do not reach it. NOT proven: that the terms
// beyond the first neglected one are smaller still (Olver 1997 pp. 374-378,
// cited by DLMF 10.41(iii) for the error bounds, was not opened); what stands
// in place of it is the same measurement as for K (kvDebyeMaxU) and the
// oracle comparison (TestIvFastOracle).
//
// Path 2, x >= ivAsymMinX (ivAsym): the large-argument expansion of the order
// v itself,
//
//	I_v(x) e^-x = 1/sqrt(2 pi x) sum_k (-1)^k a_k(v)/x^k
//
// (DLMF 10.40.1), as a pair, ok only if (a) the terms fall below 2^-66 before
// they grow, (b) the float64 rounding of the terms from the second on, bounded
// term by term, stays below 2^-57 of the sum (ivAsymPair keeps the running
// bound: that is the per-answer certificate, a high order or a small x
// declines, and so does a sum that cancels). The companion term of
// DLMF 10.40.5, e^-2x relative, is below 2^-69 for x >= 24 and is neglected.
// NOT proven: that the remainder of the truncated series is below the first
// neglected term (DLMF 10.40(ii), opened 2026-11-14, gives that bound for K
// and says only that a similar bound for I follows from 10.34.3 and
// 10.40.10-12, citing Olver 1997 p. 269, which was not opened); the oracle
// comparison holds it.
//
// Path 3, v = u + n with 1 <= n < 32, u in [0, 1) and x of the base
// (ivMiller): the downward recurrence
//
//	r_{k-1} = r_{k+1} + (2 (u+k)/x) r_k,   r_{N+1} = 0, r_N = 1
//
// (DLMF 10.29.1: I_{w-1} - I_{w+1} = (2w/x) I_w; the same law gives K_{w+1} =
// K_{w-1} + (2w/x) K_w), normalised by I_u e^-x: I_{u+n} e^-x = I_u e^-x
// r_n/r_0. All of r is positive, so no step cancels, and the step adds two
// positive terms: the errors of the steps are those of the "dd-light" chain of
// kn_fast.go. The truncation is certified a priori. The computed r is r_k =
// a (I_k K_{N+1} - K_k I_{N+1}) (order u+k; the two solutions of the
// recurrence that vanish at N+1), so r_k = a I_k (1 - e_k) with
//
//	e_k = K_k I_{N+1} / (I_k K_{N+1}) = prod_{j=k}^{N} (I_{j+1}/I_j) (K_j/K_{j+1}),
//
// and by DLMF 10.37 (opened 2026-11-14: I_nu(x) decreasing and K_nu(x)
// increasing in the order nu >= 0 at fixed x > 0) every factor is at most 1;
// by the power series DLMF 10.25.2, termwise, I_{w+1}/I_w <= x/(2(w+1)), and
// by DLMF 10.29.1 K_{w+1} >= (2w/x) K_w, so K_w/K_{w+1} <= x/(2w). With
// g_j = min(1, x/(2j)) (j >= 1, a bound over the order u + j >= j):
//
//	e_n <= P(n, N) = prod_{j=n}^{N} g_j g_{j+1},
//
// e_0 <= e_n, and the ratio r_n/r_0 = (I_n/I_0)(1-e_n)/(1-e_0) is within
// e_n/(1-e_n) of the truth. ivStart finds the least N with P <= 2^-63. The
// levels above J (the first level where P <= 2^-24) run in plain float64: a
// relative rounding d there moves the answer by at most d P(n, J), and there
// are at most N - J of them: (N - J) 4 2^-53 2^-24 < 2^-62 for N - J < 512.
// The base I_u e^-x is the pair of the I0e octave table (u = 0, x in [1/8,
// 512): 2^-58, as the K tables of kn_fast.go), or the large-argument pair of
// Path 2 for x >= ivAsymMinX when u > 0. Where the base is not available the
// path declines.
//
// Total: 0.5 ulp rounding + 2^-58 (the base, the Debye sums, the
// asymptotic pair) + 2^-62 (the certificates) + 2^-76 for the exponential:
// below 0.56 ulp for Paths 1 and 3, 0.6 for Path 2 (the 2^-57 allowed to its
// float64 terms); TestIvFastFixture bounds every row at ivFastBound.
const (
	// ivAsymMinX is where the companion term (e^-2x) falls below 2^-69 and
	// the series of an order below 2 has terms that reach 2^-66.
	ivAsymMinX = 24
	// ivSeriesMaxTerms caps the series, which the range guard ends long before
	// (130 terms where it applies); beyond it the terms are Inf and then NaN,
	// the pair is NaN and ivFinish refuses it.
	ivSeriesMaxTerms = 400
	// ivMillerMaxLevel bounds the start level N of the recurrence (the longest
	// chains are at x near 512: about 160 levels).
	ivMillerMaxLevel = 600
	// ivMaxScaledX: ivAsym keeps 8x and c/x normal.
	ivMaxScaledX = 0x1p1000
	// ivMaxUnscaledX is where the unscaled functions are left to besselIK:
	// e^x passes MaxFloat64 at 709.78 and the pair has to be multiplied by it.
	ivMaxUnscaledX = 700

	ivFastBound = 0.6
)

// ivFast returns I_v(x) (scaled: I_v(x) e^-x) for v >= 0 and finite x > 0,
// with ok false where no fast path applies.
func ivFast(v, x float64, scaled bool) (r float64, ok bool) {
	if v >= kvDebyeMinV {
		if v <= kvDebyeMaxV && x < kvDebyeMaxX {
			return ivDebye(v, x, scaled)
		}
		return 0, false
	}
	if !scaled && x > ivMaxUnscaledX {
		return 0, false
	}
	if h, l, ok := ivAsymPair(v, x); ok {
		return ivFinish(h, l, x, scaled)
	}
	return ivMiller(v, x, scaled)
}

// ivFinish rounds the pair h + l = I e^-x, multiplied by e^x unless scaled
// (the pair is not below 1e-100: no result is subnormal). e^x = m 2^k from
// dd.ExpMant; m is in about [0.99, 2.02], so m.Hi h could pass MaxFloat64
// although the result does not: h and l are scaled by 1/4 (exact) and k raised
// by 2, as knRecur.
func ivFinish(h, l, x float64, scaled bool) (float64, bool) {
	// A pair below 2^-900 is refused (lo would be subnormal: the result is not
	// rounded once), and so are the zero and NaN pairs that a base, a start level
	// or a chain that declined leaves in ivMiller.
	if !(h >= 0x1p-900) {
		return 0, false
	}
	if scaled {
		return h + l, true
	}
	m, k := dd.ExpMant(dd.DD{Hi: x})
	h, l = float64(h*0.25), float64(l*0.25)
	p, pe := dd.TwoProd(m.Hi, h)
	return math.Ldexp(p+(pe+float64(m.Hi*l)+float64(m.Lo*h)), k+2), true
}

// ivDebye is Path 1 above.
func ivDebye(v, x float64, scaled bool) (float64, bool) {
	S, e := kvDebyeExponent(v, x, scaled)
	e = dd.Neg(e)
	// kvDebyeExponent returns -v eta (+ x if scaled): for I the exponent is
	// v eta (- x if scaled).
	switch kvExpClass(e.Hi) {
	case 1:
		return math.Inf(1), true
	case -1:
		return 0, true
	case 2:
		return 0, false
	}
	m, k := dd.ExpMant(e)
	c := dd.Sqrt(dd.Div(besselInv2PiDD, S))
	s, _, done := kvDebyeTail(v, S.Hi, 1)
	if !done {
		return 0, false
	}
	uh, ul := dd.TwoSum(1, s)
	r := dd.Mul(c, m)
	p := dd.Add(dd.MulF(r, uh), dd.F(float64(r.Hi*ul)))
	// Inside the window the result is normal and finite: the product is above
	// e^-700 c (1 - 1%) with c = 1/sqrt(2 pi S) >= 3.9e-4 (S < 1e6 + 1100),
	// 3.8e-308, and below e^700 c 1.01 with c <= 1/sqrt(2 pi 32) = 0.071,
	// 7.3e302.
	return math.Ldexp(p.Hi+p.Lo, k), true
}

// ivAsymPair returns I_w(x) e^-x for w >= 0 and x >= ivAsymMinX as hi + lo by
// DLMF 10.40.1, ok false if (a), (b) or (c) of Path 2 fail. The first term is
// a double-double (w^2 up to 1000 there), the others float64 with the running
// bound of their rounding: term k carries at most (3k + 3) 2^-53 relative
// (k - 1 products and quotients from the first term, the factor and its
// division).
func ivAsymPair(w, x float64) (h, l float64, ok bool) {
	if !(x >= ivAsymMinX && x < ivMaxScaledX) {
		return 0, 0, false
	}
	return ivAsymSum(w, x)
}

// ivAsymSum is ivAsymPair without the range guard (the cap of its loop is held
// on its own: TestIvAsymRange).
func ivAsymSum(w, x float64) (h, l float64, ok bool) {
	m2 := dd.MulFF(w, w)
	mu := dd.DD{Hi: float64(4 * m2.Hi), Lo: float64(4 * m2.Lo)}
	t1 := dd.Neg(dd.DivF(dd.AddF(mu, -1), float64(8*x)))
	sh, sl := t1.Hi, t1.Lo
	t := t1.Hi
	inv := 1 / float64(8*x)
	prev := math.Abs(t)
	done := math.Abs(t) < 0x1p-66
	bound := 0.0
	for k := 2; !done; k++ {
		o := float64(2*k - 1)
		f := (mu.Hi - float64(o*o)) + mu.Lo
		t = float64(t * float64(-f*inv/float64(k)))
		at := math.Abs(t)
		if !(at <= prev) { // also NaN: a loop that no comparison ends
			return 0, 0, false
		}
		prev = at
		bound += float64(at * float64(0x1p-53*float64(3*k+3)))
		var e float64
		sh, e = dd.TwoSum(sh, t)
		sl += e
		done = at < 0x1p-66
	}
	uh, ue := dd.TwoSum(1, sh)
	// the certificate: the rounding of the terms is at most 2^-57 of the sum
	if !(bound <= float64(0x1p-57*math.Abs(uh))) {
		return 0, 0, false
	}
	c := dd.Sqrt(dd.DivF(besselInv2PiDD, x))
	r := dd.Mul(c, dd.DD{Hi: uh, Lo: ue + sl})
	return r.Hi, r.Lo, true
}

// ivStart returns the start level N and the float64 limit J of the downward
// recurrence for order u + n (n >= 1) at x, from the bound P(n, N) of Path 3:
// N is the least level with P <= 2^-63, J the first with P <= 2^-24 (n <= J <=
// N), ok false if N would pass ivMillerMaxLevel.
func ivStart(n int, x float64) (N, J int, ok bool) {
	hx := float64(x / 2)
	j0 := int(math.Ceil(hx))
	if j0 < n {
		j0 = n
	}
	if j0 > ivMillerMaxLevel {
		return 0, 0, false
	}
	L := float64(2 * libm.Log(hx))
	N = ivSolve(j0, hx, L, ivLnTargetN)
	if N < 0 {
		return 0, 0, false
	}
	J = ivSolve(j0, hx, L, ivLnTargetJ)
	return N, J, true
}

// ivLnTargetN and ivLnTargetJ are ln 2^-63 and ln 2^-24, less 1e-6 (the
// logarithms of the factorial table are good to about 1e-10).
const (
	ivLnTargetN = -63*math.Ln2 - 1e-6
	ivLnTargetJ = -24*math.Ln2 - 1e-6
)

// ivLnFact[j] = ln j!, j = 0 .. ivMillerMaxLevel+1, summed with libm.Log (the
// logarithm the architecture hash pins), so the levels do not move between
// architectures.
var ivLnFact = func() (t [ivMillerMaxLevel + 2]float64) {
	for j := range t[:len(t)-1] {
		t[j+1] = t[j] + libm.Log(float64(j+1))
	}
	return
}()

// ivSolve returns a level N >= j0 with lnP(N) <= target (-1 if none within
// ivMillerMaxLevel), where, for the levels from j0 >= x/2 on (every g_j =
// x/(2j) there, L = 2 ln(x/2)),
//
//	lnP(N) = (N-j0+1) L - (ln N! - ln (j0-1)!) - (ln (N+1)! - ln j0!)
//
// is the logarithm of the product of g_j g_{j+1} over j = j0 .. N: the factors
// below j0 are at most 1 and are left out, which only raises the bound. lnP is
// concave and decreasing for N >= j0 (its increment is L - ln N - ln (N+1) <
// 0), so a Newton step from a level with lnP above the target lands at or
// beyond the first level below it. Not the least N: any N below the target is
// a valid start, the rule of the bound.
func ivSolve(j0 int, hx, L, target float64) int {
	lf := &ivLnFact
	// first probe: lnP(N) is about -(N - x/2)^2/(x/2) (ln(N/(x/2)) <= N/(x/2) - 1
	// makes the real root later, never earlier), taken at 0.8 of its root:
	// where the slope is large, so that the Newton steps below are short and
	// land just beyond the root
	N := j0 + int(0.8*math.Sqrt(-target*hx))
	if N > ivMillerMaxLevel {
		N = ivMillerMaxLevel
	}
	for N <= ivMillerMaxLevel {
		f := float64(float64(N-j0+1)*L) - (lf[N] - lf[j0-1]) - (lf[N+1] - lf[j0])
		if f <= target {
			return N
		}
		slope := L - (lf[N] - lf[N-1]) - (lf[N+1] - lf[N])
		step := int(math.Ceil((target - f) / slope))
		// slope < 0 and f > target: step >= 1
		N += step
	}
	return -1
}

// ivBase returns I_u(x) e^-x as a pair for u in [0, 1): the I0e octave table
// for u = 0 (x in [1/8, 512): outside, ivBase declines), the large-argument pair for x >= ivAsymMinX.
func ivBase(u, x float64) (h, l float64, ok bool) {
	if u == 0 {
		i := octIndex(x, besselI0eOctMin, besselI0eOctMax)
		if i < 0 {
			return 0, 0, false
		}
		q := &besselI0eOct[i]
		h, l = dd.TwoSum(q.coef[11], octTail(q, x-q.c))
		return h, l, true
	}
	if x >= ivAsymMinX {
		return ivAsymPair(u, x)
	}
	h, l = ivSeriesPair(u, x)
	return h, l, true
}

// ivSeriesPair returns I_u(x) e^-x for u in (0, 1) and 0 < x < ivAsymMinX by
// the power series DLMF 10.25.2,
//
//	I_u(x) e^-x = e^(u ln(x/2) - x) / Gamma(1+u) sum_k t_k,  t_k = t_{k-1} q/(k (u+k)),
//
// q = x^2/4, as a pair. The prefactor exponent is a double-double (dd.Log: 2^-72
// relative, u |ln(x/2)| and x below 25: 2^-67 absolute, which is its relative
// error in the exponential), Gamma(1+u) the double-double of gamma1p. All
// terms are positive: no step cancels, the product of the exact-to-2^-104
// factors t_k = prod q/(j (u+j)) and the sum carry 2^-100 per term, k <= 120
// terms: 2^-93. The loop stops at a term below 2^-66 of the sum once the
// ratio of the next term, q/((k+1)(u+k+1)), is at most 1/2: that ratio falls
// with k, so the neglected tail is at most twice the last term, 2^-65 of the
// sum, and the terms are positive, so a computed term cannot vanish: this is a
// bound over the parameters, not a luck of the roots.
func ivSeriesPair(u, x float64) (h, l float64) {
	lx := dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo})
	m, e := dd.ExpMant(dd.AddF(dd.MulF(lx, u), -x))
	pre := dd.Div(m, gamma1p(u))
	qh, ql := dd.TwoProd(x, x)
	q := dd.DD{Hi: float64(qh * 0.25), Lo: float64(ql * 0.25)}
	t, s := dd.F(1), dd.F(1)
	for k := 1; k <= ivSeriesMaxTerms; k++ {
		fk := float64(k)
		uh, ul := dd.TwoSum(u, fk)
		dh, de := dd.TwoProd(fk, uh)
		a := dd.Mul(q, besselRecipDD(dd.DD{Hi: dh, Lo: de + float64(fk*ul)}))
		t = dd.Mul(t, a)
		s = besselAddS(s, t)
		next := float64(q.Hi / float64(float64(fk+1)*float64(u+fk+1)))
		if next <= 0.5 && t.Hi <= float64(0x1p-66*s.Hi) {
			break
		}
	}
	r := dd.Mul(pre, s)
	return math.Ldexp(r.Hi, e), math.Ldexp(r.Lo, e)
}

// ivMiller is Path 3 above for v >= 1 (with n = floor(v) >= 1); v < 1 is the
// base alone.
func ivMiller(v, x float64, scaled bool) (float64, bool) {
	nf := math.Floor(v)
	u := v - nf // exact
	n := int(nf)
	bh, bl, _ := ivBase(u, x) // a base that declines is the zero pair, which ivFinish refuses
	if n == 0 {
		return ivFinish(bh, bl, x, scaled)
	}
	N, J, _ := ivStart(n, x) // a start level that declines is N = J = 0: r_n = 0, refused by ivFinish
	// float64 levels: (a, b) = (r_{k+1}, r_k)
	a, b := 0.0, 1.0
	c2 := 2 / x
	for k := N; k > J; k-- {
		a, b = b, math.FMA(float64((u+float64(k))*c2), b, a)
	}
	// double-double levels J .. 1; the pairs carry r_{k+1}, r_k
	h0, l0, h1, l1 := a, 0.0, b, 0.0
	var nh, nl float64
	qh := 2 / x
	ql := math.FMA(-qh, x, 2) / x
	for k := J; k >= 1; k-- {
		if k == n {
			nh, nl = h1, l1
		}
		wh, wl := dd.TwoSum(u, float64(k))
		ch, ce := dd.TwoProd(wh, qh)
		cl := ce + float64(wh*ql) + float64(wl*qh)
		p, pe := dd.TwoProd(ch, h1)
		s, se := dd.TwoSum(h0, p)
		lo := (l0 + pe + se) + math.FMA(ch, l1, float64(cl*h1))
		h0, l0, h1, l1 = h1, l1, s, lo
	}
	// (a chain that overflowed, h1 Inf or NaN: nh/h1 is 0 or NaN, refused by ivFinish)
	// ratio r_n/r_0 as a pair, times the base
	q1 := nh / h1
	rem := math.FMA(-q1, h1, nh) + nl - float64(q1*l1)
	q2 := rem / h1
	ph, pe := dd.TwoProd(bh, q1)
	lo := pe + float64(bh*q2) + float64(bl*q1)
	rh, rl := dd.FastTwoSum(ph, lo)
	return ivFinish(rh, rl, x, scaled)
}
