package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Yn's and Jn's fast paths for 0 < x < 100 (the range of the cell
// polynomials and series of J0, J1, Y0, Y1): the three-term recurrence
//
//	C_{k+1}(x) = (2k/x) C_k(x) - C_{k-1}(x),   C = J, Y     (DLMF 10.6.1)
//
// carried in a "dd-light" form (as kn_fast.go: the hi chain with the exact
// error of each product and sum, the lo chain linear in the lo parts), started
// from J0, J1, Y0, Y1 as hi + lo pairs (jyCellPair: the cell polynomials
// before their last rounding; the series of Y0, Y1 in double-double). Yn runs
// it upward from Y0, Y1 (ynRecurErr); Jn upward from J0, J1 for n <= x
// (jnUp) and, for n > x, downward from the CF1 ratio J_{n+1}/J_n (Miller's
// algorithm, jnMiller), normalised by J0 or J1.
//
// Unlike K, J and Y oscillate for x > n, and a recurrence in that region
// amplifies nothing but does not damp anything either: an error in the start
// values or a rounding on the way is a solution of the recurrence, A J_k +
// B Y_k, and shows in the result as A J_n + B Y_n, which is large compared
// with the result next to a zero of it. The fast paths therefore carry a
// certificate: they bound that error from the start errors and the roundings
// they know about, and answer only when the bound is below 1.5 * 2^-51 (Yn)
// or 2^-51 (Jn) of the result: with the final rounding 6.5 ulp and 4.5 ulp at
// worst, the stated bounds. Otherwise they decline and Steed's method answers
// as before, bit for bit (about 1% of the refbench inputs, 95 of 9046 with
// n >= 2: TestJYFallbackRate). Measured on the mpmath rows they answer: Yn
// 1.40 ulp, Jn 1.04 ulp at worst (TestYnFastFixture, TestJnFastFixture).
//
// Error derivation (the Casoratian). Let d_k be an error that satisfies the
// recurrence with d_0 = e0, d_1 = e1. Then d_k = A J_k + B Y_k with
//
//	A = (e0 Y_1 - e1 Y_0) / D,  B = (e1 J_0 - e0 J_1) / D,
//	D = J_0 Y_1 - J_1 Y_0 = -2/(pi x)           (DLMF 10.5.2: the Wronskian
//	                                              J_{v+1} Y_v - J_v Y_{v+1} = 2/(pi z))
//
// (the recurrence is DLMF 10.6.1 for every cylinder function), so
// |d_n| <= (pi x / 2) [ (|e0 Y_1| + |e1 Y_0|) |J_n| + (|e1 J_0| + |e0 J_1|) |Y_n| ].
// A rounding that adds eta to C_{k+1} (the recurrence then runs from the
// pair (C_k, C_{k+1} + eta), the error pair (0, eta)) gives, in the same way,
// |d_n| <= (pi x / 2) eta (|Y_k| |J_n| + |J_k| |Y_n|).
//
// A priori bounds on J (the only place where J enters a bound for Y):
// |J_0| <= 1, |J_n| <= 2^-1/2 for n >= 1 (DLMF 10.14.1, opened 2026-11-13),
// and |J_n(x)| <= (x/2)^n / n! (DLMF 10.14.4 with nu = n, z = x real).
// Landau's sharper bounds (J. London Math. Soc. 61, 2000) were NOT opened and
// are not used. Next to a zero of Y_n, where |J_n| is large and matters, the
// Wronskian at n-1 bounds it instead, |J_n| <= (2/(pi x) + |J_{n-1}| |Y_n|) /
// |Y_{n-1}|, which is close to J_n itself there (it took the declined share of
// the refbench inputs from 3.5% to 1%). For Jn the bound of |Y_n| is that of a
// plain float64 recurrence carried alongside, itself bounded the same way.
//
// What the fast path knows: the start pairs are within an absolute error eb
// of the true values (jyCellPair for the cells, jyRhoSeries for the series;
// checked against the mpmath rows of the "j0", "j1", "y0", "y1" fixtures in
// TestJYStartError); each step rounds at most 2^-100 of its operands
// (the products, sums and the lo chain of kn_fast.go: cl l1 is 2^-106 of
// the value, the five float64 additions of the lo chain lose 2^-53 of a
// quantity that is itself 2^-52 of the value). With the lo chain removed the
// same rows reach 9e5 ulp (Yn, near zeros) and 2e4 (Jn), and the tests are
// red; with the acceptance test removed, 4e10 ulp.
const (
	// jyFastMinX: below, Y_n (n >= 3) leaves float64 within a few steps and
	// the certificate's sums of products of Y overflow (x^-3 is 2^300 here).
	jyFastMinX = 0x1p-100
	jyFastMaxX = besselZeroTaylorMax

	// jyCF1DD is the number of levels of CF1 (the lowest) evaluated in
	// double-double.
	jyCF1DD = 8

	// jyAcceptY, jyAcceptJ: the certified error must be below this fraction
	// of the result: 6 ulp at most for Yn (bound 6.5 with the final rounding),
	// 4 ulp for Jn (bound 4.5).
	jyAcceptY = 1.5 * 0x1p-51
	jyAcceptJ = 0x1p-51

	// stepEta is the rounding of one step relative to its operands.
	stepEta = 0x1p-100

	// jBound1 bounds |J_n| for n >= 1 (DLMF 10.14.1).
	jBound1 = 0.7071067811865476 * (1 + 0x1p-40)
)

// jyPair is a start value as hi + lo, with a bound eb of its absolute error.
type jyPair struct{ h, l, eb float64 }

// jyRhoSeries bounds the relative error of the series pairs of Y0
// (x <= 0.6) and Y1 (x <= 1.5), whose float64 parts are summed at 2^-53 of
// 0.2 of the value (measured: TestJYStartError).
const jyRhoSeries = 0x1p-52

// jyCellPair is besselCellEval before its last rounding, hi + lo (same
// arithmetic, s + r is its result bit for bit: TestJYCellPairIsCellEval), with
// the error bound of the pair. The only roundings that reach it are those of
// the float64 Horner sum of R, which h^2 scales: the Horner error is
// bounded by 2^-53 gamma sum |r_k| |h|^k with gamma <= 3 measured on the rows of
// the fixtures j0, j1, y0, y1 for |h| >= 0.02 (p99 2, max 3; the textbook
// bound with 24 u is 8 times that), taken 5 here, and a floor of 2^-100 of
// |value| + |z a1| for the double-double constants (the zero z + zlo is known
// to 2^-107 |z|, which a1 scales) and the tail's own rounding
// (TestJYStartError checks both on all rows, small |h| included).
func jyCellPair(cells []besselCell, lut []uint16, x float64) jyPair {
	i := int(lut[int(x*8)])
	for i+1 < len(cells) && x >= cells[i+1].lo {
		i++
	}
	c := &cells[i]
	he, hl := dd.TwoSum(x-c.z, -c.zlo)
	rc := &c.r
	t := rc[0]
	ta := math.Abs(rc[0])
	ah := math.Abs(he)
	for k := 1; k < 12; k++ {
		t = float64(t*he) + rc[k]
		ta = math.FMA(ta, ah, math.Abs(rc[k]))
	}
	wh, wl := dd.TwoSum(c.a1, float64(he*t))
	wl += c.a1lo
	p, pe := dd.TwoProd(he, wh)
	s, se := dd.TwoSum(c.a0, p)
	r := se + pe + float64(he*wl) + float64(hl*wh) + c.a0lo
	h, l := dd.TwoSum(s, r)
	eb := float64(5 * 0x1p-53 * float64(float64(he*he)*ta))
	return jyPair{h, l, eb + float64(0x1p-100*(math.Abs(s)+float64(math.Abs(c.z)*math.Abs(wh))))}
}

func besselJ0Pair(x float64) jyPair {
	return jyCellPair(besselJ0Cells[:], besselJ0CellsLUT[:], x)
}

func besselJ1Pair(x float64) jyPair {
	return jyCellPair(besselJ1Cells[:], besselJ1CellsLUT[:], x)
}

func seriesPair(d dd.DD) jyPair {
	return jyPair{d.Hi, d.Lo, float64(jyRhoSeries * math.Abs(d.Hi))}
}

// besselY0Pair is Y0(x) as a pair, for 0 < x < 100.
func besselY0Pair(x float64) jyPair {
	if x <= 0.6 {
		return seriesPair(besselY0SeriesDD(x))
	}
	return jyCellPair(besselY0Cells[:], besselY0CellsLUT[:], x)
}

func besselY1Pair(x float64) jyPair {
	if x <= 1.5 {
		return seriesPair(besselY1SeriesDD(x))
	}
	return jyCellPair(besselY1Cells[:], besselY1CellsLUT[:], x)
}

// jyStep is one step of the dd-light recurrence
// c_k = ch + cl = 2k/x (qh + ql = 2/x), new = c_k h1 - h0 (the lo chain
// alongside), and the size of the operands mu = |c_k h1| + |h0|, of which
// each step rounds at most stepEta.
func jyStep(kf, qh, ql, h0, l0, h1, l1 float64) (s, lo, mu float64) {
	ch, ce := dd.TwoProd(kf, qh)
	cl := ce + float64(kf*ql)
	p, pe := dd.TwoProd(ch, h1)
	s, se := dd.TwoSum(p, -h0)
	lo = (pe + se - l0) + math.FMA(ch, l1, float64(cl*h1))
	return s, lo, math.Abs(p) + math.Abs(h0)
}

// jyMin and jyMax are math.Min and math.Max without their handling of NaN
// and signed zeros (a call to an assembly routine on arm64): the operands
// here are finite and positive, and a NaN makes every certificate fail
// anyway.
func jyMin(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}

func jyMax(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

// jyInv2 returns 2/x as a pair.
func jyInv2(x float64) (qh, ql float64) {
	qh = 2 / x
	return qh, math.FMA(-qh, x, 2) / x
}

// ynRecur computes Yn(n, x) (n >= 2, x > 0) by the upward recurrence from
// Y0, Y1 with the certificate above; ok is false where it does not apply
// or cannot certify, and Steed's method must answer.
func ynRecur(n int, x float64) (r float64, ok bool) {
	r, _, ok = ynRecurErr(n, x)
	return
}

// ynRecurErr is ynRecur and the certified bound e of the absolute error of
// the pair h + l that the result rounds.
func ynRecurErr(n int, x float64) (r, e float64, ok bool) {
	if n < 0 {
		n = -n
	}
	if n < 2 || !(x >= jyFastMinX && x < jyFastMaxX) {
		return 0, 0, false
	}
	y0, y1 := besselY0Pair(x), besselY1Pair(x)
	qh, ql := jyInv2(x)
	h0, l0, h1, l1 := y0.h, y0.l, y1.h, y1.l
	hx := jyHalfPiX(x)
	// u = (x/2)^k / k!, bounds |J_k|: b = min(u, jBound1) for k >= 1.
	u := float64(x / 2)
	b := jyMin(u, jBound1)
	var s1, s2 float64 // sum mu |Y_k|, sum mu b_k over the steps
	bp := b            // bound of |J_{n-1}| at the end
	for k := 1; k < n; k++ {
		bp = b
		s, lo, mu := jyStep(float64(k), qh, ql, h0, l0, h1, l1)
		s1 += float64(mu * math.Abs(h1))
		s2 += float64(mu * b)
		h0, l0, h1, l1 = h1, l1, s, lo
		if !(math.Abs(s) <= math.MaxFloat64/4) { // overflow of Y_n, or NaN
			return 0, 0, false
		}
		u = float64(u*(x/2)) / float64(k+1)
		b = jyMin(u, jBound1)
	}
	yn := h1 + l1
	ay := math.Abs(yn)
	// A sharper bound of |J_n| where it matters, next to a zero of Y_n (the
	// oscillatory region), from the Wronskian at n-1 (DLMF 10.5.2):
	// J_n Y_{n-1} - J_{n-1} Y_n = 2/(pi x), so
	// |J_n| <= (2/(pi x) + |J_{n-1}| |Y_n|) / |Y_{n-1}|. Y_{n-1} is the
	// chain's own value, at most about 1e-14 off in absolute terms (2^-36
	// is taken away from it, and a relative 2^-30 added to Y_n).
	if ylow := math.Abs(h0) - 0x1p-36; ylow > 0 {
		w := 1 / float64(float64(1.5707963267948966*x)*(1-0x1p-40))
		alt := (w + float64(float64(bp*ay)*(1+0x1p-30))) / ylow
		if alt < b {
			b = alt
		}
	}
	// start errors, with |J_0| <= 1, |J_1| <= jBound1 first (enough where Y
	// dominates, x small), the computed J_0, J_1 only if that does not certify
	e = ynCertificate(hx, y0, y1, 1, jBound1, b, s1, s2, ay)
	if !(e <= jyAcceptY*ay) {
		j0, j1 := besselJ0Pair(x), besselJ1Pair(x)
		e = ynCertificate(hx, y0, y1, float64(math.Abs(j0.h)*(1+0x1p-40)), float64(math.Abs(j1.h)*(1+0x1p-40)), b, s1, s2, ay)
		if !(e <= jyAcceptY*ay) {
			return 0, 0, false
		}
	}
	return yn, e, true
}

// ynCertificate is the bound of the error of the Y recurrence at n: the
// start errors y0.eb, y1.eb (A J_n + B Y_n of the derivation above), and the
// roundings of the steps, summarised in s1 = sum mu_k |Y_k| and
// s2 = sum mu_k b_k (stepEta of each), with |J_n| <= bn, |J_0| <= aj0,
// |J_1| <= aj1.
func ynCertificate(hx float64, y0, y1 jyPair, aj0, aj1, bn, s1, s2, ay float64) float64 {
	a0, a1 := math.Abs(y0.h), math.Abs(y1.h)
	t1 := float64(y0.eb*a1) + float64(y1.eb*a0)
	t2 := float64(y1.eb*aj0) + float64(y0.eb*aj1)
	start := float64(t1*bn) + float64(t2*ay)
	steps := float64(stepEta * (float64(s1*bn) + float64(s2*ay)))
	return float64(float64(hx*(start+steps)) * (1 + 0x1p-40))
}

// jyHalfPiX is pi x / 2 rounded up (the 1 + 2^-40 covers the rounding of pi/2
// and of the product).
func jyHalfPiX(x float64) float64 {
	return float64(float64(1.5707963267948966*x) * (1 + 0x1p-40))
}

// jnUp computes Jn(n, x) for 2 <= n <= x < 100 by the upward recurrence from
// J0, J1. The recurrence runs through the oscillatory region, so the
// error is A J_n + B Y_n of the derivation, and |Y_n| must be bounded: Y is
// carried alongside in plain float64 (from the hi parts of Y0, Y1), whose own
// error is bounded in the same way: each step rounds at most 2^-50 of its
// operands (c y_k and y_{k-1}, at most 3 Ymax since 2k/x < 2), the hi parts
// start within 2^-51 Ymax, |J_k| <= 1 and pi x / 2 < 2^7.3, so with
// Ymax = max |Y_k| the error of y_n is at most
// c (Ymax + |Y_n|) with c = (n+3) 2^-41 Ymax (the Casoratian bound above),
// hence at most (4/3) c (Ymax + |y_n|) while c < 1/4; yBound takes
// 2 kappa (Ymax + |y_n|), kappa = (n+2) 2^-40 Ymax, which is larger, and
// the path declines when kappa >= 1/4.
func jnUp(n int, x float64) (r, e float64, ok bool) {
	if n < 2 || !(float64(n) <= x && x < jyFastMaxX) {
		return 0, 0, false
	}
	j0, j1 := besselJ0Pair(x), besselJ1Pair(x)
	y0, y1 := besselY0Pair(x), besselY1Pair(x)
	qh, ql := jyInv2(x)
	h0, l0, h1, l1 := j0.h, j0.l, j1.h, j1.l
	ya, yb := y0.h, y1.h
	ymax := jyMax(math.Abs(ya), math.Abs(yb))
	var s1, s2 float64 // sum mu_k, sum mu_k |J_k|
	for k := 1; k < n; k++ {
		kf := float64(k)
		s, lo, mu := jyStep(kf, qh, ql, h0, l0, h1, l1)
		s1 += mu
		s2 += float64(mu * math.Abs(h1))
		h0, l0, h1, l1 = h1, l1, s, lo
		ya, yb = yb, math.FMA(float64(kf*qh), yb, -ya)
		ymax = jyMax(ymax, math.Abs(yb))
	}
	jn := h1 + l1
	aj := math.Abs(jn)
	// |Y_n| <= yBound: the float64 chain's own certificate (see above).
	kappa := float64(float64(n+2) * 0x1p-40 * ymax)
	yBound := math.Abs(yb) + float64(2*kappa*(ymax+math.Abs(yb)))
	hx := jyHalfPiX(x)
	// start errors of J0, J1 (A J_n + B Y_n)
	a := float64(float64(float64(j0.eb*math.Abs(y1.h))+float64(j1.eb*math.Abs(y0.h))) * aj)
	bb := float64(float64(float64(j1.eb*math.Abs(j0.h))+float64(j0.eb*math.Abs(j1.h))) * yBound)
	steps := float64(stepEta * (float64(float64(s1*ymax)*aj) + float64(s2*yBound)))
	e = float64(float64(hx*(float64(a)+float64(bb)+steps)) * (1 + 0x1p-40))
	// kappa < 1/4 holds by far (Ymax <= 1 for n <= x, n + 2 < 102) and is
	// part of the same test: where it did not, the bound above is void
	if !(kappa < 0.25 && e <= jyAcceptJ*aj) {
		return 0, 0, false
	}
	return jn, e, true
}

// jnMiller computes Jn(n, x) for 2 <= n, x < n, x < 100 by Miller's
// downward recurrence: the ratio J_{n+1}/J_n from CF1 (A&S 9.1.73) evaluated
// backwards in float64 with a running error bound, then the recurrence down
// to J_1, J_0 in dd-light form, and J_n = J_0 / J_0^rec (J_1 / J_1^rec where
// |J_1| is the larger), J_0, J_1 as pairs.
//
// J is the minimal solution of the recurrence for k > x: the start error
// of CF1 does not grow going down, and from k = x down to 0 the recurrence
// is oscillatory and neutral. A relative error eps of the ratio is an
// absolute error eps J_{n+1} in J_{n+1}; the Casoratian at (n, n+1) turns it
// into (pi x/2) eps J_{n+1} (|Y_0| J_n + |J_0| |Y_n|) at J_0, i.e. relative
// to the larger of J_0, J_1 (so that |Y_0/J_0| is not large) at most
// 3 eps (pi x/2) J_{n+1} |Y_n|, and the Wronskian at n gives
// (pi x/2) J_{n+1} |Y_n| = r/(Y_{n+1}/Y_n - r) with r = J_{n+1}/J_n, which is at most
// r / (2n/x - 1 - r) since Y_{n+1}/Y_n >= 2n/x - 1 (the recurrence, with
// |Y_{n-1}| <= |Y_n| for n > x). This is the "amp" below. CF1 itself is
// evaluated backwards from level K = 12 + 2 sqrt(x) in float64, its last 8
// levels in double-double, with the running bound e_j = r_j^2 e_{j+1} + (the
// level's own rounding), started from e = 1 (the truncation r_{K+1} = 0 is
// off by at most 1 if J_{v+1} <= J_v for v > x, and |Y_{n-1}| <= |Y_n|
// above: both taken as known, neither from a source opened here; the
// fixtures check the results, not these), which makes the truncation
// certified as well. The error of the start pair of J_0 shows
// directly (the result is proportional to it), and the roundings of the
// steps at most n (pi x / 2) 2^-96 of the result.
func jnMiller(n int, x float64) (r, e float64, ok bool) {
	// (jnMillerStart is a variable only so that a test can hand jnMiller a
	// start pair with a large error: no (n, x) in range reaches the final
	// comparison with rel above jyAcceptJ, since the chain overflows first for
	// the n that would, so the comparison could not otherwise be held.)
	if n < 2 || !(x >= jyFastMinX && x < jyFastMaxX && float64(n) > x) {
		return 0, 0, false
	}
	// CF1: r_j = 1/(b_j - r_{j+1}), b_j = 2(n+j)/x, from level K; e bounds
	// |r_j - true| (e_j = r_j^2 e_{j+1} + 2^-50 r_j: the truncation r_{K+1} = 0
	// is off by at most 1 since J_{v+1} <= J_v for v > x).
	kk := 12 + int(2*math.Sqrt(x))
	qh, ql := jyInv2(x)
	rj, rl, ec := 0.0, 0.0, 1.0
	nf := float64(n)
	j := kk
	for ; j > jyCF1DD; j-- {
		// b_j = (n+j) (qh + ql): one rounding (n + j is exact)
		m := nf + float64(j)
		bj := math.FMA(m, qh, float64(m*ql))
		rj = 1 / (bj - rj)
		// each level rounds b_j (r_j^2 b_j u <= 1.5 u r_j), the difference
		// and the quotient: 4 u r_j
		ec = float64(float64(rj*rj)*ec) + float64(4*0x1p-53*rj)
	}
	// the last levels in double-double: their roundings are the ones that
	// no later level damps (each level below damps by r_j^2 only), 2^-98 r_j
	for ; j >= 1; j-- {
		m := nf + float64(j)
		bh, pe := dd.TwoProd(m, qh)
		bl := pe + float64(m*ql)
		wh, we := dd.TwoSum(bh, -rj)
		wh, wl := dd.FastTwoSum(wh, we+bl-rl)
		nr := 1 / wh
		rl = float64((math.FMA(-nr, wh, 1) - float64(nr*wl)) * nr)
		rj = nr
		ec = float64(float64(rj*rj)*ec) + float64(0x1p-98*rj)
	}
	// amp bounds the factor by which a relative error of the ratio shows in
	// J_0^rec (see above): 3 r/(2n/x - 1 - r) from Y_{n+1}/Y_n >= 2n/x - 1.
	den := float64(nf*qh) - 1 - rj
	cf1err := float64(float64(3*rj/den) * (ec / rj))
	if !(den > 0 && cf1err <= 0.75*jyAcceptJ) {
		return 0, 0, false
	}
	h0, l0, h1, l1 := rj, rl, 1.0, 0.0 // J_{n+1}^rec, J_n^rec
	for k := n; k >= 1; k-- {
		s, lo, _ := jyStep(float64(k), qh, ql, h0, l0, h1, l1)
		h0, l0, h1, l1 = h1, l1, s, lo
		if !(math.Abs(s) <= 0x1p1000) {
			return 0, 0, false
		}
	}
	// now (h0, l0) = J_1^rec, (h1, l1) = J_0^rec
	j0, j1 := jnMillerStart(x)
	t, rh, rl := j0, h1, l1
	if math.Abs(j1.h) > math.Abs(j0.h) {
		t, rh, rl = j1, h0, l0
	}
	// The error of the pair relative to the result: CF1's, the start error
	// of J_0 (J_1), which shows directly (the result is proportional to it),
	// and the roundings of the n steps, each stepEta of its operands, which
	// the oscillatory stretch carries to J_0 at most as (pi x/2) times
	// (a few)^1 (stepEta 2^-100, so n pi x 2^-96 bounds them).
	rel := cf1err + t.eb/math.Abs(t.h) + float64(float64(n)*1.5707963267948966*x*0x1p-96)
	q := t.h / rh
	rem := math.FMA(-q, rh, t.h) + t.l - float64(q*rl)
	r = q + rem/rh
	// (r and e are meaningless where ok is false; with cf1err already below
	// 3/4 of jyAcceptJ this fails only if the start error of J_0 is large)
	return r, float64(rel * math.Abs(r)), rel <= jyAcceptJ
}

// jnMillerStart returns the start pairs of J_0 and J_1 at x for jnMiller.
var jnMillerStart = func(x float64) (j0, j1 jyPair) {
	return besselJ0Pair(x), besselJ1Pair(x)
}

// jnFast is Jn's recurrence path for n >= 2 and 0 < x < 100.
func jnFast(n int, x float64) (float64, bool) {
	r, _, ok := jnFastErr(n, x)
	return r, ok
}

// jnFastErr is jnFast and the certified bound of the absolute error of the
// value it rounds.
func jnFastErr(n int, x float64) (r, e float64, ok bool) {
	if float64(n) <= x {
		return jnUp(n, x)
	}
	return jnMiller(n, x)
}
