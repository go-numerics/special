package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Kv and Kve fast paths for real order. The scheme is that of Bessels.jl's
// besselk (JuliaMath/Bessels.jl, MIT, src/BesselFunctions/besselk.jl, read
// 2026-11-13 at 009b615): Debye's uniform expansion for large orders
// (DLMF 10.41.4), and for moderate order and large x the large-argument
// expansion (DLMF 10.40.2) of two consecutive orders followed by the upward
// recurrence. Its accuracy is not taken (Bessels.jl tolerates about 2e-11
// relative error): the exponent and the prefactor are carried in
// double-double, the sums are split so that their float64 rounding is 2^-58
// of the result, and the recurrence is the "dd-light" chain of kn_fast.go.
// Everything else (small x, moderate x with a fractional order: Temme's
// series, Steed's CF2) is answered by besselIK as before, bit for bit.
//
// Path 1, v in [32, 1100], x < 1e6 (kvDebye). With z = x/v, S = sqrt(v^2 +
// x^2) = v sqrt(1+z^2), p = v/S and v eta = S + v ln(x/(v+S)) (DLMF 10.41.7),
//
//	K_v(x) = sqrt(pi/(2S)) e^(-v eta) (1 + sum_{k>=1} (-1)^k U_k(p)/v^k),
//
// (10.41.4; (1+z^2)^(-1/4) sqrt(1/v) = S^(-1/2)). U_k(p) = p^k H_k(p^2) and
// p^k/v^k = S^-k, so term k is (-1/S)^k H_k(p^2); H_k is Horner in float64
// from the exact rational coefficients of zz_kvDebye_coeffs.go (DLMF 10.41.9).
//   - S is a double-double (v^2, x^2 by TwoProd, one Newton square root:
//     2^-104). ln(x/(v+S)) = ln x - ln(v+S.Hi) - S.Lo/(v+S.Hi) with dd.Log
//     (2^-72 each): the exponent v eta is then within about v 3 2^-72 |ln x|
//     absolute (the logarithms are the only terms of size; S is 2^-104), so
//     with v <= 1100 and |ln x| <= 14 below 2^-56 of the result (0.06 ulp,
//     a first-order estimate: it is not asserted by itself, the fixtures hold
//     the sum of everything, v up to 1100 and x up to 99999, at 0.51 ulp).
//     Float64 alone would lose v |eta| 2^-53, about 700 ulp: that is the
//     2e3 ulp of Bessels.jl on the finite rows of the fixtures.
//   - the sum: |sum of terms k >= 1| <= 1/(12 v) (U_1(1) = -1/12 is its
//     largest term) so its float64 rounding is at most 2^-53/(12 * 32) * 17
//     additions < 2^-57 of the result.
//   - truncation: the loop stops when max |U_{k+1}|/v^(k+1) is below 2^-64
//     (a bound over p in [0, 1], never the computed term, which can vanish at
//     a root of U_k) and declines if k = 16 is reached without. NOT proven:
//     the terms beyond k+1 are taken to be smaller still, and Olver's bound for the remainder
//     (Olver 1997, Asymptotics and Special Functions, pp. 374-378, cited by
//     DLMF 10.41(iii)) was not opened. What stands in place of it: max |U_k|
//     on [0,1] (go run ./testdata/gen/kvdebye -report) puts the first
//     neglected term of the series, U_17/v^17, below 2^-64 for v >= 24.5,
//     and the fixtures hold the result (TestKvFastFixture).
//
// Path 2, v < 32, x >= kvAsymMinX (kvAsym): the large-argument expansion
// (DLMF 10.40.2) K_w(x) e^x = sqrt(pi/(2x)) sum a_k(w)/x^k, a_k from 10.40.1's
// term ratio ((4w^2 - (2k-1)^2)/(8kx)), for w = u and u+1 where u = v - floor(v)
// in [0, 1), then Ke_{k+1} = Ke_{k-1} + (2(u+k)/x) Ke_k (DLMF 10.29.1) in
// hi + lo as kn_fast.go, finally times e^-x from dd.ExpMant. Truncation:
// for real w and x > 0 the remainder does not exceed the first neglected term
// in absolute value (DLMF 10.40(ii), opened 2026-11-13: it states it for the
// sum truncated at k = l-1, with the same sign for l >= max(|w| - 1/2, 1)): the
// loop stops at a term below 2^-66 and declines if the terms grow first. The
// first term (w^2 up to 4 (u+1)^2 = 16) is computed in double-double, the
// rest in float64 (below 0.003 of the sum, 2^-62), the terms summed with
// TwoSum: the start pairs are within 2^-60 relative. The chain: kn_fast.go's
// derivation, R_n <= 2^-60 + n 2^-91 with n <= 31, and the order u + k is
// carried as the exact pair TwoSum(u, k).
//
// Total: 0.5 ulp rounding + 2^-57 (Debye) or 2^-59 (chain) + 2^-76 for the
// exponential: below 0.6 ulp against the exact value (TestKvFastFixture
// bounds it at kvFastBound), and no result below 2^-1000 is answered (a
// subnormal would be rounded twice).
const (
	kvDebyeMinV = 32
	kvDebyeMaxV = 1100
	kvDebyeMaxX = 1e6
	// kvDebyeTerms is the highest k of the Debye sum.
	kvDebyeTerms = 16
	// kvExpWindow: the exponent of the Debye product must stay in
	// (-kvExpWindow, kvExpWindow) so that the result is a normal float64.
	kvExpWindow = 700
	kvExpInf    = 760

	// kvAsymMinX is where the large-argument expansion of an order below 2
	// reaches 2^-66 (terms of a_k/x^k fall below it before they grow).
	kvAsymMinX = 24

	kvFastBound = 0.6
)

// kvFast returns K_v(x) (scaled: K_v(x) e^x) for v > 1, finite x > 0, with ok
// false where no fast path applies.
func kvFast(v, x float64, scaled bool) (r float64, ok bool) {
	if v >= kvDebyeMinV {
		if v <= kvDebyeMaxV && x < kvDebyeMaxX {
			return kvDebye(v, x, scaled)
		}
		return 0, false
	}
	// Where the large-argument expansion of the order v itself converges
	// (|4v^2 - 1| <= 4x, besselAsymIK) it is one series and about 60 ns:
	// that stays the answer there. The chain is for the orders it cannot reach.
	if x >= kvAsymMinX && float64(4*v*v)-1 > float64(4*x) {
		return kvAsymChain(v, x, scaled)
	}
	return 0, false
}

// kvDebyeExponent returns S = sqrt(v^2 + x^2) and the exponent -v eta (+ x if
// scaled) of the Debye product, both as double-doubles.
func kvDebyeExponent(v, x float64, scaled bool) (S, e dd.DD) {
	v2h, v2l := dd.TwoProd(v, v)
	x2h, x2l := dd.TwoProd(x, x)
	S = dd.Sqrt(dd.Add(dd.DD{Hi: v2h, Lo: v2l}, dd.DD{Hi: x2h, Lo: x2l}))
	vs := dd.AddF(S, v)
	lnt := dd.Sub(dd.Log(x), dd.Log(vs.Hi))
	lnt = dd.AddF(lnt, -vs.Lo/vs.Hi)
	e = dd.Neg(dd.Add(S, dd.MulF(lnt, v))) // -v eta
	if scaled {
		e = dd.AddF(e, x)
	}
	return S, e
}

// kvDebye is Path 1 above.
func kvDebye(v, x float64, scaled bool) (float64, bool) {
	S, e := kvDebyeExponent(v, x, scaled)
	switch kvExpClass(e.Hi) {
	case 1:
		return math.Inf(1), true
	case -1:
		return 0, true
	case 2:
		return 0, false
	}
	m, k := dd.ExpMant(e)
	c := dd.Sqrt(dd.Div(besselPiOver2DD, S))
	s, _, done := kvDebyeTail(v, S.Hi)
	if !done {
		return 0, false
	}
	uh, ul := dd.TwoSum(1, s)
	r := dd.Mul(c, m)
	p := dd.Add(dd.MulF(r, uh), dd.F(float64(r.Hi*ul)))
	// No guard on the result: inside the window it is normal and finite. The
	// product is above e^-700 c (1 - 1%) with c = sqrt(pi/(2S)) >= 1.2e-3
	// (S < 1e6 + 1100), 1.2e-307, and below e^700 c 1.01, 2e305.
	return math.Ldexp(p.Hi+p.Lo, k), true
}

// kvDebyeTail returns the sum of the terms k >= 1 of the Debye series, the
// index of the last one added, and whether the first neglected term is below
// 2^-64. That is decided by max |U_{k+1}| / v^(k+1) (kvDebyeMaxU), a bound
// that does not depend on p: stopping on a computed term is wrong, because
// U_k(p) has roots in (0, 1) and the term k can be small by accident while
// the term k+1 is not (Kv(32.5, 3.76), Kv(1000, 816.5): up to 1e9 ulp).
func kvDebyeTail(v, S float64) (s float64, last int, done bool) {
	invS := 1 / S
	invV := 1 / v
	q := float64(v / S)
	q = float64(q * q)
	pw := 1.0
	vk := 1.0 // v^-k
	for kk := 1; kk <= kvDebyeTerms; kk++ {
		pw = float64(-pw * invS)
		vk = float64(vk * invV)
		cs := &kvDebyeU[kk-1]
		h := cs[kk]
		for j := kk - 1; j >= 0; j-- {
			h = math.FMA(h, q, cs[j])
		}
		s += float64(pw * h)
		last = kk
		if float64(kvDebyeMaxU[kk+1]*float64(vk*invV)) < 0x1p-64 {
			return s, last, true
		}
	}
	return s, last, false
}

// kvExpClass classifies the exponent e of the Debye product: 0 inside the
// window (|e| < kvExpWindow: the product is a normal float64), 1 where the
// result overflows whatever the prefactor (c >= 1e-3, the sum within 1% of 1:
// e is more than kvExpInf, 50 past where e^e c crosses MaxFloat64), -1 where it
// underflows to 0 (below -kvExpInf: under half the smallest subnormal), 2 for
// the band between, and NaN, where besselIK answers.
func kvExpClass(e float64) int {
	switch {
	case e > kvExpInf:
		return 1
	case e < -kvExpInf:
		return -1
	case math.Abs(e) < kvExpWindow:
		return 0
	}
	return 2
}

// kvKeep reports whether the unscaled chain result is a float64 the fast path
// may return: a result below 2^-1000 (or the zero of e^-x underflowing) may be
// subnormal, which the final product would round twice.
func kvKeep(res float64) bool { return res >= 0x1p-1000 }

// kvAsymPair returns K_w(x) e^x for 0 <= w < 2 and x >= kvAsymMinX as hi + lo
// by DLMF 10.40.2, ok false if the terms grow before falling below 2^-66.
func kvAsymPair(w, x float64) (h, l float64, ok bool) {
	m2 := dd.MulFF(w, w)
	mu := dd.DD{Hi: float64(4 * m2.Hi), Lo: float64(4 * m2.Lo)}
	t1 := dd.DivF(dd.AddF(mu, -1), float64(8*x))
	sh, sl := t1.Hi, t1.Lo
	t := t1.Hi
	inv := 1 / float64(8*x)
	prev := math.Abs(t)
	done := math.Abs(t) < 0x1p-66
	// No cap on k: the ratio ((2k-1)^2 - mu)/(8kx) of consecutive terms grows
	// without bound, so the terms either fall below 2^-66 or start growing,
	// which returns (at x >= kvAsymMinX within 30 steps: TestKvAsymTermCount).
	for k := 2; !done; k++ {
		o := float64(2*k - 1)
		t = float64(t * float64(float64(mu.Hi-float64(o*o))*inv/float64(k)))
		at := math.Abs(t)
		if at > prev {
			return 0, 0, false
		}
		prev = at
		var e float64
		sh, e = dd.TwoSum(sh, t)
		sl += e
		done = at < 0x1p-66
	}
	uh, ue := dd.TwoSum(1, sh)
	c := dd.Sqrt(dd.DivF(besselPiOver2DD, x))
	r := dd.Mul(c, dd.DD{Hi: uh, Lo: ue + sl})
	return r.Hi, r.Lo, true
}

// kvAsymChain is Path 2 above.
func kvAsymChain(v, x float64, scaled bool) (float64, bool) {
	n := math.Floor(v)
	u := v - n // exact
	h0, l0, ok0 := kvAsymPair(u, x)
	h1, l1, ok1 := kvAsymPair(u+1, x)
	if !(ok0 && ok1) {
		return 0, false
	}
	qh := 2 / x
	ql := math.FMA(-qh, x, 2) / x
	for k := 1; k < int(n); k++ {
		wh, wl := dd.TwoSum(u, float64(k))
		ch, ce := dd.TwoProd(wh, qh)
		cl := ce + float64(wh*ql) + float64(wl*qh)
		p, pe := dd.TwoProd(ch, h1)
		s, se := dd.TwoSum(h0, p)
		lo := (l0 + pe + se) + math.FMA(ch, l1, float64(cl*h1))
		h0, l0, h1, l1 = h1, l1, s, lo
	}
	// (v > 1, so n >= 1 and the chain ends at h1)
	h, l := h1, l1
	if scaled {
		return h + l, true
	}
	m, k := dd.ExpMant(dd.DD{Hi: -x})
	p, pe := dd.TwoProd(m.Hi, h)
	res := math.Ldexp(p+(pe+float64(m.Hi*l)+float64(m.Lo*h)), k)
	if !kvKeep(res) {
		return 0, false
	}
	return res, true
}
