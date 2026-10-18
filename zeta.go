package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// bernoulliOverFactorial[j-1] = B_2j / (2j)!, j = 1..15 (mpmath).
var bernoulliOverFactorial = [...]float64{
	0.08333333333333333,
	-0.001388888888888889,
	3.306878306878307e-05,
	-8.267195767195768e-07,
	2.08767569878681e-08,
	-5.284190138687493e-10,
	1.3382536530684679e-11,
	-3.3896802963225827e-13,
	8.586062056277845e-15,
	-2.174868698558062e-16,
	5.5090028283602295e-18,
	-1.3954464685812522e-19,
	3.534707039629467e-21,
	-8.953517427037546e-23,
	2.267952452337683e-24,
}

// bernoulliOverFactorial25[j-1] = B_2j / (2j)!, j = 1..25 (mpmath), for
// hurwitzFast's longer Bernoulli tail.
var bernoulliOverFactorial25 = [...]float64{
	0.08333333333333333,
	-0.001388888888888889,
	3.306878306878307e-05,
	-8.267195767195768e-07,
	2.08767569878681e-08,
	-5.284190138687493e-10,
	1.3382536530684679e-11,
	-3.3896802963225827e-13,
	8.586062056277845e-15,
	-2.174868698558062e-16,
	5.5090028283602295e-18,
	-1.3954464685812522e-19,
	3.534707039629467e-21,
	-8.953517427037546e-23,
	2.267952452337683e-24,
	-5.744790668872202e-26,
	1.455172475614865e-27,
	-3.6859949406653103e-29,
	9.336734257095045e-31,
	-2.36502241570063e-32,
	5.990671762482134e-34,
	-1.5174548844682903e-35,
	3.843758125454189e-37,
	-9.736353072646691e-39,
	2.466247044200681e-40,
}

// scaled is a value m * 2^e with a double-double mantissa, used to sum
// terms whose magnitudes would overflow or underflow float64 on their own.
type scaled struct {
	m dd.DD
	e int
}

// powNegDD returns |b|^(-s) as a scaled value for a double-double base b
// (nonzero, finite): exp(-s log|b|) with the logarithm in double-double.
func powNegDD(b dd.DD, s float64) scaled {
	if b.Hi < 0 {
		b = dd.Neg(b)
	}
	m, e := dd.ExpMant(dd.MulF(logDD(b), -s))
	return scaled{m, e}
}

// sumScaled adds scaled terms with signs; the result is m * 2^e.
func sumScaled(terms []scaled) scaled {
	e0 := math.MinInt32
	for _, t := range terms {
		if t.m.Hi != 0 && t.e > e0 {
			e0 = t.e
		}
	}
	if e0 == math.MinInt32 {
		return scaled{}
	}
	var s dd.DD
	for _, t := range terms {
		if t.m.Hi != 0 {
			s = dd.Add(s, dd.Ldexp(t.m, t.e-e0))
		}
	}
	return scaled{s, e0}
}

// hurwitzPos returns ζ(s, q) for finite s > 1 and a double-double q > 0 as
// a scaled value.
//
// Euler–Maclaurin summation (DLMF 25.11.5 with 2.10.1; the arrangement of
// Cephes zeta.c, Moshier): ζ(s,q) = Σ_{k<N} (q+k)^-s + w^(1-s) [1/(s-1) +
// 1/(2w) + Σ_j B_2j/(2j)! (s)_{2j-1} w^-2j], w = q+N. Unlike Cephes, q+k is
// carried exactly in double-double, each power is exp(-s log(q+k)) with
// both steps in double-double, and the terms are summed with a common
// exponent so that neither q^-s nor w^(1-s) can overflow or underflow on
// its own. N is chosen so that w >= max(10, 0.7(s+30)): the ratio of
// consecutive Bernoulli terms, about (s+2j)²/(2πw)², is then below 0.052 and
// fifteen terms reach 2^-64; for large s the direct sum stops as soon as a
// term falls below 2^-80 of the first.
func hurwitzPos(s float64, q dd.DD) scaled {
	w0 := math.Max(10, 0.7*(s+30))
	n := 0.0
	if q.Hi < w0 {
		n = math.Ceil(w0 - q.Hi)
	}
	// n is huge for huge s (the sum then stops early): no capacity from it
	terms := make([]scaled, 0, int(min(n, 63))+1)
	for k := 0.0; k < n; k++ {
		t := powNegDD(dd.AddF(q, k), s)
		if t.m.Hi == 0 || math.IsInf(t.m.Hi, 0) {
			// beyond the exponential's range (|s log(q+k)| > 1e7): the
			// first term decides (0 or +Inf), later ones are smaller still
			if k == 0 {
				return t
			}
			return sumScaled(terms)
		}
		terms = append(terms, t)
		if k > 0 && t.e < terms[0].e-80 {
			// (q+k)^-s < 2^-79 q^-s: the remaining terms and the tail
			// (at most (q+k)/(s-1) times the current term) cannot reach
			// the result's last bit.
			return sumScaled(terms)
		}
	}
	w := dd.AddF(q, n)
	lw := logDD(w)
	wm, we := dd.ExpMant(dd.MulF(lw, 1-s)) // w^(1-s)
	// bracket: 1/(s-1) + 1/(2w) + Σ_j c_j (s)_{2j-1} w^-2j
	br := dd.Add(recipDD(s-1), dd.MulF(dd.Div(dd.F(1), w), 0.5))
	wf := w.Hi
	iw2 := 1 / (wf * wf)
	poch := s // (s)_{2j-1}
	pw := iw2 // w^-2j
	tail := 0.0
	for j := 1; j <= len(bernoulliOverFactorial); j++ {
		t := bernoulliOverFactorial[j-1] * poch * pw
		tail += t
		if math.Abs(t) < 0x1p-64*math.Abs(br.Hi) {
			break
		}
		k := float64(2*j - 1)
		poch *= (s + k) * (s + k + 1)
		pw *= iw2
	}
	br = dd.AddF(br, tail)
	terms = append(terms, scaled{dd.Mul(wm, br), we})
	return sumScaled(terms)
}

// Zeta returns the Riemann zeta function ζ(s) for real s.
//
// Special values: ζ(1) = +Inf, ζ(+Inf) = 1, ζ(0) = -1/2, ζ(-2k) = 0 for
// integers k >= 1; NaN for NaN and -Inf. ζ overflows to ±Inf for
// s < -260 (roughly).
//
// Algorithm: on [-1, 6], ζ(s) = 1/(s-1) + R(s) with R entire and fitted to
// mpmath (zz_zetaPieces_coeffs.go), so the pole is exact; on (6, 64),
// 1 + g(s) 2^-s with g fitted to mpmath; for s < -1 the reflection
// formula ζ(s) = 2^s π^(s-1) sin(πs/2) Γ(1-s) ζ(1-s) (DLMF 25.4.1) with the
// powers and the gamma function combined in double-double before one
// exponential and sin(πs/2) reduced exactly, so the trivial zeros are exact
// and nearby values keep their relative accuracy.
func Zeta(s float64) float64 {
	switch {
	case math.IsNaN(s) || math.IsInf(s, -1):
		return math.NaN()
	case s == 1:
		return math.Inf(1)
	case s >= 64:
		// ζ(s) - 1 < 2^-63: rounds to 1.
		return 1
	case s > 6:
		return zetaTailFast(s)
	case s >= -1:
		return zetaMidFast(s)
	}
	return zetaReflect(s)
}

var (
	zetaEst     = estPieces(zetaPieces[:])
	zetaTailEst = estPieces(zetaTailPieces[1:]) // piece 0 has degree 17
)

// zetaMidFast returns ζ(s) for -1 <= s <= 6, s != 1, as Zeta's
// 1/(s-1) + R(s) with the reciprocal and R as hi + lo (float64 kernels
// of fastpoly.go instead of double-double).
func zetaMidFast(s float64) float64 {
	dh := s - 1 // TwoSum
	bb := dh - s
	dl := (s - (dh - bb)) + (-1 - bb)
	q := 1 / dh
	ql := (math.FMA(-q, dh, 1) - q*dl) * q // 1/(s-1) = q + ql
	rh, rl := zetaEst[min(int((s+1)*2), len(zetaEst)-1)].evalShort(s)
	v := q + rh // TwoSum
	bb = v - q
	ve := (q - (v - bb)) + (rh - bb)
	return v + (ve + (ql + rl))
}

// zetaTailFast returns ζ(s) for 6 < s < 64 as 1 + g(s) 2^-s, g from the
// fitted pieces as hi + lo, 2^-s = exp(-s ln 2) from the exp kernel with
// s ln 2 carried exactly, one rounding.
func zetaTailFast(s float64) float64 {
	var gh, gl float64
	if i := int((s - 6) * 0.5); i == 0 {
		g := evalPiecesDD(zetaTailPieces[:1], s)
		gh, gl = g.Hi, g.Lo
	} else {
		gh, gl = zetaTailEst[i-1].eval(s)
	}
	xh := float64(-s * ln2DD.Hi)
	xl := math.FMA(-s, ln2DD.Hi, -xh) - s*ln2DD.Lo
	eh, el, k := expHL(xh, xl)
	sc := pow2(k) // k >= -65
	th := float64(eh * gh)
	tl := math.FMA(eh, gh, -th) + math.FMA(eh, gl, el*gh)
	th *= sc
	tl *= sc
	r := 1 + th // Fast2Sum: ζ(s) - 1 < 0.02
	return r + (((1 - r) + th) + tl)
}

// zetaAbove2 returns ζ(s) for s > 2 as a double-double: on (2, 6] from the
// fitted regular part 1/(s-1) + R(s); on (6, 64) as 1 + g(s) 2^-s with
// g(s) = (ζ(s) - 1) 2^s fitted to mpmath (zz_zetaTailPieces_coeffs.go) and
// 2^-s = exp(-s ln 2) in double-double; beyond 64, 1 + 2^-s.
func zetaAbove2(s float64) dd.DD {
	switch {
	case s >= 64:
		return dd.AddF(dd.F(1), math.Exp2(-s))
	case s <= 6:
		return dd.Add(recipDD(s-1), evalPiecesDD(zetaPieces[:], s)) // s-1 exact
	}
	m, e := dd.ExpMant(dd.MulF(ln2DD, -s))
	g := evalPiecesDD(zetaTailPieces[:], s)
	return dd.AddF(dd.Ldexp(dd.Mul(m, g), e), 1)
}

// zetaReflect returns ζ(s) for s < -1 by the reflection formula.
func zetaReflect(s float64) float64 {
	sn := sinPiDD(s / 2) // s/2 exact
	if sn.Hi == 0 {
		return 0 // trivial zero
	}
	t := -s // exact; Γ(1-s) = Γ(1+t) = t Γ(t)
	// L = s log 2 + (s-1) log π + log t + log Γ(t)
	l := dd.Add(dd.MulF(ln2DD, s), dd.Sub(dd.MulF(lnPi, s), lnPi))
	l = dd.Add(l, dd.Log(t))
	l = dd.Add(l, logGammaPos(t))
	m, e := dd.ExpMant(l)
	if math.IsInf(m.Hi, 0) {
		return math.Copysign(math.Inf(1), sn.Hi)
	}
	// ζ(1-s) = ζ(1+t); 1+t rounds, which ζ'(u)/ζ(u) ~ -2^-u ln 2 (u > 3)
	// makes harmless.
	z := zetaAbove2(1 + t)
	return math.Ldexp(dd.Mul(dd.Mul(m, sn), z).Float(), e)
}

// HurwitzZeta returns the Hurwitz zeta function ζ(s, q) = Σ_{k>=0} (k+q)^-s
// for s > 1.
//
// Special values: +Inf at s = 1 and at q = 0, -1, -2, ...; NaN for s < 1,
// for NaN arguments and for q < 0 with non-integer s (where (k+q)^-s is
// not real); ζ(s, +Inf) = 0.
//
// Algorithm: Euler–Maclaurin summation in double-double (hurwitzPos). For
// q < 0 and integer s, the negative terms (q+k)^-s, k < ceil(-q), are summed
// explicitly with their signs and the rest by hurwitzPos.
func HurwitzZeta(s, q float64) float64 {
	switch {
	case math.IsNaN(s) || math.IsNaN(q):
		return math.NaN()
	case s == 1:
		return math.Inf(1)
	case s < 1:
		return math.NaN()
	case math.IsInf(q, 1):
		return 0
	case q <= 0 && q == math.Trunc(q):
		return math.Inf(1)
	case q < 0 && s != math.Trunc(s):
		return math.NaN()
	case math.IsInf(s, 1):
		switch {
		case q > 1:
			return 0
		case q == 1:
			return 1
		case q > 0:
			return math.Inf(1)
		}
		return math.NaN()
	}
	if v, ok := hurwitzFast(s, q); ok {
		return v
	}
	h := hurwitzAny(s, q)
	return math.Ldexp(h.m.Float(), h.e)
}

// powNegHL returns (yh + yl)^-s as hi + lo for yh > 0 with the result in
// the normal range (the caller checks): exp(-s log y) with log y from the
// log kernel (2^-68 absolute), -s log y formed exactly up to its low-order
// terms, and the exp kernel (2^-68 relative); about 2^-62 relative in all
// for s log y up to 700.
func powNegHL(yh, yl, s float64) (hi, lo float64) {
	// logHL and expHL written out: this is the inner loop of HurwitzZeta.
	iyb := math.Float64bits(yh) // yh >= 2^-20 is normal
	tmp := iyb - 0x3fe8000000000000
	kf := float64(int64(tmp) >> 52)
	m := math.Float64frombits(iyb - uint64(int64(tmp)>>52)<<52)
	t := &logT[int(m*256+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1
	r2 := r * r
	q0 := math.FMA(r, 1.0/3, -0.5)
	q1 := math.FMA(r, 1.0/5, -0.25)
	q2 := math.FMA(r, 1.0/7, -1.0/6)
	q2 = math.FMA(r2, -1.0/8, q2)
	q1 = math.FMA(r2, q2, q1)
	q := r2 * math.FMA(r2, q1, q0)
	l0 := kf*ln2Hi42 + t[1]
	lh := l0 + r
	ll := (r - (lh - l0)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q)) + yl/yh
	ph := float64(-s * lh)
	pl := math.FMA(-s, lh, -ph) - s*ll
	xh := ph + pl // normalised for the reduction
	xl := pl - (xh - ph)
	kd := math.RoundToEven(xh * expInvLn2N)
	ki := int(kd)
	er := xh - kd*expC1
	dr := xl - kd*expC2
	u := er + dr
	ue := dr - (u - er)
	u2 := u * u
	pa := math.FMA(u, 1.0/6, 0.5)
	pb := math.FMA(u, 1.0/120, 1.0/24)
	pb = math.FMA(u2, 1.0/720, pb)
	pu := u2 * math.FMA(u2, pb, pa)
	et := &expT[ki&127]
	th, tl := et[0], et[1]
	tu := float64(th * u)
	h0 := th + tu
	el := ((tu - (h0 - th)) + math.FMA(th, u, -tu)) + math.FMA(th, ue+pu, math.FMA(tl, u, tl))
	h1 := h0 + el
	el -= h1 - h0
	sc := pow2(ki >> 7)
	return h1 * sc, el * sc
}

// hurwitzFast is hurwitzPos for the common range, in float64 kernels:
// 1 < s <= 64, 2^-20 <= q <= 2^40 with every term and the tail in the
// normal range (s log2(max(q, w)) <= 1000, s log2(1/q) <= 1000). The same
// Euler–Maclaurin arrangement (DLMF 25.11.5; Cephes zeta.c) and the same
// Bernoulli terms with a shorter direct sum; each (q+k)^-s from powNegHL with q + k exact as
// hi + lo, the terms summed by TwoSum, the tail w^-s [w/(s-1) + 1/2 +
// Σ_j B_2j/(2j)! (s)_{2j-1} w^(1-2j)] with its first term as hi + lo.
func hurwitzFast(s, q float64) (float64, bool) {
	if !(s > 1 && s <= 64 && q >= 0x1p-20 && q <= 0x1p40) {
		return 0, false
	}
	// The shift w = q + N: 25 Bernoulli terms reach 2^-66 of w/(s-1) for
	// w >= 7.3 + s/2 (computed with mpmath for 1 < s <= 64: the bound
	// needed is 6.1 at s = 1, 7.4 at 1.5, 8.5 at 3, 12.0 at 10, 16.5 at
	// 20, 34.8 at 64), a much shorter direct sum than hurwitzPos's
	// 0.7(s+30) with 15 terms, which allows for s > 64 (the bound is
	// checked by TestHurwitzShiftBound).
	w0 := 7.3 + 0.5*s
	n := 0.0
	if q < w0 {
		n = math.Ceil(w0 - q)
	}
	_, ew := math.Frexp(q + n)
	_, eq := math.Frexp(q)
	if s*float64(ew) > 1000 || s*float64(1-eq) > 1000 {
		return 0, false
	}
	sh, sl := 0.0, 0.0
	first := 0.0
	for k := 0.0; k < n; k++ {
		yh := q + k // TwoSum
		bb := yh - q
		yl := (q - (yh - bb)) + (k - bb)
		th, tl := powNegHL(yh, yl, s)
		if k == 0 {
			first = th
		} else if th < 0x1p-80*first {
			// (q+k)^-s < 2^-80 q^-s: the remaining terms and the tail
			// (at most (q+k)/(s-1) times this term) cannot reach the
			// result's last bit.
			return sh + sl, true
		}
		v := sh + th // TwoSum
		bb = v - sh
		sl += ((sh - (v - bb)) + (th - bb)) + tl
		sh = v
	}
	wh := q + n
	bb := wh - q
	wl := (q - (wh - bb)) + (n - bb)
	ph, pl := powNegHL(wh, wl, s) // w^-s
	// bracket: w/(s-1) + 1/2 + Σ_j c_j (s)_{2j-1} w^(1-2j)
	dh := s - 1 // TwoSum
	bb = dh - s
	dl := (s - (dh - bb)) + (-1 - bb)
	bh := wh / dh
	bl := (math.FMA(-bh, dh, wh) + wl - bh*dl) / dh
	iw := 1 / wh
	iw2 := iw * iw
	// The first Bernoulli term t1 = s/(12w) apart, as hi + lo, the others (below
	// t1 s²/(30 w²)) summed in float64: summing them onto 1/2 would leave
	// the rounding of 1/2's ulp at every step.
	iwl := (math.FMA(-iw, wh, 1) - iw*wl) * iw // 1/w = iw + iwl
	sw := float64(s * iw)
	swl := math.FMA(s, iw, -sw) + s*iwl // s/w = sw + swl
	t1 := float64(sw * (1.0 / 12))
	t1l := math.FMA(sw, 1.0/12, -t1) + (swl*(1.0/12) + sw*4.625929269271485e-18) // 1/12 - RN(1/12)
	poch := s * (s + 1) * (s + 2)                                                // (s)_{2j-1}
	pw := iw * iw2                                                               // w^(1-2j)
	rest := 0.0
	for j := 2; j <= len(bernoulliOverFactorial25); j++ {
		t := bernoulliOverFactorial25[j-1] * poch * pw
		rest += t
		if math.Abs(t) < 0x1p-64*bh {
			break // converged (TestHurwitzShiftBound: always by j = 25)
		}
		kk := float64(2*j - 1)
		poch *= (s + kk) * (s + kk + 1)
		pw *= iw2
	}
	c1 := bh + 0.5 // Fast2Sum: w/(s-1) > 1
	e1 := (bh - c1) + 0.5
	ch := c1 + t1 // Fast2Sum: t1 < 0.2
	e2 := (c1 - ch) + t1
	cl := (e1 + e2) + (bl + (t1l + rest))
	// T = w^-s · bracket
	th := float64(ph * ch)
	tl := math.FMA(ph, ch, -th) + math.FMA(ph, cl, pl*ch)
	v := sh + th // TwoSum
	bb = v - sh
	ve := (sh - (v - bb)) + (th - bb)
	return v + (ve + (sl + tl)), true
}

// hurwitzAny returns ζ(s, q) as a scaled value for finite s > 1 and q not a
// non-positive integer, q < 0 only with integer s.
func hurwitzAny(s, q float64) scaled {
	if q > 0 {
		return hurwitzPos(s, dd.F(q))
	}
	// ζ(s, q) = Σ_{k=0}^{m-1} (q+k)^-s + ζ(s, q+m), m = ceil(-q): q+k is
	// exact for k < m when |q| >= 1/2 and, carried as a double-double,
	// in every case.
	mf := math.Ceil(-q)
	odd := math.Mod(s, 2) != 0
	terms := make([]scaled, 0, int(mf)+1)
	for k := 0.0; k < mf; k++ {
		b := sumDD(q, k)
		t := powNegDD(b, s)
		if odd {
			t.m = dd.Neg(t.m) // (q+k) < 0 raised to an odd power
		}
		terms = append(terms, t)
	}
	terms = append(terms, hurwitzPos(s, sumDD(q, mf)))
	return sumScaled(terms)
}

// Polygamma returns the polygamma function ψ⁽ⁿ⁾(x), the n-th derivative of
// the digamma function; Polygamma(0, x) = Digamma(x).
//
// Special values: NaN for n < 0, NaN or x = -Inf; ψ⁽ⁿ⁾(+Inf) = 0 (n >= 1);
// at the poles x = 0, -1, -2, ... the value is (-1)^(n+1) Inf.
//
// Algorithm: ψ⁽ⁿ⁾(x) = (-1)^(n+1) n! ζ(n+1, x) (DLMF 25.11.12), with ζ from
// the double-double Euler–Maclaurin summation and n! in double-double, the
// two combined as scaled values so that the product overflows only when
// the result does.
func Polygamma(n int, x float64) float64 {
	switch {
	case n < 0 || math.IsNaN(x) || math.IsInf(x, -1):
		return math.NaN()
	case n == 0:
		return Digamma(x)
	case math.IsInf(x, 1):
		return 0
	}
	if n == 1 && x >= 0x1p-500 {
		return trigamma(x)
	}
	sign := 1.0
	if n%2 == 0 {
		sign = -1
	}
	if x <= 0 && x == math.Trunc(x) {
		return sign * math.Inf(1)
	}
	if n == 1 && x < -10 {
		return trigammaReflect(x)
	}
	h := hurwitzAny(float64(n)+1, x)
	var f scaled
	if n <= 170 {
		// n! as a normalised mantissa so that the product below cannot
		// overflow before the scaling.
		fd := factorialsDD[n]
		_, e := math.Frexp(fd.Hi)
		f = scaled{dd.Ldexp(fd, -e), e}
	} else {
		m, e := dd.ExpMant(logGammaPos(float64(n) + 1))
		f = scaled{m, e}
	}
	m := dd.Mul(h.m, f.m)
	return sign * math.Ldexp(m.Float(), h.e+f.e)
}

// polygamma1 is Polygamma(1, x) (the scalar function behind TrigammaTo).
func polygamma1(x float64) float64 {
	if x >= 0x1p-500 && x <= math.MaxFloat64 {
		return trigamma(x)
	}
	return Polygamma(1, x)
}

// trigamma returns ψ'(x) = ζ(2, x) for 2^-500 <= x < +Inf (Polygamma's
// n = 1, specialised: the integer power needs neither a logarithm nor an
// exponential). From 10, the asymptotic series (DLMF 5.15.8)
// 1/x + 1/(2x²) + Σ B_2k/x^(2k+1) with its first two terms as hi + lo and
// twelve Bernoulli terms (truncation below 2^-63 relative at x = 10).
// Below 10, the recurrence ψ'(x) = Σ_{k<n} 1/(x+k)² + ψ'(x+n) with x + n
// in [10, 11): every x + k exact as hi + lo, every 1/(x+k)² as hi + lo,
// the sum by TwoSum, and ψ'(x+n) corrected to first order for the low
// part of x + n.
func trigamma(x float64) float64 {
	if x >= 10 {
		h, l := trigammaAsym(x)
		return h + l
	}
	n := math.Ceil(10 - x)
	sh, sl := 0.0, 0.0
	for k := n - 1; k >= 0; k-- {
		yh := x + k // TwoSum
		bb := yh - x
		yl := (x - (yh - bb)) + (k - bb)
		q := 1 / yh
		ql := (math.FMA(-q, yh, 1) - q*yl) * q // 1/(x+k) = q + ql
		th := float64(q * q)
		tl := math.FMA(q, q, -th) + 2*q*ql
		v := sh + th // TwoSum
		bb = v - sh
		sl += ((sh - (v - bb)) + (th - bb)) + tl
		sh = v
	}
	yh := x + n
	bb := yh - x
	yl := (x - (yh - bb)) + (n - bb)
	ah, al := trigammaAsym(yh)
	iy := 1 / yh
	al -= yl * iy * iy * (1 + iy*(1+0.5*iy)) // ψ''(y) yl, ψ''(y) = -(1/y² + 1/y³ + 1/(2y⁴)) + O(y^-6)
	v := sh + ah
	bb = v - sh
	ve := (sh - (v - bb)) + (ah - bb)
	return v + (ve + (sl + al))
}

// trigammaReflect returns ψ'(x) for non-integer x < -10 by the reflection
// formula ψ'(x) = π²/sin²(πx) - ψ'(1 - x) (DLMF 5.15.6), with sin(πx)
// reduced exactly (sinPiDD), the quotient in double-double, and ψ'(1-x)
// from the asymptotic series at 1 - x = yh + yl (> 11) corrected to first
// order in yl. The Hurwitz sum it replaces needs ceil(-x) terms, which
// does not terminate in practice for large |x|. No cancellation: the first
// term is at least π², the second below 0.1.
func trigammaReflect(x float64) float64 {
	sn := sinPiDD(x)
	q := dd.Div(dd.Mul(piDD, piDD), dd.Mul(sn, sn))
	yh := 1 - x // TwoSum
	bb := yh - 1
	yl := (1 - (yh - bb)) + (-x - bb)
	ah, al := trigammaAsym(yh)
	iy := 1 / yh
	al -= yl * iy * iy * (1 + iy*(1+0.5*iy)) // ψ''(y) yl
	v := q.Hi - ah                           // Fast2Sum: q > 9.8 > ψ'(1-x)
	return v + (((q.Hi - v) - ah) + (q.Lo - al))
}

// trigammaAsym returns ψ'(x) as hi + lo for x >= 10 (see trigamma).
func trigammaAsym(x float64) (hi, lo float64) {
	q := 1 / x
	ql := math.FMA(-q, x, 1) * q // 1/x = q + ql
	zh := float64(q * q)
	zl := math.FMA(q, q, -zh) + 2*q*ql // 1/x² = zh + zl
	z2 := zh * zh
	z4 := z2 * z2
	a := math.FMA(math.FMA(-1.0/30, zh, 1.0/42), z2, math.FMA(-1.0/30, zh, 1.0/6))
	b := math.FMA(math.FMA(-3617.0/510, zh, 7.0/6), z2, math.FMA(-691.0/2730, zh, 5.0/66))
	c := math.FMA(math.FMA(-236364091.0/2730, zh, 854513.0/138), z2, math.FMA(-174611.0/330, zh, 43867.0/798))
	ser := math.FMA(math.FMA(c, z4, b), z4, a) * zh * q
	h := 0.5 * zh
	hi = q + h // Fast2Sum: 1/(2x²) < 1/x
	return hi, ((q - hi) + h) + (ql + 0.5*zl + ser)
}
