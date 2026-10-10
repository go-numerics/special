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
	terms := make([]scaled, 0, int(n)+1)
	for k := 0.0; k < n; k++ {
		t := powNegDD(dd.AddF(q, k), s)
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
// Algorithm: on [-1, 2], ζ(s) = 1/(s-1) + R(s) with R entire and fitted to
// mpmath (zz_zetaPieces_coeffs.go), so the pole is exact; for s > 2,
// 1 + ζ(s, 2) by Euler–Maclaurin (hurwitzPos); for s < -1 the reflection
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
	case s > 2:
		return zetaAbove2(s).Float()
	case s >= -1:
		h, l := dd.TwoSum(s, -1)
		r := dd.Div(dd.F(1), dd.DD{Hi: h, Lo: l})
		return dd.Add(r, evalPiecesDD(zetaPieces[:], s)).Float()
	}
	return zetaReflect(s)
}

// zetaAbove2 returns ζ(s) = 1 + ζ(s, 2) for s > 2 as a double-double.
func zetaAbove2(s float64) dd.DD {
	if s >= 64 {
		return dd.AddF(dd.F(1), math.Exp2(-s))
	}
	h := hurwitzPos(s, dd.F(2))
	return dd.AddF(dd.Ldexp(h.m, h.e), 1)
}

// zetaReflect returns ζ(s) for s < -1 by the reflection formula.
func zetaReflect(s float64) float64 {
	sn := sinPi(s / 2) // s/2 exact
	if sn == 0 {
		return 0 // trivial zero
	}
	t := -s // exact; Γ(1-s) = Γ(1+t) = t Γ(t)
	// L = s log 2 + (s-1) log π + log t + log Γ(t)
	l := dd.Add(dd.MulF(ln2DD, s), dd.Sub(dd.MulF(lnPi, s), lnPi))
	l = dd.Add(l, dd.Log(t))
	l = dd.Add(l, logGammaPos(t))
	m, e := dd.ExpMant(l)
	if math.IsInf(m.Hi, 0) {
		return math.Copysign(math.Inf(1), sn)
	}
	// ζ(1-s) = ζ(1+t); 1+t rounds, which ζ'(u)/ζ(u) ~ -2^-u ln 2 (u > 3)
	// makes harmless.
	z := zetaAbove2(1 + t)
	return math.Ldexp(dd.Mul(dd.MulF(m, sn), z).Float(), e)
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
	h := hurwitzAny(s, q)
	return math.Ldexp(h.m.Float(), h.e)
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
	sign := 1.0
	if n%2 == 0 {
		sign = -1
	}
	if x <= 0 && x == math.Trunc(x) {
		return sign * math.Inf(1)
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
