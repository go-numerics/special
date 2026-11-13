package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Kn's fast path: the upward recurrence on the scaled functions
//
//	Ke_{k+1}(x) = Ke_{k-1}(x) + (2k/x) Ke_k(x),   Ke_k = K_k e^x
//
// (DLMF 10.29.1), started from K0e and K1e as hi + lo pairs and carried in a
// "dd-light" form: the hi chain with the exact error of each product and sum
// (TwoProd, TwoSum), the lo chain linear in the lo parts. The result is then
// multiplied by e^-x from dd.ExpMant and rounded once.
//
// Why a recurrence is safe here, where J and Y need a certified test: for
// x > 0 every K_k > 0 and K_{k+1} > K_k, so each step adds two positive
// terms. No cancellation can occur, and an error in the start values cannot
// be amplified.
//
// Error derivation. Write R_k for the relative error of the pair (h_k, l_k)
// against the true Ke_k.
//
//	start: the pair is c11 + tail with the TwoSum remainder kept, so the only
//	  start errors are the fit of the octave interpolant (2^-60 relative on
//	  both tables, see zz_besselK0eOct_coeffs.go) and the rounding of the tail
//	  (at most half a piece of variation, 1/64 of the value: 2^-53/64 =
//	  2^-59 relative). R_0, R_1 <= 2^-58.
//	step: Ke_{k+1} = Ke_{k-1} + c Ke_k with c = 2k/x > 0, so the propagated
//	  error is a convex combination of the two inputs' errors:
//	  R_{k+1} <= max(R_{k-1}, R_k) + r, where r is what the step adds. The
//	  step forms ch = RN(k qh) with its exact remainder (TwoProd), qh + ql =
//	  2/x to 2^-106 (one FMA remainder), p + pe = ch h_k exactly, s + se =
//	  h_{k-1} + p exactly, and lo = l_{k-1} + pe + se + ch l_k + cl h_k in
//	  float64. What is lost is cl l_k (2^-106 relative) and the roundings of
//	  the lo sum: lo is itself at most about k 2^-53 of the value, so each of
//	  the five additions loses at most 2^-53 of that, i.e. r <= 6 k 2^-106
//	  (k <= knFastMaxN = 1024: 2^-91).
//	n steps: R_n <= 2^-58 + n 2^-91 <= 2^-57.
//	end: the scale e^-x = (m.Hi + m.Lo) 2^k has 2^-76 relative error (dd.ExpMant),
//	  the product p + pe is exact, and the one rounding of
//	  p + (pe + m.Hi l + m.Lo h) is at most 0.5 ulp; the three small terms
//	  are rounded at 2^-53 of 2^-52 of the value.
//
// Total: 0.5 ulp + 2^-57 (about 0.03 ulp) + 2^-76: below 0.55 ulp against the
// exact value. Measured on the rows of the "kn" fixture (mpmath) that this
// path answers: 0.518 ulp worst (TestKnFastFixture bounds it at 0.56);
// Steed's method in double-double had 2.95 ulp on the same fixture. With the
// lo chain removed, the same rows reach 297 ulp (large order) and 11 ulp
// (moderate): the roundings of the hi chain add up, and the test is red.
//
// Range: x in [1, 512) (the octave tables of K0e and K1e; below 1, K0 and
// K1 are float64 rational forms, accurate to 1-2 ulp, so the start error
// would reach the result; from 512 they are asymptotic series) and
// 2 <= |n| <= knFastMaxN; the result must come out finite and normal,
// otherwise knRecur reports false and Kv runs (overflow of Ke_n for large n
// and x near 1).
const (
	knFastMaxN = 1024
	knFastMaxX = 512 // exclusive
)

// knStart returns K0e(x) and K1e(x) as hi + lo pairs for x in [1, 512):
// the constant term of the piece plus its tail, the rounding of the sum kept.
func knStart(i int, x float64) (h0, l0, h1, l1 float64) {
	q0 := &besselK0eOct[i]
	h0, l0 = dd.TwoSum(q0.coef[11], octTail(q0, x-q0.c))
	q1 := &besselK1eOct[i]
	h1, l1 = dd.TwoSum(q1.coef[11], octTail(q1, x-q1.c))
	return
}

// knRecurScaled returns Ke_n(x) as a pair for 2 <= n <= knFastMaxN and x in
// [1, 512), through knStart and the dd-light recurrence.
func knRecurScaled(n int, x float64) (h, l float64) {
	i := octIndex(x, besselK0eOctMin, besselK0eOctMax)
	h0, l0, h1, l1 := knStart(i, x)
	// 2/x = qh + ql: the remainder of the division is one FMA.
	qh := 2 / x
	ql := math.FMA(-qh, x, 2) / x
	for k := 1; k < n; k++ {
		kf := float64(k)
		ch, ce := dd.TwoProd(kf, qh)
		cl := ce + float64(kf*ql)
		p, pe := dd.TwoProd(ch, h1)
		s, se := dd.TwoSum(h0, p)
		lo := (l0 + pe + se) + math.FMA(ch, l1, float64(cl*h1))
		h0, l0, h1, l1 = h1, l1, s, lo
	}
	return h1, l1
}

// knRecur computes Kn(n, x) by the recurrence above; ok is false where the
// fast path does not apply and Kv must answer.
func knRecur(n int, x float64) (r float64, ok bool) {
	if n < 0 {
		n = -n
	}
	if n < 2 || n > knFastMaxN || !(x >= 1 && x < knFastMaxX) {
		return 0, false
	}
	h, l := knRecurScaled(n, x)
	if math.IsInf(h, 0) { // Ke_n overflows: Kv decides whether K_n does
		return 0, false
	}
	// e^-x Ke_n is neither above Ke_n nor below e^-512 Ke_0 > 1e-224: the
	// Ldexp is exact up to the one rounding.
	m, k := dd.ExpMant(dd.DD{Hi: -x})
	p, pe := dd.TwoProd(m.Hi, h)
	return math.Ldexp(p+(pe+float64(m.Hi*l)+float64(m.Lo*h)), k), true
}
