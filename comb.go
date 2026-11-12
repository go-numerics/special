package special

import (
	"math"
	"math/bits"

	"github.com/go-numerics/special/dd"
)

// Factorial returns n! = Γ(n+1). Integers 0..170 come from a table of
// correctly rounded values; other non-negative n use the gamma function
// (with n+1 never rounded: Γ(1+n) from the core for n < 1, n Γ(n) above).
//
// Special values: NaN for NaN; 0 for n < 0 (the convention of SciPy's
// factorial); +Inf for n > 170.6243769563027.
func Factorial(n float64) float64 {
	switch {
	case math.IsNaN(n):
		return n
	case n < 0:
		return 0
	case n == math.Trunc(n):
		if n <= 170 {
			return factorials[int(n)]
		}
		return math.Inf(1)
	case n < 1:
		gh, gl := gamma1pFast(n)
		if zivRound(gh, gl, float64(gammaErrRecip*gh)) {
			return gh
		}
		return gamma1p(n).Float()
	case n > gammaMax-1:
		return math.Inf(1)
	}
	// Γ(n+1) = exp(log n + log Γ(n))
	if n <= 170 {
		if v, ok := factorialLargeFast(n); ok {
			return v
		}
	}
	return dd.Exp(dd.Add(dd.Log(n), logGammaPos(n)))
}

// factorialErr bounds |h + l - D| / |h| of factorialLargeFast, D the
// reference's value: log Γ(n) from logGammaLargeHL (absolute error about
// 2^-61 up to 171.6), log n from the log kernel (2^-68) and the exponential
// kernel (2^-68), against the reference's double-double evaluation, whose
// absolute error in the exponent is its logarithm's 2^-72.9 relative times
// (n - 1/2) log n < 2^10, below 2^-62.5. Their sum is under 2^-60; 2^-59 is
// used.
//
// The sweep of factorial_small_test.go (2·10^7 points) found that 2^-59 is
// not enough just above 10: the distance reaches 2^-58.79 on [10, 10.6]
// (2^-59.3 up to 14, 2^-60.6 from 30), where the reference's stirlingSeries
// (truncated to 2^-60 of log Γ, a float64 sum) and the fast kernel differ by
// their roundings. A bound shrinking with n was refuted too (n = 16.43): the
// distance is not proportional to the exponent. The bound is therefore
// piecewise (factorialBound): 2^-57.5 on [10, 12), 2^-58.5 on [12, 30), 2^-59
// from 30, each at least 1.6 times the largest distance measured on its piece.
//
// The derivation the old sum left out is the reference's float64
// stirlingSeries (gamma.go): its error in the exponent, in units of 2^-60,
// is the final division (up to 1), the numerator sum (up to 0.8) and the bias
// of the rounded c1 = 1/12 (0.53): at most 2.55 at n = 10 (0.45 of 2^-57.5)
// and 1.7 at n = 12 (0.59 of 2^-58.5, the thinnest margin; measured 0.569,
// the test gate is 0.62). The released 2^-59 accepted 26 of about 3.9·10^6
// points in [10, 10.67] whose value differs from the reference's by one ulp
// (#67); with this bound they fall back to the reference.
// For 1 < n < 10 (logGammaMidHL, the product Γ(z) Π (n-j) carried as hi + lo,
// error below 2^-100) the exponent is at most log 9! = 12.8: two log kernels
// (2^-68 each), the exponential (2^-68) and the reference's logarithm
// (2^-72.9 · 12.8) sum to 2^-66, the bound 2^-63 leaves 8 times, and the
// sweep measures 2^-68.8.
const factorialErr = 0x1p-59

// factorialBound returns the bound of |h + l - D| for the value h at n.
func factorialBound(n, h float64) float64 {
	e := factorialErr
	switch {
	case n < 10:
		e = 0x1p-63
	case n < 12:
		e = 0x1.6a09e667f3bcdp-58 // 2^-57.5
	case n < 30:
		e = 0x1.6a09e667f3bcdp-59 // 2^-58.5
	}
	return float64(e * h)
}

// factorialLargeFast returns n! = exp(log Γ(n) + log n) for 1 < n <= 170
// from the float64 kernels as an unevaluated sum, and whether it is certified
// equal to the double-double reference's rounding (the Ziv test of
// gamma_fast.go).
func factorialLargeFast(n float64) (float64, bool) {
	h, l := factorialLargeHL(n)
	if !zivRound(h, l, factorialBound(n, h)) {
		return 0, false
	}
	return h, true
}

// factorialLargeHL is the unevaluated sum h + l of factorialLargeFast.
func factorialLargeHL(n float64) (h, l float64) {
	var gh, gl, lh, ll float64
	if n < 10 {
		gh, gl = logGammaMidHL(n) // 1 < n < 10: Binom's kernel, the reference's own product
		lh, ll = logHLn(n)
	} else {
		gh, gl, lh, ll = logGammaLargeHLL(n)
	}
	s := gh + lh // TwoSum (the two terms have either sign below 2)
	bb := s - gh
	e := (gh - (s - bb)) + (lh - bb)
	lo := e + (gl + ll)
	hn := s + lo
	eh, el, k := expHL(hn, lo-(hn-s))
	p := pow2(k) // k <= 1023
	return float64(eh * p), float64(el * p)
}

// LogFactorial returns log(n!) = log Γ(n+1) for n >= 0 without forming
// n+1 (which would round for non-integer n). NaN for n < 0 or NaN; +Inf for
// n = +Inf.
func LogFactorial(n float64) float64 {
	switch {
	case math.IsNaN(n) || n < 0:
		return math.NaN()
	case math.IsInf(n, 1):
		return n
	case n == 0:
		return 0
	case n < 1:
		return log1pDD(gamma1pm1(n)).Float()
	}
	return dd.Add(dd.Log(n), logGammaPos(n)).Float()
}

// Factorial2 returns the double factorial n!! = n (n-2) (n-4) ..., ending
// at 1 or 2. 0!! = (-1)!! = 1 by the usual convention (SciPy's factorial2
// returns 0 for -1); n < -1 gives 0, as in SciPy. The product is carried in
// double-double and rounded once, so the result is correctly rounded (up to
// the 2^-100 double-double error) and exact whenever n!! < 2^53; it
// overflows to +Inf for n > 300.
func Factorial2(n int) float64 {
	switch {
	case n < -1:
		return 0
	case n <= 0:
		return 1
	case n > 300:
		return math.Inf(1)
	}
	return factorials2[n]
}

// factorials2[n] = n!! (n = 0..300), each the double-double product of
// factorial2DD rounded once, filled once at start-up like factorials.
var factorials2 [301]float64

func init() {
	for n := range factorials2 {
		factorials2[n] = factorial2DD(n)
	}
}

// factorial2DD is the product n (n-2) ... in double-double, rounded once.
func factorial2DD(n int) float64 {
	p := dd.F(1)
	for k := n; k > 1; k -= 2 {
		p = dd.MulF(p, float64(k))
	}
	return p.Float()
}

// Comb returns the binomial coefficient C(n, k) = n! / (k! (n-k)!), the
// number of k-element subsets of an n-set, as a float64: 0 when k < 0,
// k > n or n < 0 (as SciPy's comb with exact=False).
//
// Algorithm: for min(k, n-k) <= 64 the product Π (n-k+i)/i in
// double-double, so every result below 2^53 is exact and larger ones are
// correctly rounded; beyond, 1/((n+1) B(k+1, n-k+1)) with the beta function's
// cancellation-free double-double logarithm (LogBeta), which avoids the
// catastrophic cancellation of log n! - log k! - log (n-k)! for large n.
func Comb(n, k int64) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	if k <= 64 {
		if c, ok := combExact(uint64(n-k), uint64(k)); ok {
			return float64(c)
		}
		p := dd.F(1)
		nk := float64(n - k)
		for i := int64(1); i <= k; i++ {
			p = dd.DivF(p, float64(i))
			if p.Hi > math.MaxFloat64/(nk+float64(i)) {
				// The product overflows; the remaining factors
				// (n-k+i)/i are all >= 1, so the result does too.
				return math.Inf(1)
			}
			p = dd.MulF(p, nk+float64(i))
		}
		return p.Float()
	}
	fn, fk := float64(n), float64(k)
	l := dd.Add(dd.Log(fn+1), logBetaPos(fk+1, fn-fk+1))
	return dd.Exp(dd.Neg(l))
}

// combExact returns C(m+k, k) as an integer by the exact recurrence
// C(m+i, i) = C(m+i-1, i-1) (m+i) / i (each step is an exact division, the
// 128-bit product divided by 64-bit division), or ok = false as soon as a
// partial value would not fit in 64 bits. A value below 2^64 converted to
// float64 is rounded to nearest, which is also the double-double product's
// rounding (that one is exact below 2^106), so the two agree bit for bit.
func combExact(m, k uint64) (c uint64, ok bool) {
	c = 1
	for i := uint64(1); i <= k; i++ {
		hi, lo := bits.Mul64(c, m+i)
		if hi >= i {
			return 0, false
		}
		c, _ = bits.Div64(hi, lo, i)
	}
	return c, true
}

// Perm returns the number of k-permutations of n, n!/(n-k)! =
// n (n-1) ... (n-k+1), as a float64: 0 when k < 0, k > n or n < 0 (as
// SciPy's perm with exact=False).
//
// Algorithm: the product in double-double for k <= 64 (exact below 2^53,
// correctly rounded above); beyond, exp(log Γ(n+1) - log Γ(n-k+1)) with
// the difference formed without cancellation (logGammaDiff) when n-k+1 is
// at least 10 and k <= n-k+1.
func Perm(n, k int64) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if k <= 64 {
		p := dd.F(1)
		for i := int64(0); i < k; i++ {
			if p.Hi > math.MaxFloat64/float64(n-i) {
				return math.Inf(1) // every remaining factor is >= 1
			}
			p = dd.MulF(p, float64(n-i))
		}
		return p.Float()
	}
	fn, fk := float64(n), float64(k)
	b := fn - fk + 1
	if b >= 10 && fk <= b {
		return dd.Exp(dd.Neg(logGammaDiff(fk, b)))
	}
	return dd.Exp(dd.Sub(logGammaPos(fn+1), logGammaPos(b)))
}

// Binom returns the generalised binomial coefficient
// Γ(n+1) / (Γ(k+1) Γ(n-k+1)) for real n and k, as SciPy's binom.
//
// For integer k with 0 <= k <= 64 it is the product Π (n-k+i)/i in
// double-double (any real n, including negative); when n+1 > 0 and
// n-k+1 > 0 it is 1/((n+1) B(n-k+1, k+1)) with the beta function's
// cancellation-free evaluation; otherwise the logarithms of the three gamma
// functions (each in double-double, at double-double arguments) are combined
// before a single exponential, with their signs. It is 0 where 1/Γ(k+1) or 1/Γ(n-k+1)
// vanishes (a non-positive integer argument), NaN for NaN arguments or a
// pole of Γ(n+1) not cancelled by the denominator.
func Binom(n, k float64) float64 {
	switch {
	case math.IsNaN(n) || math.IsNaN(k) || math.IsInf(n, 0) || math.IsInf(k, 0):
		return math.NaN()
	case k == math.Trunc(k) && k >= 0 && k <= 64:
		p := dd.F(1)
		for i := 1.0; i <= k; i++ {
			// n - k + i, carried exactly
			t := dd.AddF(sumDD(n, -k), i)
			p = dd.DivF(dd.Mul(p, t), i)
		}
		return p.Float()
	}
	np1 := sumDD(n, 1)
	nk1 := dd.AddF(sumDD(n, -k), 1)
	if np1.Hi > 0 && nk1.Hi > 0 && k > -1 {
		if v, ok := binomFast(np1, nk1, k); ok {
			return v
		}
		return binomBetaDD(np1, nk1, k)
	}
	k1 := sumDD(k, 1)
	isPole := func(x dd.DD) bool { return x.Lo == 0 && x.Hi <= 0 && x.Hi == math.Trunc(x.Hi) }
	if isPole(k1) || isPole(nk1) {
		return 0
	}
	if isPole(np1) {
		return math.NaN()
	}
	// log|Γ| of each double-double argument, corrected to first order for
	// the low part (∂ log|Γ(x)| = ψ(x)), combined before one exponential.
	lg := func(x dd.DD) dd.DD {
		return dd.AddF(logGammaAbsDD(x.Hi), float64(Digamma(x.Hi)*x.Lo))
	}
	l := dd.Sub(lg(np1), dd.Add(lg(k1), lg(nk1)))
	sign := float64(GammaSign(np1.Hi) * GammaSign(k1.Hi) * GammaSign(nk1.Hi))
	return float64(sign * dd.Exp(l))
}

// binomBetaDD is Binom's reference evaluation for n+1 > 0, n-k+1 > 0 and
// k > -1: log B(a, b) at the rounded a = n-k+1 and b = k+1, corrected to
// first order for their rounding (∂/∂a log B = ψ(a) - ψ(a+b)), then
// 1/((n+1) B(a, b)) by one double-double exponential.
func binomBetaDD(np1, nk1 dd.DD, k float64) float64 {
	k1 := sumDD(k, 1)
	a, b := nk1.Float(), k1.Float()
	da, db := dd.AddF(nk1, -a).Float(), dd.AddF(k1, -b).Float()
	lb := logBetaPos(math.Min(a, b), math.Max(a, b))
	ps := Digamma(a + b)
	lb = dd.AddF(lb, float64(da*(Digamma(a)-ps))+float64(db*(Digamma(b)-ps)))
	return dd.Exp(dd.Neg(dd.Add(logDD(np1), lb)))
}
