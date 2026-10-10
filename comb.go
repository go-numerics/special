package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
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
		return gamma1p(n).Float()
	case n > gammaMax:
		return math.Inf(1)
	}
	// Γ(n+1) = exp(log n + log Γ(n))
	return dd.Exp(dd.Add(dd.Log(n), logGammaPos(n)))
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
func Comb(n, k int) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if k > n-k {
		k = n - k
	}
	if k <= 64 {
		p := dd.F(1)
		nk := float64(n - k)
		for i := 1; i <= k; i++ {
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

// Perm returns the number of k-permutations of n, n!/(n-k)! =
// n (n-1) ... (n-k+1), as a float64: 0 when k < 0, k > n or n < 0 (as
// SciPy's perm with exact=False).
//
// Algorithm: the product in double-double for k <= 64 (exact below 2^53,
// correctly rounded above); beyond, exp(log Γ(n+1) - log Γ(n-k+1)) with
// the difference formed without cancellation (logGammaDiff) when n-k+1 is
// at least 10 and k <= n-k+1.
func Perm(n, k int) float64 {
	if k < 0 || n < 0 || k > n {
		return 0
	}
	if k <= 64 {
		p := dd.F(1)
		for i := 0; i < k; i++ {
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
		// log B(a, b) at the rounded a = n-k+1 and b = k+1, corrected to
		// first order for their rounding: ∂/∂a log B = ψ(a) - ψ(a+b).
		k1 := sumDD(k, 1)
		a, b := nk1.Float(), k1.Float()
		da, db := dd.AddF(nk1, -a).Float(), dd.AddF(k1, -b).Float()
		lb := logBetaPos(math.Min(a, b), math.Max(a, b))
		ps := Digamma(a + b)
		lb = dd.AddF(lb, da*(Digamma(a)-ps)+db*(Digamma(b)-ps))
		return dd.Exp(dd.Neg(dd.Add(logDD(np1), lb)))
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
		return dd.AddF(logGammaAbsDD(x.Hi), Digamma(x.Hi)*x.Lo)
	}
	l := dd.Sub(lg(np1), dd.Add(lg(k1), lg(nk1)))
	sign := GammaSign(np1.Hi) * GammaSign(k1.Hi) * GammaSign(nk1.Hi)
	return sign * dd.Exp(l)
}
