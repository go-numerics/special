package special

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-numerics/special/dd"
)

// Equivalence lemmas for zeta.go survivors of `mutsweep -sites all`. A
// mutant that no test can kill computes the same values; each fact it relies
// on is pinned here so that the claim is a test, not a label.

// TestLemmaPolygamma1IsPolygamma (zeta.go:533): polygamma1 and Polygamma(1,
// .) return the same bits on every finite x >= 2^-500, because Polygamma's
// n == 1 branch (zeta.go:502) is `trigamma(x)` for exactly that domain. So
// the guard at :533 only skips the n/NaN/pole dispatch (a speed gain);
// weakening, tightening or inverting its bounds routes x through the other
// spelling of the same call.
func TestLemmaPolygamma1IsPolygamma(t *testing.T) {
	xs := []float64{0x1p-500, math.Nextafter(0x1p-500, 1), 0x1p-300, 1e-100, 0.1,
		0.5, 1, 1.5, 9.99, 10, 11, 100, 1e5, 1e100, 1e300, math.MaxFloat64}
	for _, x := range xs {
		a, b := polygamma1(x), Polygamma(1, x)
		if math.Float64bits(a) != math.Float64bits(b) {
			t.Errorf("polygamma1(%g) = %v, Polygamma(1, %g) = %v", x, a, x, b)
		}
	}
	// Outside the guard both spellings are the same call by construction.
	for _, x := range []float64{math.Inf(1), math.NaN(), 0, 0x1p-501, -1.5} {
		a, b := polygamma1(x), Polygamma(1, x)
		if math.Float64bits(a) != math.Float64bits(b) && !(math.IsNaN(a) && math.IsNaN(b)) {
			t.Errorf("polygamma1(%g) = %v, Polygamma(1, %g) = %v", x, a, x, b)
		}
	}
	// Reference values (DLMF 5.15.8: psi'(+Inf) = 0; pole of order 2 at 0).
	if v := polygamma1(math.Inf(1)); v != 0 {
		t.Errorf("polygamma1(+Inf) = %v, want 0", v)
	}
	if v := polygamma1(0x1p-520); !math.IsInf(v, 1) {
		t.Errorf("polygamma1(2^-520) = %v, want about 2^1040 = +Inf", v)
	}
	// trigamma itself is only defined on [2^-500, Inf): the guard (or
	// Polygamma's identical one) is what keeps Inf and tiny x out of it.
}

// TestSumScaledAllZero (zeta.go:84): a sum whose terms are all zero is the
// zero value, with exponent 0 (the MinInt32 sentinel must not leak out).
func TestSumScaledAllZero(t *testing.T) {
	if got := sumScaled(nil); got != (scaled{}) {
		t.Errorf("sumScaled(nil) = %+v, want zero", got)
	}
	if got := sumScaled([]scaled{{dd.DD{}, 5}, {dd.DD{}, -9}}); got != (scaled{}) {
		t.Errorf("sumScaled(zero terms) = %+v, want zero", got)
	}
}

// TestHurwitzPosOverflowFirstTerm (zeta.go:122): zeta(s, q) >= q^-s
// (DLMF 25.11.1, all terms positive); for q = 1/2 and s = 2e7 that is 2^(2e7),
// beyond float64, so the first term's +Inf is the answer (not the empty sum 0).
func TestHurwitzPosOverflowFirstTerm(t *testing.T) {
	got := hurwitzPos(2e7, dd.F(0.5))
	if !math.IsInf(got.m.Hi, 1) {
		t.Errorf("hurwitzPos(2e7, 0.5) = %+v, want +Inf mantissa", got)
	}
	// q = 2: q^-s = 2^-2e7 underflows; the sum is 0 (DLMF 25.11.1).
	if got := hurwitzPos(2e7, dd.F(2)); got.m.Hi != 0 {
		t.Errorf("hurwitzPos(2e7, 2) = %+v, want 0", got)
	}
}

// TestHurwitzFastDeclines (zeta.go:389): the fast kernel refuses arguments
// whose terms would leave the normal range (s*log2(max(q,w)) > 1000 or
// s*log2(1/q) > 1000) and accepts the boundary (equal to 1000), and what it
// accepts agrees with the double-double path hurwitzPos.
func TestHurwitzFastDeclines(t *testing.T) {
	for _, c := range []struct {
		s, q float64
		ok   bool
	}{
		{64, 0x1p40, false},  // s*ew = 64*41
		{64, 0x1p-20, false}, // s*(1-eq) = 64*20
		{25, 0x1p39, true},   // s*ew = 25*40 = 1000
		{26, 0x1p39, false},  // 26*40 = 1040
		{50, 0x1p-20, true},  // s*(1-eq) = 50*20 = 1000
		{51, 0x1p-20, false}, // 51*20 = 1020
	} {
		v, ok := hurwitzFast(c.s, c.q)
		if ok != c.ok {
			t.Errorf("hurwitzFast(%g, %g) ok = %v, want %v", c.s, c.q, ok, c.ok)
			continue
		}
		if !ok {
			continue
		}
		r := hurwitzPos(c.s, dd.F(c.q))
		want := math.Ldexp(r.m.Float(), r.e)
		if rel := math.Abs(v-want) / want; !(rel <= 0x1p-50) {
			t.Errorf("hurwitzFast(%g, %g) = %v, hurwitzPos %v (rel %g)", c.s, c.q, v, want, rel)
		}
	}
}

// TestLemmaFactorialTableMatchesLogGamma(zeta.go:517): for 1 <= n <= 170
// the table factorialsDD[n] is n! to double-double accuracy (math/big
// oracle), and the else branch exp(lgamma(n+1)) mantissa/exponent pair
// agrees with it to within the stated bound below, which is far inside
// Polygamma's own 1-ulp contract at the n! factor (tested against math/big in
// the accuracy suite). Taking either branch for n = 170 changes the result by
// no more than that bound.
func TestLemmaFactorialTableMatchesLogGamma(t *testing.T) {
	const bound = 0x1p-44
	worst := 0.0
	for n := 1; n <= 170; n++ {
		exact := new(big.Float).SetPrec(300).SetInt(new(big.Int).MulRange(1, int64(n)))
		fd := factorialsDD[n]
		tab := new(big.Float).SetPrec(300).SetFloat64(fd.Hi)
		tab.Add(tab, new(big.Float).SetPrec(300).SetFloat64(fd.Lo))
		diff := new(big.Float).SetPrec(300).Sub(tab, exact)
		diff.Quo(diff, exact)
		if d, _ := diff.Float64(); math.Abs(d) > 0x1p-100 {
			t.Errorf("factorialsDD[%d] rel err %g > 2^-100", n, d)
		}
		m, e := dd.ExpMant(logGammaPos(float64(n) + 1))
		lg := new(big.Float).SetPrec(300).SetFloat64(m.Hi)
		lg.Add(lg, new(big.Float).SetPrec(300).SetFloat64(m.Lo))
		lg.SetMantExp(lg, e)
		diff = new(big.Float).SetPrec(300).Sub(lg, exact)
		diff.Quo(diff, exact)
		d, _ := diff.Float64()
		worst = math.Max(worst, math.Abs(d))
		if math.Abs(d) > bound {
			t.Errorf("exp(logGammaPos(%d)) rel err %g > %g vs n! (big)", n+1, d, bound)
		}
	}
	t.Logf("worst relative error of the logGamma branch over n <= 170: %g", worst)
}
