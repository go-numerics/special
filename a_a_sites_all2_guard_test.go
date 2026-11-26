package special

import (
	"math"
	"math/big"
	"testing"
)

// Second chunk of `mutsweep -sites all` guards (go-numerics/special#97 left
// them as "equivalent"). Every reference is stated: a math/big integer
// product, bigLogGamma (Stirling's series at 2^-400, i_oracle_test.go), or the
// sign (-1)^ceil(-x) of Gamma between its poles (DLMF 5.5.4). The file sorts
// first so that a crash inside a mutant cannot hide the assertion.

// bigRelErr returns |got - want| / |want| for a math/big reference.
func bigRelErr(got float64, want *big.Float) float64 {
	d := new(big.Float).SetPrec(300).SetFloat64(got)
	d.Sub(d, want)
	d.Quo(d, want)
	f, _ := d.Float64()
	return math.Abs(f)
}

// TestRecipDDOverflow (digamma.go:28): 1/x overflows below 2^-1024, and the
// double-double answer is then exactly (±Inf, 0): a low part computed from the
// infinite quotient would be -Inf or NaN.
func TestRecipDDOverflow(t *testing.T) {
	for _, x := range []float64{5e-324, -5e-324, 0x1p-1030, -0x1p-1030, 0x1p-1024} {
		mustNotPanic(t, "recipDD", func() {
			r := recipDD(x)
			if !math.IsInf(r.Hi, 0) || math.Signbit(r.Hi) != math.Signbit(x) || r.Lo != 0 {
				t.Errorf("recipDD(%g) = {%v %v}, want {%v 0}", x, r.Hi, r.Lo, math.Copysign(math.Inf(1), x))
			}
		})
	}
}

// TestPermShortTail (comb.go:256): for k > 64 and n-k+1 < 10 the quotient
// n!/(n-k)! is a product of k factors; the reference is the math/big integer
// product. logGammaDiff, meant for n-k+1 >= 10, is 8 ulp off there
// (1.8e-15), so a bound of 2^-50 holds the branch condition.
func TestPermShortTail(t *testing.T) {
	for _, c := range [][2]int64{{100, 95}, {75, 70}, {90, 85}, {170, 100}, {170, 150}, {150, 140}, {200, 100}} {
		n, k := c[0], c[1]
		p := big.NewInt(1)
		for i := int64(0); i < k; i++ {
			p.Mul(p, big.NewInt(n-i))
		}
		want := new(big.Float).SetPrec(300).SetInt(p)
		mustNotPanic(t, "Perm", func() {
			if e := bigRelErr(Perm(n, k), want); e > 0x1p-50 {
				t.Errorf("Perm(%d, %d): relative error %.3g against the integer product, want <= 2^-50", n, k, e)
			}
		})
	}
}

// TestLogGammaBeyondOctTable (loggamma.go:72): lgammaOct's table ends near
// 1024 (it indexes out of range from about 1e4); above, Stirling's series in
// double-double answers. Reference: bigLogGamma.
func TestLogGammaBeyondOctTable(t *testing.T) {
	ivOracleInit()
	for _, x := range []float64{10, 10.5, 20.25, 100.125, 500.75, 1000.5, 1023.9, 1023.5, 1024, 1024.5, 2000.25, 1e4 + 0.5, 1e5 + 0.125, 1e7 + 0.5, 1e10 + 0.5, 1e15 + 0.5, 1e20, 1e100, 1e300, 2.5e305} {
		want := bigLogGamma(bnew().SetFloat64(x))
		mustNotPanic(t, "LogGamma", func() {
			v, s := LogGammaSign(x)
			if e := bigRelErr(v, want); e > 0x1p-52 || s != 1 {
				t.Errorf("LogGamma(%g): relative error %.3g sign %d, want <= 2^-52 and +1", x, e, s)
			}
		})
	}
}

// TestGammaDeepNegativeSign (gamma.go:335): between the poles below -190
// |Gamma| < 1e-338 underflows, and the zero carries the sign (-1)^ceil(-x)
// (reflection formula, DLMF 5.5.4; the parity is taken from the integer
// ceil(-x), not from gammaSignNeg).
func TestGammaDeepNegativeSign(t *testing.T) {
	for _, x := range []float64{-190.0000000000001, -190.5, -191.25, -200.5, -255.75, -1000.5, -12345.678, -1e6 - 0.5, -1e10 - 0.5, -1e15 - 0.5, -4503599627370495.5} {
		want := 1.0
		if int64(math.Ceil(-x))&1 == 1 {
			want = -1
		}
		mustNotPanic(t, "Gamma", func() {
			if got := Gamma(x); got != 0 || math.Signbit(got) != (want < 0) {
				t.Errorf("Gamma(%v) = %v, want a zero of sign %v", x, got, want)
			}
		})
	}
}

// TestXlogySubnormalRounded (logistic.go:296): Xlogy's documented contract is
// one rounding of x*log(y) (log as hi + lo). For subnormal y the log goes
// through logHL; x*Log(y) rounds twice and lands one ulp off about a quarter
// of the time. Reference: x * bigLog(y) from math/big, rounded once.
func TestXlogySubnormalRounded(t *testing.T) {
	ivOracleInit()
	for _, y := range []float64{5e-324, 1e-320, 3.3e-315, 2.2e-310, 1.1e-308, 0x1p-1050} {
		ly := bigLog(bnew().SetFloat64(y))
		for _, x := range []float64{1, 3, -7.3, 0.1, 123.456, 1e-5, 2.5e10, -0.37, 5.5, 9.75, 1.1, 2.9} {
			w := bnew().Mul(bnew().SetFloat64(x), ly)
			want, _ := w.Float64()
			mustNotPanic(t, "Xlogy", func() {
				if got := Xlogy(x, y); got != want {
					t.Errorf("Xlogy(%v, %v) = %v, want the correctly rounded %v", x, y, got, want)
				}
			})
		}
	}
}

// TestGammaMidAtOneTwo(gamma.go:256): Gamma(1) = Gamma(2) = 1 exactly (the
// n <= 0 branch, whose absence would evaluate x = 1 at Gamma(1+1) instead).
func TestGammaMidAtOneTwo(t *testing.T) {
	for _, x := range []float64{1, 2} {
		if h, l := gammaMid(x); h != 1 || l != 0 {
			t.Errorf("gammaMid(%v) = (%v, %v), want (1, 0)", x, h, l)
		}
	}
}

// TestNonFinitePremises pins what makes the early returns of the softmax
// family and Binom value-equivalent to their fall-through: with a non-finite
// maximum every fast evaluation already returns NaN (or the same infinity),
// and logGamma at a pole of Gamma is NaN. They hold the premise, so that
// changing the callee turns the guard into a real one visibly.
func TestNonFinitePremises(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	for _, c := range []struct {
		x []float64
		m float64
	}{{[]float64{1, nan}, nan}, {[]float64{nan}, nan}, {[]float64{1, inf}, inf}, {[]float64{inf}, inf}, {[]float64{-inf, -inf}, -inf}, {[]float64{-inf}, -inf}} {
		mustNotPanic(t, "fast path", func() {
			dst := make([]float64, len(c.x))
			softmaxFast(dst, c.x, c.m)
			for i, v := range dst {
				if !math.IsNaN(v) {
					t.Errorf("softmaxFast(%v, m=%v)[%d] = %v, want NaN", c.x, c.m, i, v)
				}
			}
			logSoftmaxFast(dst, c.x, c.m)
			for i, v := range dst {
				if !math.IsNaN(v) {
					t.Errorf("logSoftmaxFast(%v, m=%v)[%d] = %v, want NaN", c.x, c.m, i, v)
				}
			}
			if v, ok := logSumExpFast(c.x, c.m); ok && !sameFloat(v, c.m) {
				t.Errorf("logSumExpFast(%v, m=%v) = %v accepted, want it declined or %v", c.x, c.m, v, c.m)
			}
		})
	}
	for _, x := range []float64{0, -1, -2, -3, -40, -41, -1000, -1e15} {
		mustNotPanic(t, "logGammaAbsDD", func() {
			if v := logGammaAbsDD(x); !math.IsNaN(v.Hi) {
				t.Errorf("logGammaAbsDD(%v) = %v, want NaN at a pole of Gamma", x, v)
			}
		})
	}
}
