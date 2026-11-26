package special

import (
	"math"
	"math/big"
	"testing"
)

// Guards that `mutsweep -sites all` found unheld (survived or caught only by a
// panic). Every expected value is a stated reference: an exact integer or
// logarithm computed with math/big, a documented special value, or a
// tabulated constant (DLMF 10.21, A&S 6.1, A&S 9.5), never the code's own
// output.

// sitesLn2Str is ln 2 to 60 digits.
const sitesLn2Str = "0.693147180559945309417232121458176568075500134360255254120680009"

// minusKLn2 returns -k ln 2 rounded to float64 (math/big, 300 bits).
func minusKLn2(k int64) float64 {
	l, _, err := big.ParseFloat(sitesLn2Str, 10, 300, big.ToNearestEven)
	if err != nil {
		panic(err)
	}
	l.Mul(l, new(big.Float).SetPrec(300).SetInt64(-k))
	f, _ := l.Float64()
	return f
}

// mustNotPanic fails the test, instead of letting the panic end the binary,
// when f panics.
func mustNotPanic(t *testing.T, name string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s crashed instead of answering: %v", name, r)
		}
	}()
	f()
}

func sameFloat(a, b float64) bool {
	return a == b && math.Signbit(a) == math.Signbit(b) || math.IsNaN(a) && math.IsNaN(b)
}

// TestPermExactSmallK: for k <= 64 Perm is the product, exact below 2^53 and
// correctly rounded above; the reference is the math/big integer product.
func TestPermExactSmallK(t *testing.T) {
	cases := [][2]int64{{10, 3}, {20, 10}, {100, 5}, {30, 30}, {60, 30}, {1000, 64}, {200, 64}, {100, 64}, {64, 64}, {12, 0}, {40, 25}}
	for _, c := range cases {
		n, k := c[0], c[1]
		p := big.NewInt(1)
		for i := int64(0); i < k; i++ {
			p.Mul(p, big.NewInt(n-i))
		}
		want, _ := new(big.Float).SetInt(p).Float64()
		if got := Perm(n, k); got != want {
			t.Errorf("Perm(%d,%d) = %v, want %v", n, k, got, want)
		}
	}
}

// TestBinomPoleOfNumerator: Gamma(n+1) has a pole at n = -1, -2, ... that the
// denominator does not cancel for non-integer k: the documented result is NaN.
func TestBinomPoleOfNumerator(t *testing.T) {
	for _, n := range []float64{-1, -2, -7} {
		if got := Binom(n, 0.5); !math.IsNaN(got) {
			t.Errorf("Binom(%v, 0.5) = %v, want NaN", n, got)
		}
	}
}

func TestDigammaSpecialValues(t *testing.T) {
	if got := Digamma(5e-324); !math.IsInf(got, -1) {
		t.Errorf("Digamma(5e-324) = %v, want -Inf (-1/x overflows)", got)
	}
	if got := Digamma(math.Copysign(0, -1)); !math.IsInf(got, 1) {
		t.Errorf("Digamma(-0) = %v, want +Inf", got)
	}
	for _, x := range []float64{-1, -2, -3, -40, -41, -1e15} {
		if got := Digamma(x); !math.IsNaN(got) {
			t.Errorf("Digamma(%v) = %v, want NaN (pole)", x, got)
		}
	}
	// psi(1e18) = ln x - 1/(2x) - ... = ln(1e18) to double precision.
	if got, want := Digamma(1e18), 41.446531673892822; math.Abs(got-want) > 4e-15*want {
		t.Errorf("Digamma(1e18) = %v, want %v", got, want)
	}
}

// TestLogGammaSignNegatives: lgamma(-1/2) = ln(2 sqrt(pi)), Gamma(-1/2) < 0;
// lgamma(-3/2) = ln(4 sqrt(pi)/3), Gamma(-3/2) > 0; at -0 the sign is -1
// (documented, as math.Lgamma).
func TestLogGammaSignNegatives(t *testing.T) {
	cases := []struct {
		x    float64
		lg   float64
		sign int
	}{
		{-0.5, 1.2655121234846454, -1},
		{-1.5, 0.8600470153764810, 1},
		{-2.5, -0.056243716497674054, -1},
		{math.Copysign(0, -1), math.Inf(1), -1},
		{0, math.Inf(1), 1},
	}
	for _, c := range cases {
		lg, s := LogGammaSign(c.x)
		if s != c.sign || (lg != c.lg && !(math.Abs(lg-c.lg) <= 4e-16*math.Abs(c.lg))) {
			t.Errorf("LogGammaSign(%v) = (%v, %d), want (%v, %d)", c.x, lg, s, c.lg, c.sign)
		}
	}
}

// TestGammaDeepNegativeSignedZero: |Gamma(x)| < 1e-350 for x < -190: the result
// is a zero with the sign of Gamma, (-1)^ceil(-x).
func TestGammaDeepNegativeSignedZero(t *testing.T) {
	if g := Gamma(-200.5); g != 0 || !math.Signbit(g) {
		t.Errorf("Gamma(-200.5) = %v, want -0", g)
	}
	if g := Gamma(-201.5); g != 0 || math.Signbit(g) {
		t.Errorf("Gamma(-201.5) = %v, want +0", g)
	}
}

// TestJnZerosOrderAndCount: J_{-n} = (-1)^n J_n has the zeros of J_n; zero or
// negative nt gives an empty result. Reference zeros: DLMF 10.21(iv) / A&S
// table 9.5.
func TestJnZerosOrderAndCount(t *testing.T) {
	for _, nt := range []int{0, -1, -100} {
		mustNotPanic(t, "JnZeros nt<=0", func() {
			if got := JnZeros(0, nt); got == nil || len(got) != 0 {
				t.Errorf("JnZeros(0, %d) = %v, want empty non-nil", nt, got)
			}
		})
	}
	want := []float64{3.8317059702075125, 7.015586669815619}
	for _, n := range []int{1, -1} {
		got := JnZeros(n, 2)
		for i := range want {
			if math.Abs(got[i]-want[i]) > 4e-15*want[i] {
				t.Errorf("JnZeros(%d,2)[%d] = %v, want %v", n, i, got[i], want[i])
			}
		}
	}
	want0 := []float64{2.404825557695773, 5.520078110286311, 8.653727912911012}
	got0 := JnZeros(0, 3)
	for i := range want0 {
		if math.Abs(got0[i]-want0[i]) > 4e-15*want0[i] {
			t.Errorf("JnZeros(0,3)[%d] = %v, want %v", i, got0[i], want0[i])
		}
	}
}

// TestSubnormalLogs: log of a subnormal is exact to the scaling by 2^54:
// ln(2^-1050) = -1050 ln 2, for logf, Xlogy and Xlog1py's relatives.
func TestSubnormalLogs(t *testing.T) {
	y := 0x1p-1050
	want := minusKLn2(1050)
	if got := logf(y); math.Abs(got-want) > 2e-16*math.Abs(want) {
		t.Errorf("logf(2^-1050) = %v, want %v", got, want)
	}
	if got, w := Xlogy(3, y), minusKLn2(3150); math.Abs(got-w) > 2e-16*math.Abs(w) {
		t.Errorf("Xlogy(3, 2^-1050) = %v, want %v", got, w)
	}
	if got, w := logf(5e-324), minusKLn2(1074); math.Abs(got-w) > 2e-16*math.Abs(w) {
		t.Errorf("logf(5e-324) = %v, want %v", got, w)
	}
	// x = 0 gives 0 unless y is NaN; y < 0 and NaN give NaN.
	if got := Xlogy(0, 5e-324); got != 0 {
		t.Errorf("Xlogy(0, 5e-324) = %v, want 0", got)
	}
	for _, c := range [][2]float64{{1, -1}, {math.NaN(), 2}, {2, math.NaN()}, {0, math.NaN()}} {
		if got := Xlogy(c[0], c[1]); !math.IsNaN(got) {
			t.Errorf("Xlogy(%v, %v) = %v, want NaN", c[0], c[1], got)
		}
	}
}

// TestY1TinyArgument: Y1(x) = -2/(pi x) + O(x log x) (A&S 9.1.9); at x below
// 2^-1000 the leading term decides.
func TestY1TinyArgument(t *testing.T) {
	for _, x := range []float64{1e-300, 1e-305, 0x1p-1001} {
		pi, _, _ := big.ParseFloat("3.14159265358979323846264338327950288419716939937510582097494459", 10, 300, big.ToNearestEven)
		w := new(big.Float).SetPrec(300).Quo(big.NewFloat(-2).SetPrec(300), new(big.Float).SetPrec(300).Mul(pi, new(big.Float).SetPrec(300).SetFloat64(x)))
		want, _ := w.Float64()
		if got := Y1(x); math.Abs(got-want) > 2e-16*math.Abs(want) {
			t.Errorf("Y1(%v) = %v, want %v", x, got, want)
		}
	}
}

// TestSinCosPiNonFinite: sinPi and cosPi of NaN and Inf are NaN (documented).
func TestSinCosPiNonFinite(t *testing.T) {
	for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		// Without poly.go:63/94 the Inf reaches a float-to-int conversion
		// whose result is architecture-defined: a crash on amd64.
		mustNotPanic(t, "sinPi", func() {
			if got := sinPi(x); !math.IsNaN(got) {
				t.Errorf("sinPi(%v) = %v, want NaN", x, got)
			}
		})
		mustNotPanic(t, "cosPi", func() {
			if got := cosPi(x); !math.IsNaN(got) {
				t.Errorf("cosPi(%v) = %v, want NaN", x, got)
			}
		})
	}
}

// TestSoftmaxFamilyNonFinite: an empty x gives an empty result without a
// panic; NaN or +Inf, or only -Inf, give NaN everywhere (SciPy's softmax);
// -Inf entries among finite ones are exact zeros.
func TestSoftmaxFamilyNonFinite(t *testing.T) {
	type fn struct {
		name string
		f    func(dst, x []float64) []float64
	}
	fns := []fn{{"Softmax", Softmax}, {"SoftmaxExact", SoftmaxExact}, {"LogSoftmax", LogSoftmax}, {"LogSoftmaxExact", LogSoftmaxExact}}
	inf, nan := math.Inf(1), math.NaN()
	for _, f := range fns {
		mustNotPanic(t, f.name+" empty", func() {
			if got := f.f(nil, nil); len(got) != 0 {
				t.Errorf("%s(nil) = %v, want empty", f.name, got)
			}
			if got := f.f(make([]float64, 0, 4), []float64{}); len(got) != 0 {
				t.Errorf("%s([]) = %v, want empty", f.name, got)
			}
		})
		for _, x := range [][]float64{{1, inf}, {nan, 1}, {1, nan}, {math.Inf(-1), math.Inf(-1)}, {inf}, {nan}} {
			mustNotPanic(t, f.name, func() {
				got := f.f(nil, x)
				for i, v := range got {
					if !math.IsNaN(v) {
						t.Errorf("%s(%v)[%d] = %v, want NaN", f.name, x, i, v)
					}
				}
			})
		}
	}
	mustNotPanic(t, "softmax with -Inf", func() {
		if got := SoftmaxExact(nil, []float64{math.Inf(-1), 0}); got[0] != 0 || got[1] != 1 {
			t.Errorf("SoftmaxExact([-Inf 0]) = %v, want [0 1]", got)
		}
		if got := Softmax(nil, []float64{math.Inf(-1), 0}); got[0] != 0 || got[1] != 1 {
			t.Errorf("Softmax([-Inf 0]) = %v, want [0 1]", got)
		}
		for _, f := range []func(dst, x []float64) []float64{LogSoftmax, LogSoftmaxExact} {
			if got := f(nil, []float64{math.Inf(-1), 0}); !math.IsInf(got[0], -1) || got[1] != 0 {
				t.Errorf("LogSoftmax([-Inf 0]) = %v, want [-Inf 0]", got)
			}
		}
	})
}

// TestSoftmaxFamilyDstReuse: a dst too short is replaced by a result of
// len(x); a dst with enough capacity is reused; the values are the exact
// softmax, e^x / sum e^x for x = [ln 1, ln 3] = [1/4, 3/4] up to rounding of
// the logarithms (here x = [0, 0]: exactly [1/2, 1/2]; x = [0, -Inf]: [1, 0]).
func TestSoftmaxFamilyDstReuse(t *testing.T) {
	for name, f := range map[string]func(dst, x []float64) []float64{"Softmax": Softmax, "SoftmaxExact": SoftmaxExact} {
		mustNotPanic(t, name, func() {
			for _, dst := range [][]float64{nil, make([]float64, 0), make([]float64, 1), make([]float64, 0, 8)} {
				for rep := 0; rep < 3; rep++ { // rep > 0 reuses the pooled scratch
					got := f(dst, []float64{0, 0, 0, 0})
					if len(got) != 4 {
						t.Fatalf("%s: len %d, want 4", name, len(got))
					}
					for i, v := range got {
						if v != 0.25 {
							t.Errorf("%s([0 0 0 0])[%d] = %v, want 0.25", name, i, v)
						}
					}
					if cap(dst) >= 4 && &got[0] != &dst[:1][0] {
						t.Errorf("%s: dst with capacity 4 not reused", name)
					}
				}
			}
			x := []float64{0, -math.Inf(1)}
			if got := f(x, x); got[0] != 1 || got[1] != 0 {
				t.Errorf("%s in place = %v, want [1 0]", name, got)
			}
		})
	}
	for name, f := range map[string]func(dst, x []float64) []float64{"LogSoftmax": LogSoftmax, "LogSoftmaxExact": LogSoftmaxExact} {
		mustNotPanic(t, name, func() {
			for _, dst := range [][]float64{nil, make([]float64, 1), make([]float64, 0, 8)} {
				got := f(dst, []float64{0, 0, 0, 0})
				if len(got) != 4 {
					t.Fatalf("%s: len %d, want 4", name, len(got))
				}
				want := -math.Log(4) // ln 4 = 2 ln 2 = 1.3862943611198906
				for i, v := range got {
					if math.Abs(v-want) > 1e-15 {
						t.Errorf("%s([0 0 0 0])[%d] = %v, want %v", name, i, v, want)
					}
				}
			}
		})
	}
	mustNotPanic(t, "LogSumExp empty", func() {
		if got := LogSumExp(nil); !math.IsInf(got, -1) {
			t.Errorf("LogSumExp(nil) = %v, want -Inf", got)
		}
		if got := LogSumExpExact(nil); !math.IsInf(got, -1) {
			t.Errorf("LogSumExpExact(nil) = %v, want -Inf", got)
		}
	})
}

func TestLogSumExpNonFinite(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	for name, f := range map[string]func([]float64) float64{"LogSumExp": LogSumExp, "LogSumExpExact": LogSumExpExact} {
		mustNotPanic(t, name, func() {
			if got := f(nil); !math.IsInf(got, -1) {
				t.Errorf("%s(nil) = %v, want -Inf", name, got)
			}
			if got := f([]float64{1, inf}); !math.IsInf(got, 1) {
				t.Errorf("%s([1 +Inf]) = %v, want +Inf", name, got)
			}
			if got := f([]float64{math.Inf(-1), math.Inf(-1)}); !math.IsInf(got, -1) {
				t.Errorf("%s([-Inf -Inf]) = %v, want -Inf", name, got)
			}
			if got := f([]float64{1, nan}); !math.IsNaN(got) {
				t.Errorf("%s([1 NaN]) = %v, want NaN", name, got)
			}
			if got := f([]float64{nan, 1, inf}); !math.IsNaN(got) {
				t.Errorf("%s([NaN 1 +Inf]) = %v, want NaN", name, got)
			}
			if got := f([]float64{math.Inf(-1), 0}); got != 0 {
				t.Errorf("%s([-Inf 0]) = %v, want 0", name, got)
			}
		})
	}
}
