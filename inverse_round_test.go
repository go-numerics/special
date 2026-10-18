package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/internal/dd"
)

// TestRoundingStepLimits exercises the cases where the rounding step of
// the inverses declines and leaves the iterate (see inverse_round.go).
func TestRoundingStepLimits(t *testing.T) {
	lb := func(a, b float64) dd.DD {
		v, ok := logBetaAcc(a, b)
		if !ok {
			t.Fatalf("logBetaAcc(%v, %v) declined", a, b)
		}
		return v
	}
	lower := func(a, b, x float64) bool {
		_, ok := ibetaLowerAcc(a, b, dd.F(x), dd.LogAccurate(x), igamLog1pDD(dd.F(-x)), lb(a, b))
		return ok
	}
	for _, c := range []struct {
		name    string
		a, b, x float64
	}{
		{"largest term beyond the cap", 1, 1e6, 0.5},
		{"no convergence within the cap", 1, 140, 0.99}, // largest term at 13760
		{"prefix terms too large to cancel", 5e4, 5e4, 0.5},
		{"subnormal result", 1000, 1, 1e-3},
	} {
		if lower(c.a, c.b, c.x) {
			t.Errorf("%s: ibetaLowerAcc(%v, %v, %v) accepted", c.name, c.a, c.b, c.x)
		}
	}
	if _, ok := logBetaAcc(1e12, 1e13); ok {
		t.Error("logBetaAcc(1e12, 1e13) accepted (a/b not small)")
	}
	for _, c := range []struct {
		name  string
		a, x  float64
		upper bool
	}{
		{"x beyond roundMaxParam", 2e5, 2e5, false},
		{"x - a beyond the rise", 1, 9500, false},
		{"no convergence within 10000 terms", 1, 8990, false},
		{"subnormal result", 1000, 1e-3, false},
		{"complement below roundMinTail", 1, 30, true},
	} {
		if _, _, ok := igamTailAcc(c.a, c.x, c.upper); ok {
			t.Errorf("%s: igamTailAcc(%v, %v, %v) accepted", c.name, c.a, c.x, c.upper)
		}
	}
	// the step itself: an uncertain T, a T far from t, a step that is
	// not a rounding correction
	x := 0.25
	if got := roundStep(x, dd.F(0.5), 0.5, 1e-30, 1, 0x1p-80); got != x {
		t.Errorf("uncertain step taken: %v", got)
	}
	if got := roundStep(x, dd.F(0.6), 0.5, 1, 1, 0x1p-80); got != x {
		t.Errorf("step from a far T taken: %v", got)
	}
	if got := roundStep(x, dd.F(0.5+0x1p-31), 0.5, 1e-15, 1, 0x1p-80); got != x {
		t.Errorf("large step taken: %v", got)
	}
	if got := roundStep(x, dd.F(0.5), 0.5, 1, 1, 0x1p-80); got != x {
		t.Errorf("zero step moved x: %v", got)
	}
}

// TestHugeParameters holds the incomplete gamma and beta functions at
// parameters near MaxFloat64, where their prefixes overflowed to NaN, and
// checks the large-parameter beta limit against the gamma function it
// reduces to.
func TestHugeParameters(t *testing.T) {
	m := math.MaxFloat64
	for _, c := range []struct {
		name      string
		got, want float64
	}{
		{"GammaP(Max, 3.5)", GammaP(m, 3.5), 0},
		{"GammaQ(Max, 3.5)", GammaQ(m, 3.5), 1},
		{"BetaInc(30.5, Max, 0.97)", BetaInc(30.5, m, 0.97), 1},
		{"BetaIncC(30.5, Max, 0.97)", BetaIncC(30.5, m, 0.97), 0},
		{"BetaInc(Max, 30.5, 0.03)", BetaInc(m, 30.5, 0.03), 0},
		{"BetaIncC(Max, 30.5, 0.03)", BetaIncC(m, 30.5, 0.03), 1},
		// y = (b + (a-1)/2)(-log1p(-x)) = 1
		{"BetaInc(2, 1e301, 1e-301)", BetaInc(2, 1e301, 1e-301), GammaP(2, 1)},
		// 1 - I_x(a, b) = P(b, (a + (b-1)/2)(-log x)), y = 1.1e285 here
		{"BetaIncC(1e301, 2, 1-2^-53)", BetaIncC(1e301, 2, 1-0x1p-53), 1},
	} {
		// 1e-15: y itself is two roundings from (b + (a-1)/2)(-log1p(-x))
		if !(math.Abs(c.got-c.want) <= 1e-15*math.Abs(c.want)) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

// TestLogGammaAccAgainstLogGamma checks logGammaAcc against the float64
// LogGamma (itself within an ulp) across the shift to Stirling's series.
func TestLogGammaAccAgainstLogGamma(t *testing.T) {
	for _, x := range []float64{1e-300, 1e-8, 0.5, 1, 2.5, 9.99, 10, 29.9, 30, 31, 1e3, 1e5, 1e10} {
		got := logGammaAcc(x).Float()
		want := LogGamma(x)
		if math.Abs(got-want) > 4*ulp(want)+0x1p-80 { // log Γ(1) = 0: absolute
			t.Errorf("logGammaAcc(%v) = %v, LogGamma = %v", x, got, want)
		}
	}
}
