package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/dd"
)

// The 100% coverage gate sits on a knife edge (five statements of about
// ten thousand: the gate prints 99.95 as 100.0), so the branches that no
// fixture reaches are exercised here directly.
func TestCoverageGaps(t *testing.T) {
	// the rational fits of I1e and K1e for x <= 20 (the octave tables serve
	// that range; the fits remain below the asymptotic series)
	for _, x := range []float64{10, 15} {
		if got, want := besselI1eLarge(x), I1e(x); math.Abs(got-want) > float64(1e-13*want) {
			t.Errorf("besselI1eLarge(%v) = %v, I1e = %v", x, got, want)
		}
		if got, want := besselK1e(x), K1e(x); math.Abs(got-want) > float64(1e-13*want) {
			t.Errorf("besselK1e(%v) = %v, K1e = %v", x, got, want)
		}
	}
	// the Stirling reference of the gamma table stops at gammaMax
	if g := gammaLargeStirling(200); !math.IsInf(g, 1) {
		t.Errorf("gammaLargeStirling(200) = %v", g)
	}
	// the fast binomial path refuses a result outside exp(±700)
	if _, _, _, ok := binomFastHL(dd.F(1e307), dd.F(1), 0); ok {
		t.Error("binomFastHL accepted exp(-xh) for |xh| > 700")
	}
}
