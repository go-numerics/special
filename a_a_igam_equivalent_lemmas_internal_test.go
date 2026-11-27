package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/dd"
)

// TestLemmaIgamPrefixUnderflowIsZero (incgamma.go:161, :216): when
// exp(log prefix) is out of range the prefix mantissa is exactly (0, k = 0)
// (dd.ExpMant returns the zero pair below -1e7), and then the early return
// `m.Hi == 0` and the product m*S/a (resp. m/Den) are the same zero: a = 1e9
// gives log(x^a e^-x/Gamma(a)) of about -2e10 (Stirling, DLMF 5.11.1), so
// P(1e9, 1) and Q(1e9, 2e9) are below every float64. The series factor S is
// finite there (sum of positive terms, 1 for x = 1), the denominator is
// finite and nonzero in the continued fraction's domain.
func TestLemmaIgamPrefixUnderflowIsZero(t *testing.T) {
	m, k := dd.ExpMant(igamLogPrefix(1e9, 1))
	if m.Hi != 0 || k != 0 {
		t.Fatalf("ExpMant(prefix(1e9, 1)) = (%v, %d), want zero", m, k)
	}
	if s := igamLowerSeries(1e9, 1); !(s.Hi >= 1 && s.Hi < 1.0000001) {
		t.Errorf("S(1e9, 1) = %v, want 1 + 1e-9", s)
	}
	if v, kk := igamSeriesPMant(1e9, 1); v.Hi != 0 || kk != 0 {
		t.Errorf("igamSeriesPMant(1e9, 1) = (%v, %d), want zero", v, kk)
	}
	m, k = dd.ExpMant(igamLogPrefix(1e9, 2e9))
	if m.Hi != 0 || k != 0 {
		t.Fatalf("ExpMant(prefix(1e9, 2e9)) = (%v, %d), want zero", m, k)
	}
	if den := igamFractionDenDD(1e9, 2e9); !(den.Hi > 0) || math.IsInf(den.Hi, 0) {
		t.Errorf("Den(1e9, 2e9) = %v, want finite positive", den)
	}
	if v, kk := igamFractionQMant(1e9, 2e9); v.Hi != 0 || kk != 0 {
		t.Errorf("igamFractionQMant(1e9, 2e9) = (%v, %d), want zero", v, kk)
	}
}
