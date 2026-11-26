package special

import (
	"math"
	"testing"
)

// Third chunk of `mutsweep -sites all` guards: value guards found among the
// untriaged survivors. References are stated in each test.

// TestJnZerosNonPositiveCount (bessel_zeros.go:22): a non-positive count
// returns an empty, non-nil slice; make([]float64, nt) with nt < 0 would panic.
func TestJnZerosNonPositiveCount(t *testing.T) {
	for _, nt := range []int{0, -1, -7} {
		mustNotPanic(t, "JnZeros", func() {
			z := JnZeros(1, nt)
			if z == nil || len(z) != 0 {
				t.Errorf("JnZeros(1, %d) = %v, want an empty non-nil slice", nt, z)
			}
		})
	}
}

// TestLogGammaSignOfNegativeArguments (loggamma.go:62, :84): the sign of
// Gamma is (-1)^ceil(-x) between its poles (DLMF 5.5.4; parity from the
// integer ceil(-x)), and LogGammaSign(-0) is (+Inf, -1) because Gamma(-0) = -Inf.
func TestLogGammaSignOfNegativeArguments(t *testing.T) {
	mustNotPanic(t, "LogGammaSign", func() {
		if v, s := LogGammaSign(math.Copysign(0, -1)); !math.IsInf(v, 1) || s != -1 {
			t.Errorf("LogGammaSign(-0) = (%v, %d), want (+Inf, -1)", v, s)
		}
		if v, s := LogGammaSign(0); !math.IsInf(v, 1) || s != 1 {
			t.Errorf("LogGammaSign(+0) = (%v, %d), want (+Inf, 1)", v, s)
		}
		for _, x := range []float64{-0.5, -1.5, -2.5, -3.25, -10.75, -33.5, -100.125} {
			want := 1
			if int64(math.Ceil(-x))&1 == 1 {
				want = -1
			}
			if _, s := LogGammaSign(x); s != want {
				t.Errorf("LogGammaSign(%v) sign = %d, want %d", x, s, want)
			}
		}
	})
}
