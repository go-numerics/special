package special

import (
	"math"
	"testing"
)

// gammaNeg returns the certified fast path's value when it certifies, and only
// else the double-double fallback. Both give the same bits, so no value test
// can tell them apart: the mutation that never takes the early return
// (gamma_fast.go, `if ok { return v }`) leaves every result unchanged. mutsweep
// reports its deletion as "not a mutant" (it does not compile). The route is
// held without timing: the fallback is replaced by a function that returns NaN,
// and over points every region certifies gammaNeg must never return NaN.
func TestGammaNegTakesTheFastPath(t *testing.T) {
	old := gammaNegSlow
	defer func() { gammaNegSlow = old }()
	gammaNegSlow = func(float64) float64 { return math.NaN() }
	pts := []float64{-0.3, -0.45, -1.5, -2.5, -7.25, -20.5, -33.5, -50.5, -100.5, -150.25, -180.5}
	for _, x := range pts {
		if v := gammaNeg(x); math.IsNaN(v) {
			t.Errorf("gammaNeg(%g) took the fallback: the certified fast path is not taken", x)
		}
	}
}
