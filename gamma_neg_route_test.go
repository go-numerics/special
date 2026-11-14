package special

import (
	"testing"
	"time"
)

// gammaNeg returns the certified fast path's value when it certifies, and only
// else the double-double fallback. Both give the same bits, so no value test
// can tell them apart: the mutation that never takes the early return
// (gamma_fast.go, `if ok { return v }`) leaves every result unchanged and was
// reported by mutsweep as "not a mutant" (it does not compile when the guard is
// deleted outright; the same guard held false compiles and passes the suite).
// What the early return buys is time, so the test asserts the route by its
// cost: over points every region certifies, gammaNeg must take clearly less
// than gammaNegDD (best of several interleaved rounds, so a loaded machine
// cannot fail it, only a missing fast path).
func TestGammaNegTakesTheFastPath(t *testing.T) {
	pts := []float64{-0.3, -0.45, -1.5, -2.5, -7.25, -20.5, -33.5, -50.5, -100.5, -150.25, -180.5}
	const reps = 400
	var sink float64
	best := func(f func(float64) float64) time.Duration {
		b := time.Duration(1 << 62)
		for round := 0; round < 7; round++ {
			t0 := time.Now()
			for r := 0; r < reps; r++ {
				for _, x := range pts {
					sink += f(x)
				}
			}
			if d := time.Since(t0); d < b {
				b = d
			}
		}
		return b
	}
	dd := best(gammaNegDD)
	fast := best(gammaNeg)
	_ = sink
	// Measured: the fast path is 1.7x to 4x faster across architectures
	// (riscv64 and loong64 under qemu the lowest); the never-taken mutant is
	// about 0.7x (it pays both). 1.25x separates them.
	if fast*5 > dd*4 {
		t.Errorf("gammaNeg %v vs gammaNegDD %v: the certified fast path is not taken (want at least 1.25x faster)", fast, dd)
	}
}
