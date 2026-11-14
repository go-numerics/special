package special

import (
	"math"
	"testing"
	"time"
)

// The guard tests of i_fast.go. The file name sorts it ahead of the other test
// files: the first panic ends the test binary, and a guard moved one step lets
// a point into code that indexes past a table (see a_guard_test.go). These run
// first and fail by assertion.

// The guards of i_fast.go, each held by an assertion at its edge (mutation
// gate, tools/mutgate). A guard that stops an index from leaving a table is
// held by recovering the panic its removal would cause and reporting it as a
// failure: an assertion, not luck.

func ivNoPanic(t *testing.T, what string, f func()) {
	t.Helper()
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("%s: panic %v", what, r)
		}
	}()
	f()
}

// TestIvFastRangeEdges: which path answers at the exact edges.
func TestIvFastRangeEdges(t *testing.T) {
	// unscaled results are left to besselIK above x = 700, scaled ones are not
	if _, ok := ivFast(5.5, ivMaxUnscaledX, false); !ok {
		t.Error("ivFast(5.5, 700) declined")
	}
	if _, ok := ivFast(5.5, math.Nextafter(ivMaxUnscaledX, 1e9), false); ok {
		t.Error("ivFast(5.5, 700+) answered unscaled")
	}
	if _, ok := ivFast(5.5, 800, true); !ok {
		t.Error("ivFast(5.5, 800, scaled) declined")
	}
	// the base of a fractional order: the series below x = 24, the
	// large-argument pair from it
	h, l, ok := ivBase(0.3, math.Nextafter(ivAsymMinX, 0))
	sh, sl := ivSeriesPair(0.3, math.Nextafter(ivAsymMinX, 0))
	if !ok || h != sh || l != sl {
		t.Errorf("ivBase(0.3, 24-) = %v %v %v, series %v %v", h, l, ok, sh, sl)
	}
	h, l, ok = ivBase(0.3, ivAsymMinX)
	ah, al, aok := ivAsymPair(0.3, ivAsymMinX)
	if !ok || !aok || h != ah || l != al {
		t.Errorf("ivBase(0.3, 24) = %v %v %v, asymptotic %v %v %v", h, l, ok, ah, al, aok)
	}
	// the base of order 0: the I0e table, [1/8, 512)
	for _, c := range []struct {
		x    float64
		want bool
	}{
		{0.1, false}, {math.Nextafter(0.125, 0), false}, {0.125, true},
		{math.Nextafter(512, 0), true}, {512, false}, {2000, false},
	} {
		ivNoPanic(t, "ivBase(0, x)", func() {
			if _, _, ok := ivBase(0, c.x); ok != c.want {
				t.Errorf("ivBase(0, %v) ok = %v", c.x, ok)
			}
		})
		ivNoPanic(t, "ivMiller(3, x)", func() {
			if _, ok := ivMiller(3, c.x, true); ok != c.want {
				t.Errorf("ivMiller(3, %v) ok = %v", c.x, ok)
			}
		})
	}
	// the start level passes the table: ivMiller declines (an answer of 0 would be wrong)
	ivNoPanic(t, "ivMiller(5.5, 2000)", func() {
		if r, ok := ivMiller(5.5, 2000, true); ok {
			t.Errorf("ivMiller(5.5, 2000) = %v answered", r)
		}
	})
	// the chain overflows before the start: 1e-300 to the 7th power
	if r, ok := ivFast(7.3, 1e-300, false); ok {
		t.Errorf("ivFast(7.3, 1e-300) = %v answered with an overflowed chain", r)
	}
	if _, ok := ivFast(7.3, 1e-30, false); !ok {
		t.Error("ivFast(7.3, 1e-30) declined")
	}
}

// TestIvStartLevels: the table edge of the start level.
func TestIvStartLevels(t *testing.T) {
	// j0 = ceil(x/2): 600 is the last level of the table, and 2^-63 is not
	// reached by it there
	ivNoPanic(t, "ivStart", func() {
		if N, J, ok := ivStart(10, 1200); ok {
			t.Errorf("ivStart(10, 1200) = %d %d answered", N, J)
		}
		if N, J, ok := ivStart(10, 1202); ok {
			t.Errorf("ivStart(10, 1202) = %d %d answered past the table", N, J)
		}
		if N, J, ok := ivStart(10, 2000); ok {
			t.Errorf("ivStart(10, 2000) = %d %d answered past the table", N, J)
		}
		// j0 = 595: the level reaching 2^-63 is beyond 600
		if N, J, ok := ivStart(31, 1190); ok {
			t.Errorf("ivStart(31, 1190) = %d %d answered", N, J)
		}
		if N, J, ok := ivStart(1, 40); !ok || !(1 <= J && J <= N && N <= 80) {
			t.Errorf("ivStart(1, 40) = %d %d %v", N, J, ok)
		}
	})
	// a start level exactly at the end of the table
	hx := 1e-9
	if N := ivSolve(ivMillerMaxLevel, hx, float64(2*math.Log(hx)), ivLnTargetN); N != ivMillerMaxLevel {
		t.Errorf("ivSolve(600, 1e-9) = %d, want 600", N)
	}
}

// TestIvStartCertificate: P(n, N) <= 2^-63 and P(n, J) <= 2^-24 by a direct
// product (independent of the logarithms), N - J < 512, on a grid.
func TestIvStartCertificate(t *testing.T) {
	for n := 1; n <= 31; n += 3 {
		for x := 0.01; x < 1100; x = float64(x * 1.19) {
			N, J, ok := ivStart(n, x)
			if !ok {
				continue
			}
			hx := float64(x / 2)
			P := func(upTo int) float64 {
				p := 1.0
				for j := n; j <= upTo; j++ {
					p = float64(p * float64(math.Min(1, float64(hx/float64(j)))*math.Min(1, float64(hx/float64(j+1)))))
				}
				return p
			}
			if P(N) > 0x1p-63 || P(J) > 0x1p-24 || N-J >= 512 || J < n {
				t.Errorf("n=%d x=%g: N=%d J=%d P(N)=%g P(J)=%g", n, x, N, J, P(N), P(J))
			}
		}
	}
}

// TestIvFastDebyeEdges: the exact edges of the Debye range.
func TestIvFastDebyeEdges(t *testing.T) {
	// v = 32 is Debye (the recurrence would answer too, with other bits at
	// some x: both are within the bound, they differ at a few)
	tells := 0
	for x := 5.0; x < 80; x += 0.173 {
		d, ok := ivDebye(32, x, true)
		m, ok2 := ivMiller(32, x, true)
		if !ok || !ok2 {
			continue
		}
		if d == m {
			continue
		}
		tells++
		if got, ok := ivFast(32, x, true); !ok || got != d {
			t.Errorf("ivFast(32, %g) = %v, %v; Debye %v, recurrence %v", x, got, ok, d, m)
		}
	}
	if tells == 0 {
		t.Error("the recurrence and Debye agree at v = 32 everywhere: the edge v >= 32 is unheld")
	}
	// v = 1100 is the last Debye order, 1100.5 is not
	d, ok := ivDebye(1100, 1000, true)
	if !ok {
		t.Fatal("ivDebye(1100, 1000) declined")
	}
	if got, ok := ivFast(1100, 1000, true); !ok || got != d {
		t.Errorf("ivFast(1100, 1000) = %v, %v; Debye %v", got, ok, d)
	}
	if _, ok := ivFast(math.Nextafter(1100, 2000), 1000, true); ok {
		t.Error("ivFast answered above the last Debye order")
	}
	// x = 1e6 is the first x above Debye's range
	if _, ok := ivFast(100.5, kvDebyeMaxX, true); ok {
		t.Error("ivFast answered at x = kvDebyeMaxX")
	}
	if _, ok := ivFast(100.5, math.Nextafter(kvDebyeMaxX, 0), true); !ok {
		t.Error("ivFast declined just below kvDebyeMaxX")
	}
}

// TestIvFinishEdge: a pair below 2^-900 is not rounded here (its product with
// e^x or the sum could be subnormal and rounded twice).
func TestIvFinishEdge(t *testing.T) {
	if _, ok := ivFinish(0x1p-900, 0, 1, true); !ok {
		t.Error("ivFinish(2^-900) declined")
	}
	if _, ok := ivFinish(math.Nextafter(0x1p-900, 0), 0, 1, true); ok {
		t.Error("ivFinish(2^-900-) answered")
	}
}

// TestIvAsymRange: the large-argument pair answers on [24, 2^1000) and
// nowhere else (a series run at small x does not stop: its terms overflow).
func TestIvAsymRange(t *testing.T) {
	for _, c := range []struct {
		x    float64
		want bool
	}{
		{ivAsymMinX, true}, {math.Nextafter(ivAsymMinX, 0), false}, {1e-300, false}, {5, false},
		{math.Nextafter(ivMaxScaledX, 0), true}, {ivMaxScaledX, false}, {math.MaxFloat64, false},
	} {
		var ok bool
		done := make(chan bool, 1)
		go func() {
			defer func() {
				if recover() != nil {
					done <- false
				}
			}()
			_, _, ok = ivAsymPair(0.3, c.x)
			done <- true
		}()
		select {
		case fine := <-done:
			if !fine {
				t.Errorf("ivAsymPair(0.3, %g) panicked", c.x)
			} else if ok != c.want {
				t.Errorf("ivAsymPair(0.3, %g) ok = %v, want %v", c.x, ok, c.want)
			}
		case <-time.After(20 * time.Second):
			t.Fatalf("ivAsymPair(0.3, %g) did not return", c.x)
		}
	}
	// small x of a fractional order is the series, answered by Iv
	for _, x := range []float64{1e-300, 1e-30, 0.5, 5, 23.9} {
		if _, ok := ivFast(7.3, x, true); !ok && x != 1e-300 {
			t.Errorf("ivFast(7.3, %g) declined", x)
		}
	}
}

// TestIvLoopCaps: the loops stop at their caps where no range guard has
// stopped them (the terms overflow, then are NaN: neither test ends them).
func TestIvLoopCaps(t *testing.T) {
	type res struct {
		ok bool
		h  float64
	}
	run := func(f func() res) (res, bool) {
		ch := make(chan res, 1)
		go func() { ch <- f() }()
		select {
		case r := <-ch:
			return r, true
		case <-time.After(20 * time.Second):
			return res{}, false
		}
	}
	if r, done := run(func() res { _, _, ok := ivAsymSum(0.3, 1e-300); return res{ok: ok} }); !done || r.ok {
		t.Errorf("ivAsymSum(0.3, 1e-300): returned %v, answered %v", done, r.ok)
	}
	if r, done := run(func() res { h, _ := ivSeriesPair(0.3, 1e300); return res{h: h} }); !done || !math.IsNaN(r.h) {
		t.Errorf("ivSeriesPair(0.3, 1e300): returned %v, h = %v", done, r.h)
	}
	// and the caps are far above what the guarded ranges need
	if _, _, ok := ivAsymSum(1.5, ivAsymMinX); !ok {
		t.Error("ivAsymSum(1.5, 24) declined: cap too low or order out of reach")
	}
	if h, _ := ivSeriesPair(0.99, 23.99); math.IsNaN(h) {
		t.Error("ivSeriesPair(0.99, 23.99) hit the cap")
	}
}

// TestIvDebyeDeclines: a Debye sum that cannot have converged (v = 20 needs
// more than 16 terms at small x) is declined.
func TestIvDebyeDeclines(t *testing.T) {
	if _, ok := ivDebye(20, 5, false); ok {
		t.Error("ivDebye(20, 5) answered with a sum that cannot have converged")
	}
}
