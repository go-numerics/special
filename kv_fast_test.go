package special

import (
	"math"
	"testing"
)

// kvFixtureRows are the rows of kv, kv_edge (unscaled) or kve, kve_edge
// (scaled): the regions of the other functions and the corners of kv_fast.go.
func kvFixtureRows(t testing.TB, scaled bool) []fixtureRow {
	if scaled {
		return append(loadFixture(t, "kve"), loadFixture(t, "kve_edge")...)
	}
	return append(loadFixture(t, "kv"), loadFixture(t, "kv_edge")...)
}

// kvFastRow answers a row by the fast path alone, on |v| as Kv does.
func kvFastRow(r fixtureRow, scaled bool) (float64, bool) {
	v, x := math.Abs(r.args[0]), r.args[1]
	if !(v > 1 && x > 0 && x <= math.MaxFloat64) || v != v || math.IsInf(v, 0) {
		return 0, false
	}
	return kvFast(v, x, scaled)
}

// TestKvFastFixture: against the mpmath rows, the rows the fast paths answer
// stay below kvFastBound, in both paths and for Kv and Kve.
func TestKvFastFixture(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		worst := map[string]float64{}
		count := map[string]int{}
		var debye, asym int
		for _, r := range kvFixtureRows(t, scaled) {
			got, ok := kvFastRow(r, scaled)
			if !ok {
				continue
			}
			if math.Abs(r.args[0]) >= kvDebyeMinV {
				debye++
			} else {
				asym++
			}
			count[r.region]++
			if e := ulpErr(got, r.hi, r.lo); e > worst[r.region] {
				worst[r.region] = e
			}
		}
		if debye < 100 || asym < 100 {
			t.Fatalf("scaled=%v: only %d Debye and %d large-argument rows reached the fast paths", scaled, debye, asym)
		}
		for reg, w := range worst {
			t.Logf("scaled=%v %-22s %5d rows  worst %.3f ulp", scaled, reg, count[reg], w)
			if w > kvFastBound {
				t.Errorf("scaled=%v region %q: %.3f ulp > %.2f", scaled, reg, w, kvFastBound)
			}
		}
	}
}

// TestKvFastBeforeAfter logs, on the rows the fast paths answer, the worst
// error of besselIK (Kv before this change) next to the fast path.
func TestKvFastBeforeAfter(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		oldW, newW := map[string]float64{}, map[string]float64{}
		for _, r := range kvFixtureRows(t, scaled) {
			got, ok := kvFastRow(r, scaled)
			if !ok {
				continue
			}
			_, old := besselIK(math.Abs(r.args[0]), r.args[1], false, true, scaled)
			if e := ulpErr(old, r.hi, r.lo); e > oldW[r.region] {
				oldW[r.region] = e
			}
			if e := ulpErr(got, r.hi, r.lo); e > newW[r.region] {
				newW[r.region] = e
			}
		}
		for reg := range newW {
			t.Logf("scaled=%v %-22s before %.3f ulp  after %.3f ulp", scaled, reg, oldW[reg], newW[reg])
		}
	}
}

// TestKvFastFallbackRate logs the share of the fixture rows (v > 1) that the
// fast paths decline.
func TestKvFastFallbackRate(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		var n, hit int
		for _, r := range kvFixtureRows(t, scaled)[:] {
			v, x := math.Abs(r.args[0]), r.args[1]
			if !(v > 1 && x > 0 && x <= math.MaxFloat64) || math.IsInf(v, 0) || v == math.Trunc(v) {
				continue
			}
			n++
			if _, ok := kvFast(v, x, scaled); ok {
				hit++
			}
		}
		t.Logf("scaled=%v: %d of %d non-integer rows with v > 1 answered by kvFast (%.1f%% fall back)",
			scaled, hit, n, 100*float64(n-hit)/float64(n))
	}
}

// TestKvFastFallback: where kvFast declines, Kv and Kve are besselIK, bit for
// bit.
func TestKvFastFallback(t *testing.T) {
	for _, c := range []struct{ v, x float64 }{
		{2.5, 3}, {7.7, 10}, {33.3, 1e6}, {1200.5, 5}, {3.3, 23.9}, {0.4, 5}, {50.5, 2e6}, {2.5, 30}, {4.9, 40},
	} {
		if _, ok := kvFast(c.v, c.x, false); ok {
			t.Errorf("kvFast(%g, %g) answered, expected a decline", c.v, c.x)
			continue
		}
		_, want := besselIK(c.v, c.x, false, true, false)
		if got := Kv(c.v, c.x); got != want && !(got != got && want != want) {
			t.Errorf("Kv(%g, %g) = %g, besselIK %g", c.v, c.x, got, want)
		}
		_, want = besselIK(c.v, c.x, false, true, true)
		if got := Kve(c.v, c.x); got != want && !(got != got && want != want) {
			t.Errorf("Kve(%g, %g) = %g, besselIK %g", c.v, c.x, got, want)
		}
	}
}

// TestKvFastNegativeOrder: K_{-v} = K_v exactly, through the same code.
func TestKvFastNegativeOrder(t *testing.T) {
	for _, v := range []float64{2.5, 20.3, 31.9, 32, 41.5, 300.5, 1099.5} {
		for _, x := range []float64{0.5, 3, 30, 300} {
			if Kv(-v, x) != Kv(v, x) || Kve(-v, x) != Kve(v, x) {
				t.Errorf("K(-%g, %g) != K(%g, %g)", v, x, v, x)
			}
		}
	}
}

// TestKvFastAgainstSteed: on a grid, the fast paths and besselIK (Steed's CF2
// in double-double) agree to the sum of their bounds.
func TestKvFastAgainstSteed(t *testing.T) {
	worst := 0.0
	for _, v := range []float64{1.3, 2.5, 7.7, 15.2, 31.5, 32, 40.3, 120.7, 500.4, 1099} {
		for x := 0.5; x < 3000; x = float64(x * 1.071) {
			for _, scaled := range []bool{false, true} {
				got, ok := kvFast(v, x, scaled)
				if !ok {
					continue
				}
				_, want := besselIK(v, x, false, true, scaled)
				if want == 0 || math.IsInf(want, 0) || math.Abs(want) < 0x1p-1000 {
					continue
				}
				if d := math.Abs(got-want) / ulp(want); d > worst {
					worst = d
				}
			}
		}
	}
	if worst > 3.5 {
		t.Errorf("kvFast vs besselIK: %.2f ulp apart", worst)
	}
	t.Logf("kvFast vs besselIK: %.2f ulp apart at worst", worst)
}

// The guards of kv_fast.go, each held by an assertion at its edge
// (mutation gate, tools/mutgate).

// TestKvFastRangeEdges: which path answers at the exact edges of the ranges.
func TestKvFastRangeEdges(t *testing.T) {
	// v = 32 is Debye (the chain, which would answer too, has other bits at
	// some x: the two are each within 0.6 ulp, so they differ at a few)
	tells := 0
	for x := 40.0; x < 80; x += 0.173 {
		d, ok := kvDebye(32, x, false)
		c, ok2 := kvAsymChain(32, x, false)
		if !ok || !ok2 {
			t.Fatalf("v = 32, x = %g: Debye %v, chain %v", x, ok, ok2)
		}
		if d == c {
			continue
		}
		tells++
		if got, ok := kvFast(32, x, false); !ok || got != d {
			t.Errorf("kvFast(32, %g) = %v, %v; Debye %v, chain %v", x, got, ok, d, c)
		}
	}
	if tells == 0 {
		t.Error("the chain and Debye agree at v = 32 everywhere: the edge v >= 32 is unheld")
	}
	// v = 1100 is the last Debye order, 1100.5 is not
	d, ok := kvDebye(1100, 1000, false)
	if !ok {
		t.Fatal("kvDebye(1100, 1000) declined")
	}
	if got, ok := kvFast(1100, 1000, false); !ok || got != d {
		t.Errorf("kvFast(1100, 1000) = %v, %v; Debye %v", got, ok, d)
	}
	if _, ok := kvFast(math.Nextafter(1100, 2000), 1000, false); ok {
		t.Error("kvFast answered above the last Debye order")
	}
	// x = 1e6 is the first x above Debye's range
	if _, ok := kvFast(100.5, kvDebyeMaxX, false); ok {
		t.Error("kvFast answered at x = kvDebyeMaxX")
	}
	if _, ok := kvFast(100.5, math.Nextafter(kvDebyeMaxX, 0), false); !ok {
		t.Error("kvFast declined just below kvDebyeMaxX")
	}
	// the chain starts at x = 24, for orders the single series cannot reach
	if _, ok := kvFast(7.5, kvAsymMinX, false); !ok {
		t.Error("kvFast(7.5, 24) declined")
	}
	if _, ok := kvFast(7.5, math.Nextafter(kvAsymMinX, 0), false); ok {
		t.Error("kvFast(7.5, 24-) answered")
	}
	// 4 v^2 - 1 = 4 x: the single series of besselAsymIK still converges
	if _, ok := kvFast(5.5, 30, false); ok {
		t.Error("kvFast(5.5, 30) answered where 4v^2-1 = 4x")
	}
	if _, ok := kvFast(math.Nextafter(5.5, 6), 30, false); !ok {
		t.Error("kvFast(5.5+, 30) declined")
	}
}

// TestKvExpClass: the classes at their exact edges.
func TestKvExpClass(t *testing.T) {
	for _, c := range []struct {
		e    float64
		want int
	}{
		{0, 0}, {math.Nextafter(kvExpWindow, 0), 0}, {-math.Nextafter(kvExpWindow, 0), 0},
		{kvExpWindow, 2}, {-kvExpWindow, 2}, {kvExpInf, 2}, {-kvExpInf, 2},
		{math.Nextafter(kvExpInf, 1e9), 1}, {-math.Nextafter(kvExpInf, 1e9), -1}, {math.NaN(), 2},
	} {
		if got := kvExpClass(c.e); got != c.want {
			t.Errorf("kvExpClass(%v) = %d, want %d", c.e, got, c.want)
		}
	}
}

// TestKvKeep: a result below 2^-1000 is not kept.
func TestKvKeep(t *testing.T) {
	if !kvKeep(0x1p-1000) || kvKeep(math.Nextafter(0x1p-1000, 0)) || kvKeep(0) || !kvKeep(1) {
		t.Error("kvKeep")
	}
	// the guard held on the chain: e^-720 is subnormal
	if r, ok := kvAsymChain(7.5, 720, false); ok {
		t.Errorf("kvAsymChain(7.5, 720) = %g answered, subnormal", r)
	}
	if _, ok := kvAsymChain(7.5, 640, false); !ok {
		t.Error("kvAsymChain(7.5, 640) declined")
	}
}

// TestKvDebyeBand: in the band between the window and the Inf/0 classes the
// Debye path declines (K e^e is finite there, beside the edge of the float64
// range), and so does a sum that has not converged: v = 20 needs more than 16
// terms; v = 28 with p near 1 needs exactly 16.
func TestKvDebyeBand(t *testing.T) {
	// v = 100.5: find x with exponent about 705, where K_v(x) is near 1e305
	lo, hi := 1e-3, 100.0
	for i := 0; i < 200; i++ {
		mid := math.Sqrt(lo * hi)
		_, e := kvDebyeExponent(100.5, mid, false)
		if e.Hi > 705 {
			lo = mid
		} else {
			hi = mid
		}
	}
	if _, ok := kvDebye(100.5, lo, false); ok {
		t.Errorf("kvDebye(100.5, %g) answered with exponent in the band", lo)
	}
	if _, ok := kvDebye(20, 5, false); ok {
		t.Error("kvDebye(20, 5) answered with a sum that cannot have converged")
	}
	got, ok := kvDebye(28, 1e-3, false)
	if !ok {
		t.Fatal("kvDebye(28, 1e-3) declined: the 16th term must be reached")
	}
	_, want := besselIK(28, 1e-3, false, true, false)
	if d := math.Abs(got-want) / ulp(want); d > 3.5 {
		t.Errorf("kvDebye(28, 1e-3) = %g, besselIK %g: %.1f ulp", got, want, d)
	}
}

// TestKvDebyeLastTerm: some (v, x) below the Debye range need all 16 terms,
// and kvDebye answers them; at v = 32 the 16th is never needed.
func TestKvDebyeLastTerm(t *testing.T) {
	need16 := 0
	for v := 24.0; v < 32; v += 0.25 {
		for x := 1e-3; x < 200; x = float64(x * 1.37) {
			S, _ := kvDebyeExponent(v, x, false)
			_, last, done := kvDebyeTail(v, S.Hi)
			if !done || last != kvDebyeTerms {
				continue
			}
			need16++
			if e := kvExpClass(func() float64 { _, e := kvDebyeExponent(v, x, false); return e.Hi }()); e != 0 {
				continue
			}
			if _, ok := kvDebye(v, x, false); !ok {
				t.Fatalf("kvDebye(%g, %g) declined with the 16th term converging", v, x)
			}
		}
	}
	if need16 == 0 {
		t.Fatal("no (v, x) needs the 16th term")
	}
	worst := 0
	for x := 1e-3; x < 1e6; x = float64(x * 1.37) {
		S, _ := kvDebyeExponent(kvDebyeMinV, x, false)
		_, last, done := kvDebyeTail(kvDebyeMinV, S.Hi)
		if !done {
			t.Fatalf("v = 32, x = %g: the Debye sum did not converge", x)
		}
		if last > worst {
			worst = last
		}
	}
	t.Logf("v = 32: at most %d terms; %d (v, x) below 32 reach the 16th", worst, need16)
	if worst >= kvDebyeTerms {
		t.Errorf("v = 32 needs %d terms, the cap is %d", worst, kvDebyeTerms)
	}
}

// TestKvAsymMonotone: the order u + 1 converges only where u does (the terms
// of the larger order are the larger): the chain declines when either pair
// does, and the second is the binding one.
func TestKvAsymMonotone(t *testing.T) {
	for u := 0.0; u < 1; u += 0.0101 {
		for x := 3.0; x < 60; x += 0.37 {
			_, _, ok0 := kvAsymPair(u, x)
			_, _, ok1 := kvAsymPair(u+1, x)
			if ok1 && !ok0 {
				t.Errorf("u = %g, x = %g: u + 1 converges, u does not", u, x)
			}
		}
	}
}

// TestKvAsymDeclines: the pair declines where the expansion cannot reach
// 2^-66 (the terms grow first), and the chain follows it, for the first order
// and for the second.
func TestKvAsymDeclines(t *testing.T) {
	if _, _, ok := kvAsymPair(0.3, 3); ok {
		t.Error("kvAsymPair(0.3, 3) answered")
	}
	if _, ok := kvAsymChain(7.3, 3, false); ok {
		t.Error("kvAsymChain(7.3, 3) answered")
	}
	// an x where the order u converges but u + 1 does not
	found := false
	for x := 30.0; x > 3; x -= 0.01 {
		_, _, ok0 := kvAsymPair(0.99, x)
		_, _, ok1 := kvAsymPair(1.99, x)
		if ok0 && !ok1 {
			found = true
			if _, ok := kvAsymChain(5.99, x, false); ok {
				t.Errorf("kvAsymChain(5.99, %g) answered, the order 1.99 does not converge", x)
			}
			break
		}
	}
	if !found {
		t.Error("no x with u converging and u+1 not: the second decline of kvAsymChain is unheld")
	}
}

// TestKvAsymTermCount: the large-argument terms of every order below 2 fall
// below 2^-66 within 40 steps for x >= kvAsymMinX (independent recount; the
// loop of kvAsymPair has no cap, its termination is the growing ratio).
func TestKvAsymTermCount(t *testing.T) {
	worst := 0
	for w := 0.0; w < 2; w += 0.0137 {
		for _, x := range []float64{kvAsymMinX, 24.5, 25, 30, 50, 100, 1e4} {
			mu := float64(float64(4*w) * w)
			term := (mu - 1) / float64(8*x)
			k := 1
			for ; math.Abs(term) >= 0x1p-66 && k < 200; k++ {
				o := float64(2*(k+1) - 1)
				term = float64(term * ((mu - float64(o*o)) / float64(float64(8*float64(k+1))*x)))
			}
			if k > worst {
				worst = k
			}
		}
	}
	if worst > 40 {
		t.Errorf("a series needs %d terms, more than 40", worst)
	}
	t.Logf("most terms needed: %d", worst)
}
