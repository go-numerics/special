package special

import (
	"math"
	"testing"
)

// TestKnFastAgainstKv: the recurrence and Steed's method in double-double
// agree to the sum of their bounds on a grid, and the sign of n does not
// matter. The accuracy of the fast path itself is TestAccuracyBound's "kn"
// fixture (mpmath).
func TestKnFastAgainstKv(t *testing.T) {
	worst := 0.0
	for _, n := range []int{2, 3, 4, 7, 10, 20, 33, 100, 400, 1024} {
		for x := 1.0; x < 512; x = float64(x * 1.0371) {
			r, ok := knRecur(n, x)
			if !ok {
				continue
			}
			want := Kv(float64(n), x)
			if d := math.Abs(r-want) / ulp(want); d > worst {
				worst = d
			}
			if Kn(-n, x) != Kn(n, x) || Kn(n, x) != r {
				t.Fatalf("Kn(%d, %g) = %g, knRecur %g, Kn(-n) %g", n, x, Kn(n, x), r, Kn(-n, x))
			}
		}
	}
	if worst > 3.5 {
		t.Errorf("Kn vs Kv: %.2f ulp apart", worst)
	}
}

// TestKnFallback: outside the fast range Kn is Kv, bit for bit.
func TestKnFallback(t *testing.T) {
	for _, c := range []struct {
		n int
		x float64
	}{{0, 2}, {1, 2}, {5, 0.5}, {5, 600}, {2000, 3000}, {500, 1.5}, {3, 0}, {3, math.Inf(1)}, {3, -1}} {
		if a, b := Kn(c.n, c.x), Kv(float64(c.n), c.x); a != b && !(a != a && b != b) {
			t.Errorf("Kn(%d, %g) = %g, Kv = %g", c.n, c.x, a, b)
		}
	}
	if !math.IsInf(Kn(500, 1.5), 1) {
		t.Error("Kn(500, 1.5) must overflow")
	}
}

// TestKnFastFixture: against the mpmath rows of the "kn" fixture, restricted
// to the rows the recurrence answers, the error stays below the derived bound
// (0.5 ulp rounding + 2^-57 + 2^-76 of kn_fast.go; 0.56 with margin for the
// fixture's own double-double rounding of the exact value).
func TestKnFastFixture(t *testing.T) {
	const bound = 0.56
	worst := map[string]float64{}
	count := map[string]int{}
	for _, r := range knFixtureRows(t) {
		n, x := int(r.args[0]), r.args[1]
		got, ok := knRecur(n, x)
		if !ok {
			continue
		}
		count[r.region]++
		if e := ulpErr(got, r.hi, r.lo); e > worst[r.region] {
			worst[r.region] = e
		}
	}
	if len(count) == 0 {
		t.Fatal("no fixture row reached the recurrence")
	}
	for reg, w := range worst {
		t.Logf("%-22s %5d rows  worst %.3f ulp", reg, count[reg], w)
		if w > bound {
			t.Errorf("region %q: %.3f ulp > %.2f", reg, w, bound)
		}
	}
}

// TestKnFastBeforeAfter logs, on the same rows, the worst error of the
// double-double Steed path (Kv, what Kn was before) next to the recurrence.
func TestKnFastBeforeAfter(t *testing.T) {
	old := map[string]float64{}
	for _, r := range knFixtureRows(t) {
		if _, ok := knRecur(int(r.args[0]), r.args[1]); !ok {
			continue
		}
		if e := ulpErr(Kv(r.args[0], r.args[1]), r.hi, r.lo); e > old[r.region] {
			old[r.region] = e
		}
	}
	for reg, w := range old {
		t.Logf("%-22s Steed dd worst %.3f ulp", reg, w)
	}
}

// TestKnFastOverflowBand: where Ke_n is finite but m.Hi Ke_n (m from ExpMant,
// up to 2.02) passes MaxFloat64, the product must not overflow to NaN: Kn
// answers like Kv, finite or +Inf, on every float around the edge where K_n
// crosses MaxFloat64.
func TestKnFastOverflowBand(t *testing.T) {
	for _, n := range []int{170, 300, 500, 700, 1000} {
		// the smallest x in [1, 512) where the scaled chain Ke_n is finite
		// (Ke_n decreases in x); the band starts there
		lo, hi := math.Float64bits(1), math.Float64bits(511.9)
		if h, _ := knRecurScaled(n, 1); !math.IsInf(h, 1) {
			t.Fatalf("n = %d: Ke_n finite at x = 1, the band is out of reach", n)
		}
		for lo < hi {
			mid := (lo + hi) / 2
			if h, _ := knRecurScaled(n, math.Float64frombits(mid)); math.IsInf(h, 1) {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		for b := lo - 1500; b < lo+1500; b++ {
			x := math.Float64frombits(b)
			got, want := Kn(n, x), Kv(float64(n), x)
			if got != got {
				t.Fatalf("Kn(%d, %.17g) = NaN, Kv = %g", n, x, want)
			}
			if math.IsInf(want, 1) != math.IsInf(got, 1) && !(want > 1e308 || got > 1e308) {
				t.Fatalf("Kn(%d, %.17g) = %g, Kv = %g", n, x, got, want)
			}
			if !math.IsInf(want, 1) && !math.IsInf(got, 1) && math.Abs(got-want) > 3.5*ulp(want) {
				t.Fatalf("Kn(%d, %.17g) = %g, Kv = %g", n, x, got, want)
			}
		}
	}
}

// knFixtureRows are the rows of the "kn" fixture (regions of the other
// functions) and of "kn_edge" (the corners of the recurrence: n up to 1024, x
// just above 1 and just below 512, the overflow edge of the scaled chain; rows
// where the chain overflows are answered by Kv, so 1024 is a fixture size, not
// a limit of the fast path).
func knFixtureRows(t *testing.T) []fixtureRow {
	return append(loadFixture(t, "kn"), loadFixture(t, "kn_edge")...)
}
