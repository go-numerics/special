package special

import (
	"math"
	"testing"
)

// TestKnFastRange: where the recurrence applies and where Kv must answer.
func TestKnFastRange(t *testing.T) {
	for _, c := range []struct {
		n    int
		x    float64
		fast bool
	}{
		{2, 1, true},
		{2, 0.999, false},
		{2, 511.9, true},
		{2, 512, false},
		{2, math.NaN(), false},
		{2, math.Inf(1), false},
		{2, -3, false},
		{0, 3, false},
		{1, 3, false},
		{-1, 3, false},
		{-7, 3, true},
		{600, 500, true},
		{knFastMaxN + 1, 500, false},
		{math.MinInt, 3, false},
		{500, 1.5, false}, // Ke_n overflows
	} {
		r, ok := knRecur(c.n, c.x)
		if ok != c.fast {
			t.Errorf("knRecur(%d, %g) ok = %v, want %v", c.n, c.x, ok, c.fast)
		}
		if ok && (r <= 0 || math.IsInf(r, 0)) {
			t.Errorf("knRecur(%d, %g) = %g", c.n, c.x, r)
		}
	}
}

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
	for _, r := range loadFixture(t, "kn") {
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
	for _, r := range loadFixture(t, "kn") {
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
