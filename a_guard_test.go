package special

import (
	"math"
	"testing"
	"time"

	"github.com/go-numerics/special/dd"
)

// The guard tests of the fast paths (kn_fast.go, jn_yn_fast.go). This file is
// named to sort first among the test files: go test runs tests in file-name
// order, and a guard moved one step lets a point into code that indexes past a
// table, a panic that ends the test binary (the fixture tests reach x = 100
// and n = 2^63 too). The first panic ends the run, so a guard test that comes
// after it never runs and mutsweep reads the suite as "caught by PANIC": red
// by luck, not by an assertion. These tests run before, catch the panic or
// the hang themselves, and fail by an assertion (special#71, #72, #73).

// guardCall runs f and reports whether it panicked. A range guard moved one
// step lets a point into code that indexes past a table; the tests below turn
// that into an assertion, so that the guard is held by what it is for and not
// by the index panic (special#71, #72, #73; mutsweep "caught by PANIC").
func guardCall(f func()) (panicked bool) {
	defer func() {
		if recover() != nil {
			panicked = true
		}
	}()
	f()
	return
}

// TestJYGuardEdgeUpper: x = jyFastMaxX is outside every Jn/Yn fast path (the
// cell tables end there), and the float just below it is inside the
// recurrences' domain without touching a cell past the last one.
func TestJYGuardEdgeUpper(t *testing.T) {
	x := float64(jyFastMaxX)
	below := math.Nextafter(x, 0)
	for _, c := range []struct {
		name string
		call func(x float64) bool // ok
	}{
		{"ynRecurErr", func(x float64) bool { _, _, ok := ynRecurErr(3, x); return ok }},
		{"jnUp", func(x float64) bool { _, _, ok := jnUp(3, x); return ok }},
		{"jnMiller", func(x float64) bool { _, _, ok := jnMiller(120, x); return ok }},
		{"jnFastErr", func(x float64) bool { _, _, ok := jnFastErr(3, x); return ok }},
	} {
		var ok bool
		if guardCall(func() { ok = c.call(x) }) {
			t.Errorf("%s at x = %g panicked: the upper guard let the point in", c.name, x)
		} else if ok {
			t.Errorf("%s at x = %g answered: the fast path ends below it", c.name, x)
		}
		if guardCall(func() { c.call(below) }) {
			t.Errorf("%s at x = %.17g (just below the edge) panicked", c.name, below)
		}
	}
}

// TestJYCellPairLastCell: the walk from the lookup table's cell to the cell
// holding x stops at the last cell and never reads one past it, and the
// Horner loop reads exactly the 12 coefficients of a cell.
func TestJYCellPairLastCell(t *testing.T) {
	for _, x := range []float64{0.7, 1, 50, math.Nextafter(jyFastMaxX, 0)} {
		for name, f := range map[string]func(float64) jyPair{
			"J0": besselJ0Pair, "J1": besselJ1Pair, "Y0": besselY0Pair, "Y1": besselY1Pair,
		} {
			var p jyPair
			if guardCall(func() { p = f(x) }) {
				t.Errorf("%s pair at x = %.17g panicked", name, x)
				continue
			}
			if math.IsNaN(p.h) || math.IsInf(p.h, 0) || p.eb < 0 {
				t.Errorf("%s pair at x = %.17g = %+v", name, x, p)
			}
		}
	}
}

// TestJYGuardEdgesAnswer: the lower edges of the fast paths answer (they
// certify), and one step outside does not. Measured, then asserted: a guard
// moved one step either way fails here by its assertion.
func TestJYGuardEdgesAnswer(t *testing.T) {
	below := math.Nextafter(jyFastMinX, 0)
	for _, c := range []struct {
		name string
		call func() bool
		want bool
	}{
		{"jnMiller(2, jyFastMinX)", func() bool { _, _, ok := jnMiller(2, jyFastMinX); return ok }, true},
		{"jnMiller(2, below jyFastMinX)", func() bool { _, _, ok := jnMiller(2, below); return ok }, false},
		{"jnMiller(1, 1.5)", func() bool { _, _, ok := jnMiller(1, 1.5); return ok }, false},
		{"jnMiller(2, 1.5)", func() bool { _, _, ok := jnMiller(2, 1.5); return ok }, true},
		// n = x: the upward recurrence answers, Miller's (n > x) does not
		{"jnUp(5, 5)", func() bool { _, _, ok := jnUp(5, 5); return ok }, true},
		{"jnMiller(5, 5)", func() bool { _, _, ok := jnMiller(5, 5); return ok }, false},
		{"jnFastErr(5, 5)", func() bool { _, _, ok := jnFastErr(5, 5); return ok }, true},
	} {
		var got bool
		if guardCall(func() { got = c.call() }) {
			t.Errorf("%s panicked", c.name)
		} else if got != c.want {
			t.Errorf("%s ok = %v, want %v", c.name, got, c.want)
		}
	}
}

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
		{2, math.Nextafter(1, 0), false},
		{2, math.Nextafter(knFastMaxX, 0), true},
		{2, math.Nextafter(knFastMaxX, 1e9), false},
		{math.MaxInt, 3, false}, // stops at the first overflow, not after n steps
		{math.MinInt, 3, false},
		{500, 1.5, false}, // Ke_n overflows
	} {
		r, ok, panicked := knRecurSafe(c.n, c.x)
		if panicked {
			t.Errorf("knRecur(%d, %g) panicked or never returned: a guard let the point into the recurrence", c.n, c.x)
			continue
		}
		if ok != c.fast {
			t.Errorf("knRecur(%d, %g) ok = %v, want %v", c.n, c.x, ok, c.fast)
		}
		if ok && (r <= 0 || math.IsInf(r, 0)) {
			t.Errorf("knRecur(%d, %g) = %g", c.n, c.x, r)
		}
	}
}

// knRecurSafe is knRecur with a panic turned into a value, so that a guard
// moved one step lets a point into the recurrence and the test fails by its
// assertion (a mutated range guard used to be held only by the index panic in
// knStart: special#72, mutsweep "caught by PANIC").
//
// It also gives the call 20 s: with the overflow guard of knRecurScaled gone,
// n = MaxInt runs for ever, and a hang is not an assertion either (mutsweep
// HUNG); here it is reported as a panic-like fault.
func knRecurSafe(n int, x float64) (r float64, ok, panicked bool) {
	type res struct {
		r     float64
		ok, p bool
	}
	ch := make(chan res, 1)
	go func() {
		var v res
		defer func() {
			if recover() != nil {
				v.p = true
			}
			ch <- v
		}()
		v.r, v.ok = knRecur(n, x)
	}()
	select {
	case v := <-ch:
		return v.r, v.ok, v.p
	case <-time.After(20 * time.Second):
		return 0, false, true
	}
}

// TestKnGuardEdges: on both edges of x in [1, 512) the guard declines (ok ==
// false) and Kn is Kv bit for bit, or accepts and agrees with Kv within the
// sum of the bounds. The four points of special#72: 512, nextafter(512, 0),
// 1 and nextafter(1, 0), for several n and both signs of n.
func TestKnGuardEdges(t *testing.T) {
	below := func(x float64) float64 { return math.Nextafter(x, 0) }
	for _, c := range []struct {
		x    float64
		fast bool
	}{
		{512, false},
		{below(512), true},
		{1, true},
		{below(1), false},
	} {
		for _, n := range []int{2, 3, 50, -2, -50} {
			_, ok, panicked := knRecurSafe(n, c.x)
			if panicked {
				t.Errorf("knRecur(%d, %.17g) panicked instead of answering ok = %v", n, c.x, c.fast)
				continue
			}
			if ok != c.fast {
				t.Errorf("knRecur(%d, %.17g) ok = %v, want %v", n, c.x, ok, c.fast)
			}
			got, want := Kn(n, c.x), Kv(float64(n), c.x)
			if !c.fast && got != want {
				t.Errorf("Kn(%d, %.17g) = %g, want Kv = %g bit for bit", n, c.x, got, want)
			}
			if c.fast && math.Abs(got-want) > 3.5*ulp(want) {
				t.Errorf("Kn(%d, %.17g) = %g, Kv = %g", n, c.x, got, want)
			}
		}
	}
}

// TestKnChainOverflowed: the edge of the overflow test, which no (n, x) can
// reach: MaxFloat64 itself is finite, Inf and NaN are not.
func TestKnChainOverflowed(t *testing.T) {
	for _, c := range []struct {
		s    float64
		want bool
	}{
		{math.MaxFloat64, false},
		{math.Nextafter(math.MaxFloat64, 0), false},
		{1, false},
		{0, false},
		{math.Inf(1), true},
		{math.NaN(), true},
	} {
		if got := knChainOverflowed(c.s); got != c.want {
			t.Errorf("knChainOverflowed(%g) = %v, want %v", c.s, got, c.want)
		}
	}
}

// TestBinomFastGuard: the region of binomFastHL (p = min(a, b) >= 1/2, and
// either p + q < 10 with q < 10, or p < 2 with 10 <= q <= 2^40), on both sides
// of each edge. a = n-k+1 and b = k+1 are passed as nk1 and k.
func TestBinomFastGuard(t *testing.T) {
	for _, c := range []struct {
		name string
		p, q float64
		want bool
	}{
		{"p = 1/2", 0.5, 20, true},
		{"p below 1/2", math.Nextafter(0.5, 0), 20, false},
		{"p = 2 with q >= 10", 2, 20, false},
		{"p below 2 with q >= 10", math.Nextafter(2, 0), 20, true},
		{"q = 10", 1, 10, true},
		{"q below 10, p + q >= 10", 1, math.Nextafter(10, 0), false},
		{"small: q < 10 and p + q < 10", 1, 8.5, true},
		{"q = 2^40", 1, 0x1p40, true},
		{"q above 2^40", 1, math.Nextafter(0x1p40, math.Inf(1)), false},
		{"p + q = 10 with q < 10, p >= 2", 3, 7, false},
	} {
		np1 := dd.DD{Hi: c.p + c.q - 1}
		var ok bool
		if guardCall(func() { _, _, _, ok = binomFastHL(np1, dd.DD{Hi: c.p}, c.q-1) }) {
			t.Errorf("%s (p = %g, q = %g): binomFastHL panicked", c.name, c.p, c.q)
		} else if ok != c.want {
			t.Errorf("%s (p = %g, q = %g): ok = %v, want %v", c.name, c.p, c.q, ok, c.want)
		}
	}
	// logGammaUnitHL on [3/2, 2) is the second formula; on [1/2, 3/2) the first
	for _, x := range []float64{0.5, 1, math.Nextafter(1.5, 0), 1.5, 1.75, math.Nextafter(2, 0)} {
		var h, l float64
		if guardCall(func() { h, l = logGammaUnitHL(x) }) {
			t.Errorf("logGammaUnitHL(%.17g) panicked", x)
			continue
		}
		if want, _ := math.Lgamma(x); math.Abs((h+l)-want) > 1e-14*math.Max(1, math.Abs(want)) {
			t.Errorf("logGammaUnitHL(%.17g) = %g, Lgamma %g", x, h+l, want)
		}
	}
}

// TestJyMinMax: jyMax feeds the running maximum of the Y chain in jnUp, whose
// certificate would be understated by a jyMax that returned its second operand
// (mutsweep deletion of `a > b` survived: nothing asserted the order).
func TestJyMinMax(t *testing.T) {
	for _, c := range []struct{ a, b, min, max float64 }{
		{3, 2, 2, 3}, {2, 3, 2, 3}, {2, 2, 2, 2}, {0.5, 1e-300, 1e-300, 0.5}, {1e-300, 0.5, 1e-300, 0.5},
	} {
		if got := jyMin(c.a, c.b); got != c.min {
			t.Errorf("jyMin(%g, %g) = %g, want %g", c.a, c.b, got, c.min)
		}
		if got := jyMax(c.a, c.b); got != c.max {
			t.Errorf("jyMax(%g, %g) = %g, want %g", c.a, c.b, got, c.max)
		}
	}
}
