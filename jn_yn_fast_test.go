package special

import (
	"math"
	"testing"
	"time"
)

// TestJYStartError: the start pairs of the recurrences are within their own
// error bound eb of the mpmath rows of the "j0", "j1", "y0", "y1" fixtures
// (hi + lo is the value to 2^-107), on every row, small |h| next to the
// centre of a cell included.
func TestJYStartError(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(x float64) jyPair
	}{
		{"j0", besselJ0Pair},
		{"j1", besselJ1Pair},
		{"y0", besselY0Pair},
		{"y1", besselY1Pair},
	} {
		worst, n := 0.0, 0
		var wx float64
		for _, r := range loadFixture(t, c.name) {
			x := r.args[0]
			if !(x >= jyFastMinX && x < jyFastMaxX) || r.hi == 0 {
				continue
			}
			p := c.f(x)
			d := math.Abs((p.h - r.hi) + (p.l - r.lo))
			n++
			if v := d / p.eb; v > worst {
				worst, wx = v, x
			}
		}
		t.Logf("%s: %d rows, worst error %.3g of its bound at x = %g", c.name, n, worst, wx)
		if n == 0 {
			t.Fatalf("%s: no row in range", c.name)
		}
		if worst > 1 {
			t.Errorf("%s: start pair error %.3g times its bound at x = %g", c.name, worst, wx)
		}
	}
}

// TestJYCellPairIsCellEval: jyCellPair is besselCellEval's arithmetic, bit for
// bit, on a grid (it is a copy that also returns the error bound).
func TestJYCellPairIsCellEval(t *testing.T) {
	for x := 0.01; x < jyFastMaxX; x = float64(x * 1.00931) {
		if a, b := besselJ0Pair(x), J0(x); a.h+a.l != b {
			t.Fatalf("J0 pair at %g: %g, J0 %g", x, a.h+a.l, b)
		}
		if a, b := besselJ1Pair(x), J1(x); a.h+a.l != b {
			t.Fatalf("J1 pair at %g: %g, J1 %g", x, a.h+a.l, b)
		}
		if x > 1.5 {
			if a, b := besselY0Pair(x), Y0(x); a.h+a.l != b {
				t.Fatalf("Y0 pair at %g: %g, Y0 %g", x, a.h+a.l, b)
			}
			if a, b := besselY1Pair(x), Y1(x); a.h+a.l != b {
				t.Fatalf("Y1 pair at %g: %g, Y1 %g", x, a.h+a.l, b)
			}
		}
	}
	for _, x := range []float64{0.01, 0.3, 0.6, 1, 1.5} {
		if a, b := besselY0Pair(x), Y0(x); a.h+a.l != b {
			t.Fatalf("Y0 pair at %g: %g, Y0 %g", x, a.h+a.l, b)
		}
		if a, b := besselY1Pair(x), Y1(x); a.h+a.l != b {
			t.Fatalf("Y1 pair at %g: %g, Y1 %g", x, a.h+a.l, b)
		}
	}
}

// jyFixtureRows are the rows of "yn" and "yn_edge" ("jn", "jn_edge").
func jyFixtureRows(t *testing.T, name string) []fixtureRow {
	return append(loadFixture(t, name), loadFixture(t, name+"_edge")...)
}

func testJYFast(t *testing.T, name string, fast func(int, float64) (float64, bool), bound float64) {
	worst := map[string]float64{}
	count := map[string]int{}
	dec := map[string]int{}
	rows, declined := 0, 0
	for _, r := range jyFixtureRows(t, name) {
		if r.args[0] < 2 || r.args[0] != math.Trunc(r.args[0]) || !(r.args[1] > 0) {
			continue
		}
		got, ok := fast(int(r.args[0]), r.args[1])
		if !(r.args[1] >= jyFastMinX && r.args[1] < jyFastMaxX) {
			if ok {
				t.Errorf("%s fast path answered outside its range: n = %g, x = %g", name, r.args[0], r.args[1])
			}
			continue
		}
		rows++
		if !ok {
			declined++
			dec[r.region]++
			continue
		}
		count[r.region]++
		if e := ulpErr(got, r.hi, r.lo); e > worst[r.region] || e != e {
			worst[r.region] = e
		}
	}
	if len(count) == 0 {
		t.Fatal("no fixture row reached the fast path")
	}
	t.Logf("%s: %d rows in range, %d declined (%.2f%%)", name, rows, declined, 100*float64(declined)/float64(rows))
	for reg, w := range worst {
		t.Logf("%-18s %5d answered (%d declined)  worst %.3f ulp", reg, count[reg], dec[reg], w)
		if !(w <= bound) {
			t.Errorf("region %q: %.3f ulp > %.2f", reg, w, bound)
		}
	}
}

func TestYnFastFixture(t *testing.T) { testJYFast(t, "yn", ynRecur, 6.5) }
func TestJnFastFixture(t *testing.T) { testJYFast(t, "jn", jnFast, 4.5) }

// TestJYFallbackRate logs, on the refbench inputs of the Gaps rows (order i
// mod 21, x on (0.1, 50), 10000 points), how many points the fast paths
// decline: those run Steed's method (about 30 times slower).
func TestJYFallbackRate(t *testing.T) {
	x := gapsX(0.1, 50)
	for _, c := range []struct {
		name string
		f    func(int, float64) (float64, bool)
	}{{"Yn", ynRecur}, {"Jn", func(n int, x float64) (float64, bool) {
		if besselSeriesOK(float64(n), x, besselJSeriesRatio) {
			return 0, true // the power series answers
		}
		return jnFast(n, x)
	}}} {
		fast, dec := 0, 0
		for i := range x {
			n := i % 21
			if n < 2 {
				continue
			}
			if _, ok := c.f(n, x[i]); ok {
				fast++
			} else {
				dec++
			}
		}
		t.Logf("%s: %d of %d points (n >= 2) declined = %.2f%% of those, %.2f%% of all 10000", c.name, dec, fast+dec,
			100*float64(dec)/float64(fast+dec), 100*float64(dec)/float64(len(x)))
	}
}

// TestJYCertificateHolds: where a fast path answers, the error against the
// mpmath row is within the certified bound it computed (plus the half ulp of
// the final rounding), not only below the stated bound in ulps: the
// certificate itself is what is under test. 1e-3 ulp is the fixture's own
// double-double rounding.
func TestJYCertificateHolds(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(int, float64) (float64, float64, bool)
	}{{"yn", ynRecurErr}, {"jn", jnFastErr}} {
		answered, tight := 0, 0.0
		for _, r := range jyFixtureRows(t, c.name) {
			if r.args[0] < 2 || r.args[0] != math.Trunc(r.args[0]) || !(r.args[1] > 0) {
				continue
			}
			got, e, ok := c.f(int(r.args[0]), r.args[1])
			if !ok {
				continue
			}
			answered++
			err := math.Abs((got - r.hi) - r.lo)
			lim := e + float64((0.5+1e-3)*ulp(r.hi))
			if err > lim {
				t.Fatalf("%s(%g, %g): error %.3g > certified %.3g + half ulp %.3g", c.name, r.args[0], r.args[1], err, e, float64(0.5*ulp(r.hi)))
			}
			if v := err / lim; v > tight {
				tight = v
			}
		}
		t.Logf("%s: %d rows answered, closest to its certificate: %.2f of it", c.name, answered, tight)
	}
}

// TestJYFastRange: where the fast paths apply. The assertions are on the
// range guards, so that moving one fails here and not by an index panic or a
// distant fixture row.
func TestJYFastRange(t *testing.T) {
	// The cell tables answer below 100 and the lookup tables hold 8 entries per
	// unit: a guard moved beyond that is wrong before it is a panic.
	if jyFastMaxX != besselZeroTaylorMax {
		t.Fatalf("jyFastMaxX = %g, the cells end at %g", float64(jyFastMaxX), float64(besselZeroTaylorMax))
	}
	for name, l := range map[string]int{"J0": len(besselJ0CellsLUT), "J1": len(besselJ1CellsLUT), "Y0": len(besselY0CellsLUT), "Y1": len(besselY1CellsLUT)} {
		if float64(l) < 8*jyFastMaxX {
			t.Fatalf("%s lookup table has %d entries, not enough for x < %g", name, l, float64(jyFastMaxX))
		}
	}
	if !(jyFastMinX >= 0x1p-100) {
		t.Fatalf("jyFastMinX = %g: Y_3 and the certificate's sums of products overflow below 2^-100", float64(jyFastMinX))
	}
	acceptedSomewhere := func(f func(int, float64) (float64, bool), x float64) (ok bool) {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("panic at x = %g: %v", x, r)
			}
		}()
		for _, n := range []int{2, 3, 5, 8, 13, 21, 34} {
			if _, ok := f(n, x); ok {
				return true
			}
		}
		return false
	}
	yn := func(n int, x float64) (float64, bool) { return ynRecur(n, x) }
	for _, c := range []struct {
		name string
		f    func(int, float64) (float64, bool)
	}{{"Yn", yn}, {"Jn", jnFast}} {
		if c.name == "Yn" && !acceptedSomewhere(c.f, jyFastMinX) {
			t.Errorf("%s: declines at x = 2^-100", c.name)
		}
		for _, x := range []float64{math.Nextafter(jyFastMinX, 0), 0, -1, math.Inf(1), math.NaN(), jyFastMaxX, math.Nextafter(jyFastMaxX, 1e9), 1e9} {
			for _, n := range []int{2, 3, 10, 40} {
				func() {
					// a guard moved out is reported as an assertion, not as the
					// panic of the table lookup it no longer protects
					defer func() {
						if r := recover(); r != nil {
							t.Errorf("%s(%d, %g) panicked (range guard): %v", c.name, n, x, r)
						}
					}()
					if _, ok := c.f(n, x); ok {
						t.Errorf("%s(%d, %g) answered outside its range", c.name, n, x)
					}
				}()
			}
		}
		if !acceptedSomewhere(c.f, math.Nextafter(jyFastMaxX, 0)) {
			t.Errorf("%s: declines below x = 100", c.name)
		}
		for _, n := range []int{0, 1, -1, -2, -7, math.MinInt, math.MaxInt} {
			if n < 0 && n > -1000 && c.name == "Yn" { // Yn's sign is the caller's
				continue
			}
			if _, ok := c.f(n, 3); ok && (n == 0 || n == 1 || n == -1 || n == math.MinInt || n == math.MaxInt) {
				t.Errorf("%s(%d, 3) answered", c.name, n)
			}
		}
	}
	// n <= x goes upward, n > x downward, and a path never answers for the
	// other's n
	if _, _, ok := jnUp(30, 10); ok {
		t.Error("jnUp answered for n > x")
	}
	if _, _, ok := jnMiller(10, 30); ok {
		t.Error("jnMiller answered for n <= x")
	}
	if _, _, ok := jnMiller(10, 10); ok {
		t.Error("jnMiller answered at n = x")
	}
}

// steedYn and steedJn are what Yn and Jn computed before the fast paths:
// Steed's method in double-double.
func steedYn(n int, x float64) float64 {
	_, y := besselJYPos(float64(n), x, false, true)
	return y
}

func steedJn(n int, x float64) float64 {
	j, _ := besselJYPos(float64(n), x, true, false)
	return j
}

// TestJYFallbackBits: where a fast path declines, Yn and Jn are Steed's method
// (or, for Jn, the power series) bit for bit.
func TestJYFallbackBits(t *testing.T) {
	declinedY, declinedJ := 0, 0
	for _, n := range []int{2, 3, 5, 9, 14, 20, 33, 60, 150} {
		for x := 0.05; x < 140; x = float64(x * 1.0173) {
			if _, ok := ynRecur(n, x); !ok {
				declinedY++
				if got, want := Yn(n, x), steedYn(n, x); got != want && !(got != got && want != want) {
					t.Fatalf("Yn(%d, %.17g) = %g, Steed %g", n, x, got, want)
				}
			}
			if besselSeriesOK(float64(n), x, besselJSeriesRatio) {
				continue
			}
			if _, ok := jnFast(n, x); !ok || !(x < jyFastMaxX) {
				declinedJ++
				if got, want := Jn(n, x), steedJn(n, x); got != want && !(got != got && want != want) {
					t.Fatalf("Jn(%d, %.17g) = %g, Steed %g", n, x, got, want)
				}
			}
		}
	}
	if declinedY == 0 || declinedJ == 0 {
		t.Fatalf("nothing declined (Y %d, J %d): the test checks nothing", declinedY, declinedJ)
	}
	t.Logf("declined and equal to Steed: Yn %d points, Jn %d", declinedY, declinedJ)
}

// TestJYFastAgainstSteed: on a grid the fast paths and Steed's method in
// double-double agree to the sum of their bounds, the sign of n and (for Jn)
// of x are what they should be.
func TestJYFastAgainstSteed(t *testing.T) {
	worstY, worstJ := 0.0, 0.0
	for _, n := range []int{2, 3, 4, 7, 10, 20, 33, 60, 99} {
		for x := 0.1; x < 99.9; x = float64(x * 1.0137) {
			if r, ok := ynRecur(n, x); ok {
				want := steedYn(n, x)
				if d := math.Abs(r-want) / ulp(want); d > worstY {
					worstY = d
				}
				if Yn(-n, x) != Yn(n, x)*math.Pow(-1, float64(n&1)) || Yn(n, x) != r {
					t.Fatalf("Yn(%d, %g) = %g, recurrence %g, Yn(-n) %g", n, x, Yn(n, x), r, Yn(-n, x))
				}
			}
			if r, ok := jnFast(n, x); ok && !besselSeriesOK(float64(n), x, besselJSeriesRatio) {
				want := steedJn(n, x)
				if d := math.Abs(r-want) / ulp(want); d > worstJ {
					worstJ = d
				}
				if Jn(n, x) != r || Jn(n, -x) != r*math.Pow(-1, float64(n&1)) {
					t.Fatalf("Jn(%d, %g) = %g, recurrence %g, Jn(n, -x) %g", n, x, Jn(n, x), r, Jn(n, -x))
				}
			}
		}
	}
	t.Logf("fast vs Steed: Yn %.2f ulp, Jn %.2f ulp", worstY, worstJ)
	if worstY > 6.5+3.5 {
		t.Errorf("Yn vs Steed: %.2f ulp apart", worstY)
	}
	if worstJ > 4.5+3.5 {
		t.Errorf("Jn vs Steed: %.2f ulp apart", worstJ)
	}
}

// TestYnFastOverflowBand: around the x where Y_n crosses -MaxFloat64 (the
// chain gives up at a quarter of it) Yn is what Steed's method says, never
// NaN, on every float in a band.
func TestYnFastOverflowBand(t *testing.T) {
	for _, n := range []int{100, 200, 500} {
		// the largest x where Y_n overflows to -Inf (|Y_n| increases as x falls)
		lo, hi := math.Float64bits(0.01), math.Float64bits(99.9)
		if !math.IsInf(steedYn(n, 0.01), -1) || math.IsInf(steedYn(n, 99.9), -1) {
			t.Fatalf("n = %d: no overflow edge in (0.01, 99.9)", n)
		}
		for lo < hi {
			mid := (lo + hi) / 2
			if math.IsInf(steedYn(n, math.Float64frombits(mid)), -1) {
				lo = mid + 1
			} else {
				hi = mid
			}
		}
		fast := 0
		for b := lo - 700; b < lo+700; b++ {
			x := math.Float64frombits(b)
			got, want := Yn(n, x), steedYn(n, x)
			if got != got {
				t.Fatalf("Yn(%d, %.17g) = NaN, Steed %g", n, x, want)
			}
			if math.IsInf(got, 0) != math.IsInf(want, 0) && !(math.Abs(want) > 1e308 || math.Abs(got) > 1e308) {
				t.Fatalf("Yn(%d, %.17g) = %g, Steed %g", n, x, got, want)
			}
			if !math.IsInf(want, 0) && !math.IsInf(got, 0) && math.Abs(got-want) > 10*ulp(want) {
				t.Fatalf("Yn(%d, %.17g) = %g, Steed %g", n, x, got, want)
			}
			if _, ok := ynRecur(n, x); ok {
				fast++
			}
		}
		t.Logf("n = %d: %d of 1400 floats around the overflow edge answered by the recurrence", n, fast)
	}
}

// TestJnMillerOverflow: where J_0/J_n overflows the chain (large n, small x
// not caught by the series), Jn is Steed's method or the series, finite and
// never NaN.
func TestJnMillerOverflow(t *testing.T) {
	for _, n := range []int{60, 100, 200, 400} {
		for x := 0.5; x < 40; x = float64(x * 1.07) {
			got := Jn(n, x)
			if got != got || math.IsInf(got, 0) {
				t.Fatalf("Jn(%d, %g) = %g", n, x, got)
			}
			if r, _, ok := jnMiller(n, x); ok {
				if want := steedJn(n, x); want != 0 && math.Abs(r-want) > 8*ulp(want) {
					t.Fatalf("jnMiller(%d, %g) = %g, Steed %g", n, x, r, want)
				}
			}
		}
	}
}

// TestJYFastBeforeAfter logs, per region and on the rows the fast paths
// answer, the worst error of the recurrence next to that of Steed's method in
// double-double (what Jn and Yn were before).
func TestJYFastBeforeAfter(t *testing.T) {
	for _, c := range []struct {
		name  string
		fast  func(int, float64) (float64, bool)
		steed func(int, float64) float64
	}{{"yn", ynRecur, steedYn}, {"jn", jnFast, steedJn}} {
		after, before := map[string]float64{}, map[string]float64{}
		for _, r := range jyFixtureRows(t, c.name) {
			if r.args[0] < 2 || r.args[0] != math.Trunc(r.args[0]) || !(r.args[1] > 0) || !(r.args[1] < jyFastMaxX) {
				continue
			}
			n, x := int(r.args[0]), r.args[1]
			got, ok := c.fast(n, x)
			if !ok || (c.name == "jn" && besselSeriesOK(float64(n), x, besselJSeriesRatio)) {
				continue
			}
			if e := ulpErr(got, r.hi, r.lo); e > after[r.region] {
				after[r.region] = e
			}
			if e := ulpErr(c.steed(n, x), r.hi, r.lo); e > before[r.region] {
				before[r.region] = e
			}
		}
		for reg, a := range after {
			t.Logf("%s %-18s recurrence %.2f ulp, Steed dd %.2f ulp (same rows)", c.name, reg, a, before[reg])
		}
	}
}

// TestYnFastNextToZeroOfPrevious: where Y_{n-1} has a zero (the bound of |J_n|
// from the Wronskian at n-1 divides by it) Yn(3, x) at the zero of Y_2 is
// what Steed's method says, within the bounds of both.
func TestYnFastNextToZeroOfPrevious(t *testing.T) {
	// first zero of Y_2, by bisection on the sign of Steed's Y_2
	lo, hi := 3.0, 4.0
	for i := 0; i < 80; i++ {
		mid := float64(float64(lo+hi) / 2)
		if steedYn(2, mid) < 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	for _, x := range []float64{lo, hi, math.Nextafter(lo, 0), math.Nextafter(hi, 9)} {
		got := Yn(3, x)
		want := steedYn(3, x)
		if math.Abs(got-want) > float64(8*ulp(want)) {
			t.Errorf("Yn(3, %.17g) = %g, Steed %g", x, got, want)
		}
	}
}

// withinSeconds runs f and fails (by assertion) if it has not returned after
// 20 s: a guard that ends a loop is tested by the loop ending, not by the
// suite's own timeout.
func withinSeconds(t *testing.T, what string, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() { f(); close(done) }()
	select {
	case <-done:
	case <-time.After(20 * time.Second):
		t.Fatalf("%s: still running after 20 s", what)
	}
}

// TestYnFastHugeN: the chain ends where Y_n leaves float64 (the MaxFloat64/4
// guard of ynRecurErr), however large n: the fast path declines at once.
func TestYnFastHugeN(t *testing.T) {
	for _, n := range []int{1 << 20, 1 << 30, math.MaxInt} {
		withinSeconds(t, "ynRecur", func() {
			if r, ok := ynRecur(n, 3); ok {
				t.Errorf("ynRecur(%d, 3) = %g, want a decline", n, r)
			}
		})
	}
	if got := Yn(math.MaxInt, 3); !math.IsInf(got, -1) {
		t.Errorf("Yn(MaxInt, 3) = %g, want -Inf", got)
	}
}

// TestJnMillerChainGuard: the downward chain is abandoned once it passes
// 2^1000 (the 0x1p1000 guard of jnMiller). Where that guard is weakened
// jnMiller(72, 0.0023759256118315846) returned NaN with ok = true.
func TestJnMillerChainGuard(t *testing.T) {
	for _, c := range []struct {
		n int
		x float64
	}{{72, 0.0023759256118315846}, {200, 0.5}, {1 << 30, 3}, {math.MaxInt, 3}} {
		withinSeconds(t, "jnMiller", func() {
			r, _, ok := jnMiller(c.n, c.x)
			if ok && (r != r || math.IsInf(r, 0)) {
				t.Errorf("jnMiller(%d, %g) = %g with ok", c.n, c.x, r)
			}
			if ok {
				if want := steedJn(c.n, c.x); math.Abs(r-want) > float64(8*ulp(want)) {
					t.Errorf("jnMiller(%d, %g) = %g, Steed %g", c.n, c.x, r, want)
				}
			}
		})
	}
	if _, _, ok := jnMiller(math.MaxInt, 3); ok {
		t.Error("jnMiller(MaxInt, 3) answered")
	}
}
