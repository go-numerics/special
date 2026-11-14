package special

import (
	"flag"
	"math"
	"math/big"
	"math/rand"
	"runtime"
	"sync"
	"testing"
)

var ivLong = flag.Int("ivfast.n", 0, "TestIvFastOracle: random points per region (default 4000; 1000000 for the full sweep)")

// ivRegion is a window of (v, x) and how a point of it is drawn.
type ivRegion struct {
	name string
	draw func(r *rand.Rand) (v, x float64)
}

func ivLogUniform(r *rand.Rand, lo, hi float64) float64 {
	return float64(lo * math.Exp(float64(r.Float64()*math.Log(hi/lo))))
}

// ivRegions are the windows of the paths of i_fast.go and the corners between
// them. The bound asserted on each is ivFastBound; "fallback" rows are points
// the fast path may decline, then the answer is besselIK and only counted.
var ivRegions = []ivRegion{
	{"integer, small x", func(r *rand.Rand) (float64, float64) {
		return float64(2 + r.Intn(30)), ivLogUniform(r, 0.125, 4)
	}},
	{"integer, moderate x", func(r *rand.Rand) (float64, float64) {
		return float64(2 + r.Intn(30)), 4 + float64(r.Float64()*60)
	}},
	{"integer, large x", func(r *rand.Rand) (float64, float64) {
		return float64(2 + r.Intn(30)), 64 + float64(r.Float64()*448)
	}},
	{"fractional, small x", func(r *rand.Rand) (float64, float64) {
		return float64(r.Float64() * 32), ivLogUniform(r, 0.01, 24)
	}},
	{"fractional, order < 1", func(r *rand.Rand) (float64, float64) {
		return r.Float64(), ivLogUniform(r, 0.01, 40)
	}},
	{"fractional, large x", func(r *rand.Rand) (float64, float64) {
		return float64(r.Float64() * 32), 24 + float64(r.Float64()*488)
	}},
	{"large order", func(r *rand.Rand) (float64, float64) {
		return 32 + float64(r.Float64()*1068), ivLogUniform(r, 0.5, 5000)
	}},
	{"large order, integer", func(r *rand.Rand) (float64, float64) {
		return float64(32 + r.Intn(1069)), ivLogUniform(r, 0.5, 5000)
	}},
	{"huge x, scaled", func(r *rand.Rand) (float64, float64) {
		return float64(r.Float64() * 1100), ivLogUniform(r, 1000, 1e6)
	}},
}

// ivSweep runs n points of the region through the fast path (scaled and
// unscaled alternately) against the math/big oracle, in parallel, and returns
// the worst error in ulps, the number of points answered and the number
// declined, and the worst point.
func ivSweep(reg ivRegion, n int, seed int64) (worst float64, answered, declined int, wv, wx float64, wscaled bool) {
	type res struct {
		worst    float64
		ans, dec int
		v, x     float64
		sc       bool
	}
	workers := runtime.GOMAXPROCS(0)
	if workers > 8 {
		workers = 8
	}
	out := make([]res, workers)
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			var v, x float64
			defer func() {
				if p := recover(); p != nil { // an assertion, not a crash of the binary
					out[w].worst, out[w].v, out[w].x = math.Inf(1), v, x
				}
			}()
			r := rand.New(rand.NewSource(seed + int64(w)*7919))
			for i := w; i < n; i += workers {
				v, x = reg.draw(r)
				scaled := i%2 == 0
				got, ok := ivFast(v, x, scaled)
				if !ok {
					out[w].dec++
					continue
				}
				want := bigIve(v, x)
				if !scaled {
					want = bnew().Mul(want, bigExp(bf(x)))
				}
				out[w].ans++
				if wf, _ := want.Float64(); math.IsInf(wf, 0) || (wf != 0 && math.Abs(wf) < 0x1p-1000) {
					// beyond what this path answers
					if !math.IsInf(got, 0) && got != 0 {
						// finite where the oracle overflows: fall through to the ulp check
					} else if got == wf || got == 0 {
						continue
					}
				}
				if e := ivOracleUlps(got, want); e > out[w].worst {
					out[w].worst, out[w].v, out[w].x, out[w].sc = e, v, x, scaled
				}
			}
		}(w)
	}
	wg.Wait()
	for _, o := range out {
		answered += o.ans
		declined += o.dec
		if o.worst > worst {
			worst, wv, wx, wscaled = o.worst, o.v, o.x, o.sc
		}
	}
	return
}

// TestIvFastOracle: random points of every window against the math/big
// oracle (i_oracle_test.go), each at most ivFastBound ulps, with the share the
// fast path declines. Default 4000 points per window; -ivfast.n 1000000 for
// the full sweep.
func TestIvFastOracle(t *testing.T) {
	n := 300
	if *ivLong > 0 {
		n = *ivLong
	}
	for i, reg := range ivRegions {
		worst, ans, dec, wv, wx, wsc := ivSweep(reg, n, int64(100+i))
		t.Logf("%-24s %8d answered %7d declined (%.1f%%)  worst %.3f ulp at v=%.17g x=%.17g scaled=%v",
			reg.name, ans, dec, 100*float64(dec)/float64(ans+dec), worst, wv, wx, wsc)
		if worst > ivFastBound {
			t.Errorf("%s: %.3f ulp > %.2f at v=%.17g x=%.17g scaled=%v", reg.name, worst, ivFastBound, wv, wx, wsc)
		}
	}
}

var _ = big.NewInt

// ivFixtureRows are the rows of iv, i_edge (unscaled) or ive, ive_edge
// (scaled): the regions of the other functions and the corners of i_fast.go.
func ivFixtureRows(t testing.TB, scaled bool) []fixtureRow {
	if scaled {
		return append(loadFixture(t, "ive"), loadFixture(t, "ive_edge")...)
	}
	return append(loadFixture(t, "iv"), loadFixture(t, "i_edge")...)
}

// ivFastRow answers a row by the fast path alone, on |v| and |x| as Iv does
// (the parity of an integer order only gives a sign).
func ivFastRow(r fixtureRow, scaled bool) (float64, bool) {
	v, x := r.args[0], math.Abs(r.args[1])
	if v < 0 && v != math.Trunc(v) {
		return 0, false // the reflection is besselIK's
	}
	if r.args[1] < 0 && v != math.Trunc(v) {
		return 0, false // NaN
	}
	v = math.Abs(v)
	if !(v > 0 && x > 0 && x <= math.MaxFloat64) || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	if v == 1 {
		return 0, false // I1 and I1e
	}
	return ivFast(v, x, scaled)
}

// ivWant is the signed fixture value matching ivFastRow's |x|: the sign of
// I_n(-x) for odd n is not the fast path's.
func ivSign(r fixtureRow) float64 {
	if r.args[1] < 0 && math.Mod(r.args[0], 2) != 0 {
		return -1
	}
	return 1
}

// TestIvFastFixture: against the mpmath rows, the rows the fast paths answer
// stay below ivFastBound, in every path, for Iv and Ive.
func TestIvFastFixture(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		worst := map[string]float64{}
		count := map[string]int{}
		var debye, asym, miller int
		for _, r := range ivFixtureRows(t, scaled) {
			got, ok := ivFastRow(r, scaled)
			if !ok {
				continue
			}
			v, x := math.Abs(r.args[0]), math.Abs(r.args[1])
			switch {
			case v >= kvDebyeMinV:
				debye++
			case x >= ivAsymMinX:
				asym++
			default:
				miller++
			}
			count[r.region]++
			if e := ulpErr(float64(got*ivSign(r)), r.hi, r.lo); e > worst[r.region] {
				worst[r.region] = e
			}
		}
		if debye < 100 || asym < 100 || miller < 100 {
			t.Fatalf("scaled=%v: only %d Debye, %d large-argument and %d recurrence rows reached the fast paths", scaled, debye, asym, miller)
		}
		for reg, w := range worst {
			t.Logf("scaled=%v %-24s %5d rows  worst %.3f ulp", scaled, reg, count[reg], w)
			if w > ivFastBound {
				t.Errorf("scaled=%v region %q: %.3f ulp > %.2f", scaled, reg, w, ivFastBound)
			}
		}
	}
}

// ivOld is Iv (Ive) before the fast paths: besselIK on the folded arguments.
func ivOld(v, x float64, scaled bool) float64 {
	sign := 1.0
	if x < 0 {
		x = -x
		if math.Mod(v, 2) != 0 {
			sign = -1
		}
	}
	if v == math.Trunc(v) {
		v = math.Abs(v)
	}
	i, _ := besselIK(v, x, true, false, scaled)
	return float64(sign * i)
}

// TestIvFastBeforeAfter logs, on the rows the fast paths answer, the worst
// error of besselIK (Iv before this change) next to the fast path.
func TestIvFastBeforeAfter(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		oldW, newW := map[string]float64{}, map[string]float64{}
		for _, r := range ivFixtureRows(t, scaled) {
			got, ok := ivFastRow(r, scaled)
			if !ok {
				continue
			}
			old := ivOld(r.args[0], r.args[1], scaled)
			if e := ulpErr(old, r.hi, r.lo); e > oldW[r.region] {
				oldW[r.region] = e
			}
			if e := ulpErr(float64(got*ivSign(r)), r.hi, r.lo); e > newW[r.region] {
				newW[r.region] = e
			}
		}
		for reg := range newW {
			t.Logf("scaled=%v %-24s before %.3f ulp  after %.3f ulp", scaled, reg, oldW[reg], newW[reg])
		}
	}
}

// TestIvFastFallbackRate logs the share of the fixture rows (positive or
// integer order, other than 0 and 1) that the fast paths decline.
func TestIvFastFallbackRate(t *testing.T) {
	for _, scaled := range []bool{false, true} {
		var n, hit int
		byReg := map[string][2]int{}
		for _, r := range ivFixtureRows(t, scaled) {
			v, x := r.args[0], math.Abs(r.args[1])
			if (v < 0 && v != math.Trunc(v)) || math.IsInf(v, 0) || math.Abs(v) <= 1 || !(x > 0) || x > math.MaxFloat64 || math.IsNaN(v) {
				continue
			}
			n++
			c := byReg[r.region]
			c[1]++
			if _, ok := ivFastRow(r, scaled); ok {
				hit++
				c[0]++
			}
			byReg[r.region] = c
		}
		t.Logf("scaled=%v: %d of %d rows with order above 1 answered by ivFast (%.1f%% fall back)",
			scaled, hit, n, 100*float64(n-hit)/float64(n))
		for reg, c := range byReg {
			t.Logf("scaled=%v   %-24s %4d of %4d answered", scaled, reg, c[0], c[1])
		}
	}
}

// TestIvDebyeRootRows: the inputs of the review of #78, where a computed
// Debye term vanished at a root of its polynomial, against the math/big
// oracle, for I (the same stopping rule as K).
func TestIvDebyeRootRows(t *testing.T) {
	for _, c := range []struct{ v, x float64 }{
		{32.5, 3.7595746690511485}, {32.5, 62.43318536919962}, {1000, 816.4965809277256},
		{1100, 898.1462390204982}, {300, 576.3063264847694},
	} {
		got, ok := ivFast(c.v, c.x, true)
		if !ok {
			t.Errorf("ivFast(%g, %g) declined", c.v, c.x)
			continue
		}
		if e := ivOracleUlps(got, bigIve(c.v, c.x)); e > ivFastBound {
			t.Errorf("Ive(%g, %g) = %.17g: %.3g ulp from the oracle", c.v, c.x, got, e)
		}
	}
}
