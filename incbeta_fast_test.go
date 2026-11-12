package special

import (
	"math"
	"testing"
)

// BenchmarkBetaIncBox is the refbench box of #33: a, b in (0.5, 50), x in
// (0.01, 0.99), 10000 points; ns/op / 10000 = ns per element.
func BenchmarkBetaIncBox(b *testing.B) {
	const n = 10000
	r := gapsRand(1)
	pa, pb, px := make([]float64, n), make([]float64, n), make([]float64, n)
	pp := make([]float64, n)
	for i := range pa {
		pa[i] = 0.5 + float64(r()*49.5)
		pb[i] = 0.5 + float64(r()*49.5)
		px[i] = 0.01 + float64(r()*0.98)
		pp[i] = 0.01 + float64(r()*0.98)
	}
	for _, f := range []struct {
		name string
		f    func(a, b, x float64) float64
		p    bool
	}{{"BetaInc", BetaInc, false}, {"BetaIncC", BetaIncC, false}, {"BetaIncInv", BetaIncInv, true}, {"BetaIncCInv", BetaIncCInv, true}} {
		b.Run(f.name, func(b *testing.B) {
			var s float64
			for j := 0; j < b.N; j++ {
				for i := 0; i < n; i++ {
					x := px[i]
					if f.p {
						x = pp[i]
					}
					s += f.f(pa[i], pb[i], x)
				}
			}
			besselSink = s
		})
	}
}

func relDiff(a, b float64) float64 {
	if a == b {
		return 0
	}
	return math.Abs(a-b) / math.Abs(b)
}

func ulpDiff(a, b float64) float64 {
	if a == b {
		return 0
	}
	u := math.Abs(math.Nextafter(b, math.Inf(1)) - b)
	return math.Abs(a-b) / u
}

// The float64 prefix against the double-double prefix: it moves the prefix by
// up to 2^-56, so BetaInc and BetaIncC differ only where the dd value lies
// within that of a rounding boundary: the flip rate (results that differ at
// all) of each box must stay under a ceiling about three times the measured
// rate (0.067 %, 0.056 %, 0.014 %, 0.24 %), which a mutated ψ(a + b) term
// (ψ × 0.75 below 10: 10 % of the values) exceeds by far. Worst relative
// difference 2^-50: the complement 1 - w and the Bup + Bgrat sums amplify the
// prefix difference (up to 4 ulp in (0.01, 3)); accuracy against mpmath is the
// fixtures' (regions "a,b in [0.5,3)" ... "[10,85]").
func TestBetaIncFastPrefixVsDD(t *testing.T) {
	boxes := []struct {
		name         string
		alo, ahi     float64
		blo, bhi     float64
		logUniformAB bool
		ceilPercent  float64
	}{
		{"refbench (0.5,50)", 0.5, 50, 0.5, 50, false, 0.25},
		{"(1,85)", 1, 85, 1, 85, false, 0.25},
		{"(2^-20,85) log", 0x1p-20, 85, 0x1p-20, 85, true, 0.1},
		{"(0.01,3)", 0.01, 3, 0.01, 3, false, 0.8},
	}
	r := gapsRand(21)
	n := sweepN(2500) // CI: the suite is near its timeout under -race on Windows
	for _, bx := range boxes {
		flips, total, worst := 0, 0, 0.0
		for i := 0; i < n; i++ {
			var a, b float64
			if bx.logUniformAB {
				a = float64(bx.alo * math.Exp2(float64(float64(r())*math.Log2(bx.ahi/bx.alo))))
				b = float64(bx.blo * math.Exp2(float64(float64(r())*math.Log2(bx.bhi/bx.blo))))
			} else {
				a = bx.alo + float64(r()*(bx.ahi-bx.alo))
				b = bx.blo + float64(r()*(bx.bhi-bx.blo))
			}
			x := float64(0.001 + float64(r()*0.998))
			if i%7 == 0 {
				x = float64(r() * math.Exp2(-float64(int(r()*40)))) // small x: exponents beyond the kernel's range decline
			}
			if x == 0 {
				continue
			}
			ibetaNoFast = true
			w0, c0 := BetaInc(a, b, x), BetaIncC(a, b, x)
			ibetaNoFast = false
			w1, c1 := BetaInc(a, b, x), BetaIncC(a, b, x)
			total += 2
			for _, p := range [][2]float64{{w1, w0}, {c1, c0}} {
				if p[0] != p[1] {
					flips++
				}
				if u := ulpDiff(p[0], p[1]); u > worst {
					worst = u
				}
				if math.IsNaN(p[0]) != math.IsNaN(p[1]) {
					t.Fatalf("NaN mismatch at a=%v b=%v x=%v", a, b, x)
				}
			}
			// The prefixes differ by under 2^-56 (both are about 2^-58 from the
			// truth: the float64 Stirling series); the complement formed as
			// 1 - w and the Bup + Bgrat sums amplify that, never beyond 2^-50.
			if relDiff(w1, w0) > 0x1p-50 || relDiff(c1, c0) > 0x1p-50 {
				t.Fatalf("a=%v b=%v x=%v: fast %v %v, dd %v %v", a, b, x, w1, c1, w0, c0)
			}
		}
		ibetaNoFast = false
		rate := 100 * float64(flips) / float64(total)
		t.Logf("%s: %d values, flips %.4f%%, worst %.3f ulp from the dd prefix", bx.name, total, rate, worst)
		if rate > bx.ceilPercent {
			t.Errorf("%s: flip rate %.4f%% above its ceiling %.2f%%", bx.name, rate, bx.ceilPercent)
		}
	}
}
