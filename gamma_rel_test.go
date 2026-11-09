package special

import (
	"math"
	"testing"
)

// gammaRel against the double-double Γ (gammaPos: Stirling in double-double,
// 2^-90 relative) on random points of [10, gammaMax]: the error in ulps of
// the new kernel and of the old (gammaLargeStirling), and how often the two
// round differently. Worst ulp error must not exceed the old kernel's.
func TestGammaRelAccuracy(t *testing.T) {
	n := 2000000
	if testing.Short() || raceEnabled {
		n = 100000
	}
	r := gapsRand(11)
	var worstNew, worstOld float64
	flips, infOld := 0, 0
	for i := 0; i < n; i++ {
		x := 10 + float64(r()*(gammaMax-10))
		if x == math.Trunc(x) {
			continue
		}
		ref := gammaPos(x)
		ulp := math.Nextafter(ref.Hi, math.Inf(1)) - ref.Hi
		gn, go_ := gammaRel(x), gammaLargeStirling(x)
		en := math.Abs((gn-ref.Hi)-ref.Lo) / ulp
		eo := math.Abs((go_-ref.Hi)-ref.Lo) / ulp
		if math.IsInf(go_, 0) {
			infOld++ // the old kernel overflowed (x within 2^-12 of gammaMax)
			eo = 0
		}
		worstNew, worstOld = math.Max(worstNew, en), math.Max(worstOld, eo)
		if gn != go_ {
			flips++
		}
	}
	t.Logf("n=%d: worst error new %.4f ulp, old %.4f ulp (%d old overflows); outputs differ at %d points (%.3f%%)", n, worstNew, worstOld, infOld, flips, float64(100*flips)/float64(n))
	if worstNew > worstOld+0.002 {
		t.Fatalf("new kernel worst %.4f ulp > old %.4f ulp", worstNew, worstOld)
	}
}

// lgammaOct against the double-double log Γ (logGammaStirling, 2^-70
// absolute at worst here) on [10, 1024): worst error in ulps, new against
// old (logGammaLargeHL, rounded), and the number of different outputs.
func TestLgammaOctAccuracy(t *testing.T) {
	n := 2000000
	if testing.Short() || raceEnabled {
		n = 100000
	}
	r := gapsRand(13)
	var worstNew, worstOld float64
	flips := 0
	for i := 0; i < n; i++ {
		x := 10 + float64(r()*1014)
		ref := logGammaStirling(x)
		ulp := math.Nextafter(ref.Hi, math.Inf(1)) - ref.Hi
		gn := lgammaOct(x)
		oh, ol := logGammaLargeHL(x)
		go_ := oh + ol
		en := math.Abs((gn-ref.Hi)-ref.Lo) / ulp
		eo := math.Abs((go_-ref.Hi)-ref.Lo) / ulp
		worstNew, worstOld = math.Max(worstNew, en), math.Max(worstOld, eo)
		if gn != go_ {
			flips++
		}
	}
	t.Logf("n=%d: worst error new %.4f ulp, old %.4f ulp; outputs differ at %d points (%.3f%%)", n, worstNew, worstOld, flips, float64(100*flips)/float64(n))
	if worstNew > worstOld+0.002 {
		t.Fatalf("new kernel worst %.4f ulp > old %.4f ulp", worstNew, worstOld)
	}
}
