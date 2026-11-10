package special

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"strconv"
	"testing"
)

// erfOld and erfcOld are Erf and Erfc of v0.7.0 (erfcHL: exponential and
// fitted erfcx), the reference the table kernels are compared with.
func erfOld(x float64) float64 {
	ax := math.Abs(x)
	if ax > 0.5 && ax < 6 {
		h, l, k := erfcHLmain(ax, 0)
		s := pow2(k)
		h = float64(h * (s))
		l = float64(l * (s))
		v := 1 - h
		return math.Copysign(v+(((1-v)-h)-l), x)
	}
	return Erf(x)
}

func erfcOld(x float64) float64 {
	switch {
	case x >= 0.5 && x <= 26.5:
		h, l, k := erfcHL(x, 0)
		return float64((h + l) * pow2(k))
	case x <= -0.5 && x > -6:
		h, l, k := erfcHLmain(-x, 0)
		s := pow2(k)
		h = float64(h * (s))
		l = float64(l * (s))
		v := 2 - h
		return v + (((2 - v) - h) - l)
	}
	return Erfc(x)
}

// logNdtrUpOld is LogNdtr(x) for 3/4 <= x <= 36 with Φ(-x) from the previous
// erfcHL (the branch log1p(-Φ(-x)) of LogNdtr).
func logNdtrUpOld(x float64) float64 {
	zh, zl := ndtrZHL(x)
	h, l, k := erfcHL(-zh, -zl)
	s := pow2(k - 1)
	h = float64(h * (s))
	l = float64(l * (s))
	if h < 0x1p-20 {
		return -h - (l + float64(h*h*math.FMA(h, math.FMA(h, 0.25, 1.0/3), 0.5)))
	}
	wh := 1 - h
	wl := ((1 - wh) - h) - l
	lh, ll := logHLn(wh)
	return lh + (ll + wl/wh)
}

// TestErfTableAgainstOld compares Erf, Erfc and Ndtr with the previous
// kernels: within 1 ulp everywhere, bit-equal except for a rate of flips
// (stated in the PR and release notes; measured with ERF_SWEEP=<points per
// function>). The default sweep's output hash must be the same on every
// architecture (the 8 CI lanes run this).
//
// Big sweep (ERF_SWEEP=1000000): see the PR.
const erfTableSweepHash = "444930e66f02107eaa0addafffc545efae2e4c7caeb6ed52431cfc573b8fe859"

func TestErfTableAgainstOld(t *testing.T) {
	n := 20000
	if s := os.Getenv("ERF_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	x := uint64(0x9e3779b97f4a7c15)
	rnd := func() float64 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		return math.Ldexp(float64(x>>11), -53)
	}
	h := sha256.New()
	rows := []struct {
		name     string
		new, old func(float64) float64
		lo, hi   float64
	}{
		{"Erf", Erf, erfOld, -6, 6},
		{"Erfc", Erfc, erfcOld, -6, 25},
		{"Ndtr", Ndtr, ndtrOld, -36, 8.5},
		{"LogNdtr", LogNdtr, logNdtrOld, -100, -8}, // the lower tail; x > 0.75 is checked below
		{"LogNdtrUp", LogNdtr, logNdtrUpOld, 0.75, 36},
	}
	for _, r := range rows {
		flips, worst := 0, 0.0
		for i := 0; i < n; i++ {
			// uniform, and denser towards lo
			for _, v := range []float64{r.lo + float64((r.hi-r.lo)*rnd()), r.lo + float64(float64((r.hi-r.lo)*rnd())*rnd())} {
				a, b := r.new(v), r.old(v)
				binary.Write(h, binary.LittleEndian, a)
				if a != b {
					flips++
					if u := math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b); u > worst {
						worst = u
					}
				}
			}
		}
		t.Logf("%s: %d points, %d flips (%.3g), worst %.2f ulp", r.name, 2*n, flips, float64(flips)/float64(2*n), worst)
		if worst > 1 {
			t.Fatalf("%s differs from the previous kernel by %v ulp", r.name, worst)
		}
	}
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != erfTableSweepHash {
		t.Fatalf("Erf/Erfc/Ndtr sweep hash %s, want %s", got, erfTableSweepHash)
	}
	// the ends of every table's range and their neighbours: within 1 ulp
	for _, c := range []struct {
		name     string
		new, old func(float64) float64
		x        float64
	}{
		{"Erf", Erf, erfOld, 0.5}, {"Erf", Erf, erfOld, 6}, {"Erfc", Erfc, erfcOld, 0.5}, {"Erfc", Erfc, erfcOld, 25},
		{"Erfc", Erfc, erfcOld, -0.5}, {"Ndtr", Ndtr, ndtrOld, -0.75}, {"Ndtr", Ndtr, ndtrOld, -36}, {"Ndtr", Ndtr, ndtrOld, 8.4999},
		{"LogNdtr", LogNdtr, logNdtrOld, -8}, {"LogNdtr", LogNdtr, logNdtrOld, -2047.9}, {"LogNdtr", LogNdtr, logNdtrUpOld, 0.75}, {"LogNdtr", LogNdtr, logNdtrUpOld, 36},
	} {
		for _, x := range []float64{c.x, math.Nextafter(c.x, 0), math.Nextafter(c.x, math.Inf(1)), math.Nextafter(c.x, math.Inf(-1))} {
			a, b := c.new(x), c.old(x)
			if u := math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b); u > 1 {
				t.Errorf("%s(%v) = %v, previous kernel %v (%v ulp)", c.name, x, a, b, u)
			}
		}
	}
}

func TestErfTablesShape(t *testing.T) {
	for name, p := range map[string][]polyPiece{"erfc": erfcPieces[:], "ndtrNeg": ndtrNegPieces[:], "ndtrPos": ndtrPosPieces[:]} {
		if d := maxDegree(p); d > 16 {
			t.Errorf("%s: degree %d > 16", name, d)
		}
	}
	// the indices stay inside the tables
	for _, x := range []float64{0.5, 3.9999999, 4, 24.999999999, 25} {
		z2 := float64(x * x)
		if i := min(int(4*z2), int(z2)+48); i < 1 || i >= len(erfcTabEst) {
			t.Errorf("erfc index %d at %v, table %d", i, x, len(erfcTabEst))
		}
	}
	for _, x := range []float64{0.75, 3.9999999, 5.7, 36} {
		z2 := float64(x * x)
		if i := min(int(2*z2), int(0.5*z2)+48); i < 1 || i >= len(ndtrNegTabEst) {
			t.Errorf("ndtr index %d at %v, table %d", i, x, len(ndtrNegTabEst))
		}
	}
	top := math.Nextafter(8.5, 0)
	if i := int((top + 0.75) * 8); i >= len(ndtrPosTabEst) {
		t.Errorf("ndtrPos index %d, table %d", i, len(ndtrPosTabEst))
	}
}
