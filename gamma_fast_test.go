package special

import (
	"math"
	"math/rand/v2"
	"testing"
)

// gammaReference is Gamma as it was before the fast paths: the
// double-double evaluation on 0 < x < 1 and x < 0.
func gammaReference(x float64) float64 {
	switch {
	case math.IsNaN(x) || math.IsInf(x, -1):
		return math.NaN()
	case math.IsInf(x, 1):
		return x
	case x == 0:
		return math.Copysign(math.Inf(1), x)
	}
	if x == math.Trunc(x) {
		if x < 0 {
			return math.NaN()
		}
		if x <= 171 {
			return factorials[int(x)-1]
		}
		return math.Inf(1)
	}
	if x > 0 {
		if x >= 10 {
			return gammaLarge(x)
		}
		if x >= 1 {
			h, l := gammaMid(x)
			return h + l
		}
		return gammaPos(x).Float()
	}
	return gammaNegDD(x)
}

func sameBits(a, b float64) bool {
	return math.Float64bits(a) == math.Float64bits(b) || (math.IsNaN(a) && math.IsNaN(b))
}

// gammaRegion is one region of a fast path with the points that probe it.
type gammaRegion struct {
	name  string
	fast  func(float64) (float64, bool)
	slow  func(float64) float64
	gen   func(r *rand.Rand, i int) float64
	count int
}

// nearPole returns a point within a few hundred ulps, or a random power of
// two, of the negative integer -k (k in [lo, hi]).
func nearPole(r *rand.Rand, lo, hi int) float64 {
	k := float64(lo + r.IntN(hi-lo+1))
	if r.IntN(2) == 0 {
		d := math.Ldexp(1, -1-r.IntN(52))
		if r.IntN(2) == 0 {
			d = -d
		}
		return -k + d
	}
	j := 1 + r.IntN(512)
	if r.IntN(2) == 0 {
		return math.Nextafter(-k, math.Inf(1)) + float64(j-1)*ulp(k)
	}
	return math.Nextafter(-k, math.Inf(-1)) - float64(j-1)*ulp(k)
}

func logUniform(r *rand.Rand, e0, e1 int) float64 {
	return math.Ldexp(1+r.Float64(), e0+r.IntN(e1-e0))
}

func gammaRegions() []gammaRegion {
	unitSlow := func(x float64) float64 { return gammaPos(x).Float() }
	return []gammaRegion{
		{"(2^-56, 1)", gammaRecipFast, unitSlow, func(r *rand.Rand, i int) float64 {
			if i%2 == 0 {
				return logUniform(r, -56, 0) * 0.5 // [2^-56, 1)
			}
			return r.Float64()
		}, 2_000_000},
		{"(-1/2, -2^-56)", gammaRecipFast, gammaNegDD, func(r *rand.Rand, i int) float64 {
			if i%2 == 0 {
				return -logUniform(r, -56, -1) * 0.5 * 0.999
			}
			return -0.5 * (1 - r.Float64())
		}, 2_000_000},
		{"(-34, -1/2)", gammaProdFast, gammaNegDD, func(r *rand.Rand, i int) float64 {
			if i%4 == 0 {
				return nearPole(r, 1, 33)
			}
			return -0.5 - 33.5*r.Float64()
		}, 3_000_000},
		{"(-171.6, -34)", gammaReflFast, gammaNegDD, func(r *rand.Rand, i int) float64 {
			if i%4 == 0 {
				return nearPole(r, 35, 171)
			}
			return -34 - 137.6*r.Float64()
		}, 3_000_000},
		{"(-190, -171.6)", gammaReflFast, gammaNegDD, func(r *rand.Rand, i int) float64 {
			if i%4 == 0 {
				return nearPole(r, 172, 189)
			}
			return -171.6 - 18.4*r.Float64()
		}, 1_000_000},
	}
}

// TestGammaFastEqualsReference checks fast == reference bit for bit on
// 10^7 points (2·10^5 with -short or -race), Gamma == reference on the same points,
// and logs the fallback rate of each region.
func TestGammaFastEqualsReference(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	total := 0
	for _, reg := range gammaRegions() {
		n := reg.count
		if testing.Short() || raceEnabled {
			n /= 50
		}
		fallbacks, tested, bad := 0, 0, 0
		for i := 0; i < n; i++ {
			x := reg.gen(rng, i)
			if x == math.Trunc(x) {
				continue
			}
			tested++
			want := reg.slow(x)
			got, ok := reg.fast(x)
			if !ok {
				fallbacks++
			} else if !sameBits(got, want) {
				if bad < 10 {
					t.Errorf("%s: fast(%v) = %v, reference %v", reg.name, x, got, want)
				}
				bad++
			}
			if g, w := Gamma(x), gammaReference(x); !sameBits(g, w) {
				if bad < 10 {
					t.Errorf("%s: Gamma(%v) = %v, reference %v", reg.name, x, g, w)
				}
				bad++
			}
		}
		total += tested
		t.Logf("%-16s %9d points, %8d fallbacks (%.4f%%), %d mismatches",
			reg.name, tested, fallbacks, 100*float64(fallbacks)/float64(tested), bad)
	}
	t.Logf("total %d points", total)
}

// TestGammaReflSubnormal checks the region where Γ(x) leaves the normal
// range, on a dense grid: fast == reference wherever the fast path
// answers, with subnormal results and zeros among the answers.
func TestGammaReflSubnormal(t *testing.T) {
	var subnormal, zero, declined int
	for x := -171.0; x > -190; x -= 0x1p-10 * 1.0001 {
		v, ok := gammaReflFast(x)
		w := gammaNegDD(x)
		switch {
		case !ok:
			declined++
		case !sameBits(v, w):
			t.Errorf("gammaReflFast(%v) = %v, reference %v", x, v, w)
		case v == 0:
			zero++
		case math.Abs(v) < 0x1p-1022:
			subnormal++
		}
	}
	t.Logf("(-190, -171): %d subnormal, %d zero, %d declined", subnormal, zero, declined)
	if subnormal == 0 || zero == 0 {
		t.Error("the grid did not reach subnormal and zero results")
	}
}

// TestMod2 checks mod2 == math.Mod(·, 2) bit for bit: random magnitudes
// from subnormal to MaxFloat64, and the neighbours of every power of two.
func TestMod2(t *testing.T) {
	check := func(a float64) {
		if got, want := mod2(a), math.Mod(a, 2); !sameBits(got, want) {
			t.Fatalf("mod2(%v) = %v, math.Mod %v", a, got, want)
		}
	}
	rng := rand.New(rand.NewPCG(3, 4))
	n := 1_000_000
	if testing.Short() || raceEnabled {
		n /= 50
	}
	for i := 0; i < n; i++ {
		check(math.Float64frombits(rng.Uint64() &^ (1 << 63) % 0x7ff0000000000000))
		check(float64(rng.IntN(1<<20)) + rng.Float64())
	}
	for e := -1074; e <= 1023; e++ {
		p := math.Ldexp(1, e)
		check(p)
		check(math.Nextafter(p, 0))
		check(math.Nextafter(p, math.Inf(1)))
		check(3 * p)
	}
	check(0)
	check(math.MaxFloat64)
}

func FuzzMod2(f *testing.F) {
	for _, a := range []float64{0, 1, 1.5, 2, 3.75, 0x1p52 + 1, 0x1p53 + 2, 5e-324} {
		f.Add(a)
	}
	f.Fuzz(func(t *testing.T, a float64) {
		a = math.Abs(a)
		if math.IsInf(a, 0) || math.IsNaN(a) {
			return
		}
		if got, want := mod2(a), math.Mod(a, 2); !sameBits(got, want) {
			t.Errorf("mod2(%v) = %v, math.Mod %v", a, got, want)
		}
	})
}

func TestZivRound(t *testing.T) {
	h := 1.5
	half := ulp(h) / 2
	for _, c := range []struct {
		h, l, e float64
		want    bool
	}{
		{h, 0, 0, true},
		{h, half / 2, half / 4, true},
		{h, half / 2, half / 2, false}, // reaches the midpoint
		{h, -half, 0, false},           // a tie
		{1, 0, 0, true},                // power of two: cells ulp/2 above,
		{1, 0x1p-53 * 0.9, 0, true},    // ulp/4 below
		{1, -0x1p-54 * 0.9, 0, true},   //
		{1, -0x1p-54, 0, false},        // the tie below 1
		{1, -0x1p-54 * 0.9, 0x1p-56, false},
		{-1, 0x1p-54 * 0.9, 0, true}, // mirrored for negative h
		{-1, 0x1p-54, 0, false},
		{-1, -0x1p-53 * 0.9, 0, true},
		{-1.5, 0, 0x1p-60, true},
		{0x1p-1000 * 1.5, 0, 0, false}, // too close to the subnormal range
		{math.Inf(1), 0, 0, false},
		{math.NaN(), 0, 0, false},
	} {
		if got := zivRound(c.h, c.l, c.e); got != c.want {
			t.Errorf("zivRound(%v, %v, %v) = %v, want %v", c.h, c.l, c.e, got, c.want)
		}
	}
}

// FuzzGammaFast: Gamma equals the double-double reference bit for bit.
func FuzzGammaFast(f *testing.F) {
	for _, x := range []float64{0.3, 1e-10, -1e-10, -0.25, -0.5, -1.5, -33.9, -34.5, -100.25, -171.5, -185.3, 5.5, 150.5} {
		f.Add(x)
	}
	f.Fuzz(func(t *testing.T, x float64) {
		if g, w := Gamma(x), gammaReference(x); !sameBits(g, w) {
			t.Errorf("Gamma(%v) = %v, reference %v", x, g, w)
		}
	})
}
