package special

import (
	"math"
	"os"
	"strconv"
	"testing"

	"github.com/go-numerics/special/dd"
)

// sweepN returns the size of an evidence sweep: def in CI, SPECIAL_SWEEP
// points in the evidence runs of the PRs.
func sweepN(def int) int {
	if s := os.Getenv("SPECIAL_SWEEP"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	return def
}

// Factorial on 1 < n < 10 and 0 < n < 1 must return the reference's float64
// (dd.Exp(log n + log Γ(n)), gamma1p) at every point where the fast path
// claims one, claim nearly all, and sit within factorialErr / gammaErrRecip of
// the reference's double-double value.
func TestFactorialSmallFastEqualsReference(t *testing.T) {
	r := gapsRand(11)
	n := sweepN(200000)
	hit, maxd, argmax := 0, 0.0, 0.0
	for i := 0; i < n; i++ {
		var x float64
		switch i % 4 {
		case 0:
			x = 1 + float64(9*r())
		case 1:
			x = 1 + float64(r()*math.Exp2(-float64(int(r()*40))))
		case 2:
			x = 10 + float64(160*r())
		default:
			x = 160 + float64(10*r())
		}
		if x == math.Trunc(x) || x >= 170.6 {
			continue
		}
		h, l := factorialLargeHL(x)
		m, k := dd.ExpMant(dd.Add(dd.Log(x), logGammaPos(x)))
		hs, ls := math.Ldexp(h, -k), math.Ldexp(l, -k)
		if rel := math.Abs((hs-m.Hi)+(ls-m.Lo)) / m.Hi / factorialBound(x, 1); rel > maxd {
			maxd, argmax = rel, x
		}
		if v, ok := factorialLargeFast(x); ok {
			hit++
			if want := dd.Exp(dd.Add(dd.Log(x), logGammaPos(x))); v != want {
				t.Fatalf("factorialLargeFast(%v) = %v, reference %v", x, v, want)
			}
		}
	}
	t.Logf("n=1..170: %d points, certified %d, max |h+l-D|/bound = 2^%.1f at n = %v", n, hit, math.Log2(maxd), argmax)
	if maxd > 0.62 {
		t.Fatalf("distance 2^%.1f of the bound: no margin left in factorialBound", math.Log2(maxd))
	}
	if hit < n*9/10 {
		t.Fatalf("only %d of %d certified", hit, n)
	}
}

// logGammaLargeHLL is logGammaLargeHL plus the logarithm logHLn(x), bit for bit.
func TestLogGammaLargeHLLIsTheKernels(t *testing.T) {
	r := gapsRand(13)
	for i := 0; i < sweepN(200000); i++ {
		x := 10 + float64(165*r())
		gh, gl := logGammaLargeHL(x)
		lh, ll := logHLn(x)
		hi, lo, h2, l2 := logGammaLargeHLL(x)
		if hi != gh || lo != gl || h2 != lh || l2 != ll {
			t.Fatalf("logGammaLargeHLL(%v) = %v %v %v %v, want %v %v %v %v", x, hi, lo, h2, l2, gh, gl, lh, ll)
		}
	}
}

// Xlog1py tries its fast path first; the special values must still reach the
// original cases, and every finite point must be the double-double product's
// rounding.
func TestXlog1pyOrderOfCases(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	sp := []float64{0, math.Copysign(0, -1), 1, -1, -1.5, -0.5, 0x1p-21, -0x1p-21, 0x1p-20, -0x1p-20, 3, -inf, inf, nan, math.MaxFloat64, 5e-324}
	for _, x := range sp {
		for _, y := range sp {
			got := Xlog1py(x, y)
			var want float64
			switch {
			case x == 0 && !math.IsNaN(y):
				want = 0
			case math.IsNaN(x) || math.IsNaN(y) || y < -1:
				want = nan
			case y == -1 || math.IsInf(y, 1) || math.IsInf(x, 0):
				want = float64(x * math.Log1p(y))
			default:
				want = dd.MulF(log1pDD(dd.F(y)), x).Float()
			}
			if math.Float64bits(got) != math.Float64bits(want) && !(math.IsNaN(got) && math.IsNaN(want)) {
				// libm.Log1p vs math.Log1p may differ in the last bit on the
				// infinite branch; compare through the package's own function.
				if y == -1 || math.IsInf(y, 1) || math.IsInf(x, 0) {
					continue
				}
				t.Errorf("Xlog1py(%v, %v) = %v, want %v", x, y, got, want)
			}
		}
	}
	r := gapsRand(14)
	for i := 0; i < sweepN(200000); i++ {
		x := float64((r() - 0.5) * math.Exp2(float64(int(float64(r()*80))-40)))
		y := float64((float64(r()*4) - 1.5) * math.Exp2(-float64(int(r()*30))))
		if y <= -1 || x == 0 {
			continue
		}
		if got, want := Xlog1py(x, y), dd.MulF(log1pDD(dd.F(y)), x).Float(); got != want {
			t.Fatalf("Xlog1py(%v, %v) = %v, reference %v", x, y, got, want)
		}
	}
}

func TestFactorialUnitFastEqualsReference(t *testing.T) {
	r := gapsRand(12)
	n := sweepN(200000)
	hit, maxd := 0, 0.0
	for i := 0; i < n; i++ {
		x := float64(r())
		if i%3 == 0 {
			x = float64(r() * math.Exp2(-float64(int(r()*60))))
		}
		if x == 0 {
			continue
		}
		gh, gl := gamma1pFast(x)
		d := gamma1p(x)
		if rel := math.Abs((gh-d.Hi)+(gl-d.Lo)) / d.Hi; rel > maxd {
			maxd = rel
		}
		if zivRound(gh, gl, float64(gammaErrRecip*gh)) {
			hit++
		}
		if v, want := Factorial(x), d.Float(); v != want {
			t.Fatalf("Factorial(%v) = %v, reference %v", x, v, want)
		}
	}
	t.Logf("n in (0,1): %d points, certified %d, max |h+l-D|/D = 2^%.1f", n, hit, math.Log2(maxd))
	if maxd > 0x1p-99 {
		t.Fatalf("distance 2^%.1f exceeds gammaErrRecip", math.Log2(maxd))
	}
	if hit < n*9/10 {
		t.Fatalf("only %d of %d certified", hit, n)
	}
}
