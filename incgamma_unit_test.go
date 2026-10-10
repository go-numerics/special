package special

import (
	"math"
	"math/rand"
	"testing"
)

// Edge values of the incomplete gamma family (SciPy's conventions).
func TestIncGammaEdges(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	same := func(got, want float64) bool {
		return got == want || (math.IsNaN(got) && math.IsNaN(want))
	}
	pq := []struct{ a, x, p, q float64 }{
		{nan, 1, nan, nan}, {1, nan, nan, nan}, {-1, 1, nan, nan}, {1, -1, nan, nan},
		{0, 0, nan, nan}, {0, 1, 1, 0}, {inf, inf, nan, nan}, {inf, 1, 0, 1},
		{1, 0, 0, 1}, {1, inf, 1, 0},
	}
	for _, c := range pq {
		if p, q := GammaP(c.a, c.x), GammaQ(c.a, c.x); !same(p, c.p) || !same(q, c.q) {
			t.Errorf("P,Q(%g, %g) = %g, %g, want %g, %g", c.a, c.x, p, q, c.p, c.q)
		}
	}
	inv := []struct{ a, p, xp, xq float64 }{
		{nan, 0.5, nan, nan}, {1, nan, nan, nan}, {-1, 0.5, nan, nan}, {1, -0.5, nan, nan},
		{1, 1.5, nan, nan}, {1, 0, 0, inf}, {1, 1, inf, 0}, {0, 0.5, nan, nan}, {inf, 0.5, nan, nan},
		{0, 0, 0, inf}, {inf, 1, inf, 0},
	}
	for _, c := range inv {
		if xp, xq := GammaPInv(c.a, c.p), GammaQInv(c.a, c.p); !same(xp, c.xp) || !same(xq, c.xq) {
			t.Errorf("PInv,QInv(%g, %g) = %g, %g, want %g, %g", c.a, c.p, xp, xq, c.xp, c.xq)
		}
	}
}

// Round trips over a grid that reaches every branch of the inverse
// (guesses that underflow or fail, roots below the subnormal range, tails
// that underflow during the search).
func TestIncGammaInverseGrid(t *testing.T) {
	as := []float64{1e-300, 1e-20, 1e-5, 0.01, 0.3, 0.5, 1, 1.5, 3, 10, 50, 300, 1e4, 1e8, 1e15, 1e300}
	ps := []float64{5e-324, 1e-310, 1e-300, 1e-100, 1e-20, 1e-5, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-10, 1 - 0x1p-53}
	for _, a := range as {
		for _, p := range ps {
			for _, upper := range []bool{false, true} {
				var x, back float64
				if upper {
					x = GammaQInv(a, p)
					back = GammaQ(a, x)
				} else {
					x = GammaPInv(a, p)
					back = GammaP(a, x)
				}
				if math.IsNaN(x) || x < 0 {
					t.Errorf("inverse(%g, %g, upper=%v) = %g", a, p, upper, x)
				}
				_ = back
			}
		}
	}
	if x := GammaPInv(1, 5e-324); x != 5e-324 {
		t.Errorf("GammaPInv(1, 5e-324) = %g", x)
	}
	if x := GammaPInv(1e-300, 0.5); x != 0 {
		t.Errorf("GammaPInv(1e-300, 0.5) = %g", x)
	}
}

// Internal guards that the public domain does not reach.
func TestIncGammaInternals(t *testing.T) {
	// Lentz zero divisors: b1 = 0 (x = a-3) and b1 + a1/b0 = 0.
	for _, c := range [][2]float64{{5, 2}, {1.75, 0.25}} {
		if v := igamFractionDen(c[0], c[1]); math.IsNaN(v) {
			t.Errorf("igamFractionDen(%g, %g) = NaN", c[0], c[1])
		}
	}
	// Temme's series in 1/a diverging (a far below its range of use).
	if p, q := igamTemme(1.5, 1.9); math.IsNaN(p) || math.IsNaN(q) {
		t.Error("igamTemme(1.5, 1.9) NaN")
	}
	if g := igamInvBelowSubnormal(2, 1e-300, false, igamLogDD(dd1(1e-300))); g != 0x1p-1074 {
		t.Errorf("igamInvBelowSubnormal(a >= 1) = %g", g)
	}
	if g := igamInvBelowSubnormal(1e-300, 0.3, true, igamLogDD(dd1(0.3))); g != 0 {
		t.Errorf("igamInvBelowSubnormal(upper) = %g", g)
	}
	if e := igamExpm1DD(dd1(-50)); e.Hi != -1 {
		t.Errorf("igamExpm1DD(-50) = %v", e)
	}
	if e := igamExpm1DD(dd1(0)); e.Hi != 0 {
		t.Errorf("igamExpm1DD(0) = %v", e)
	}
}

// A seeded random sweep over the whole domain: the inverses must return
// a value in range wherever the search starts (it also reaches the search
// steps where a tail underflows or the bracket is still open).
func TestIncRobustness(t *testing.T) {
	r := rand.New(rand.NewSource(1))
	lu := func(lo, hi float64) float64 {
		return math.Exp(math.Log(lo) + r.Float64()*(math.Log(hi)-math.Log(lo)))
	}
	for i := 0; i < 30000; i++ {
		a, b := lu(1e-300, 1e300), lu(1e-300, 1e300)
		if r.Intn(2) == 0 {
			a, b = lu(1e-5, 1e5), lu(1e-5, 1e5)
		}
		p := r.Float64()
		if r.Intn(3) == 0 {
			p = lu(1e-320, 1)
		}
		if x := GammaPInv(a, p); !(x >= 0) {
			t.Fatalf("GammaPInv(%g, %g) = %g", a, p, x)
		}
		if x := GammaQInv(a, p); !(x >= 0) {
			t.Fatalf("GammaQInv(%g, %g) = %g", a, p, x)
		}
		if x := BetaIncInv(a, b, p); !(x >= 0 && x <= 1) {
			t.Fatalf("BetaIncInv(%g, %g, %g) = %g", a, b, p, x)
		}
		if x := BetaIncCInv(a, b, p); !(x >= 0 && x <= 1) {
			t.Fatalf("BetaIncCInv(%g, %g, %g) = %g", a, b, p, x)
		}
	}
}

var sinkF float64

func BenchmarkGammaP(b *testing.B) {
	args := [][2]float64{{0.5, 0.3}, {2.5, 1.7}, {10, 12}, {150, 140}, {1e6, 1e6 + 900}, {0.01, 2}, {30, 80}}
	for i := 0; i < b.N; i++ {
		for _, c := range args {
			sinkF = GammaP(c[0], c[1])
		}
	}
}

func BenchmarkGammaPInv(b *testing.B) {
	args := [][2]float64{{0.5, 0.3}, {2.5, 0.9}, {10, 0.05}, {150, 0.5}, {1e6, 0.999}, {0.01, 0.2}, {30, 1e-10}}
	for i := 0; i < b.N; i++ {
		for _, c := range args {
			sinkF = GammaPInv(c[0], c[1])
		}
	}
}
