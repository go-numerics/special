package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/internal/dd"
)

func dd1(x float64) dd.DD { return dd.F(x) }

// Edge values of the incomplete beta family (SciPy's conventions).
func TestIncBetaEdges(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	same := func(got, want float64) bool {
		return got == want || (math.IsNaN(got) && math.IsNaN(want))
	}
	cs := []struct{ a, b, x, w, w1 float64 }{
		{nan, 1, 0.5, nan, nan}, {1, -1, 0.5, nan, nan}, {1, 1, 1.5, nan, nan},
		{0, 0, 0.5, nan, nan}, {inf, inf, 0.5, nan, nan}, {0, 2, 0, 0, 1}, {0, 2, 0.5, 1, 0},
		{2, inf, 0.5, 1, 0}, {2, 0, 0.5, 0, 1}, {2, 0, 1, 1, 0}, {inf, 2, 0.5, 0, 1},
		{2, 3, 0, 0, 1}, {2, 3, 1, 1, 0},
	}
	for _, c := range cs {
		if w, w1 := BetaInc(c.a, c.b, c.x), BetaIncC(c.a, c.b, c.x); !same(w, c.w) || !same(w1, c.w1) {
			t.Errorf("I,Ic(%g, %g, %g) = %g, %g, want %g, %g", c.a, c.b, c.x, w, w1, c.w, c.w1)
		}
	}
	inv := []struct{ a, b, p, x, xc float64 }{
		{nan, 1, 0.5, nan, nan}, {0, 1, 0.5, nan, nan}, {1, 1, -1, nan, nan},
		{2, 3, 0, 0, 1}, {2, 3, 1, 1, 0}, {inf, 3, 0.5, nan, nan}, {2, 2, 0.5, 0.5, 0.5},
	}
	for _, c := range inv {
		if x, xc := BetaIncInv(c.a, c.b, c.p), BetaIncCInv(c.a, c.b, c.p); !same(x, c.x) || !same(xc, c.xc) {
			t.Errorf("Inv,CInv(%g, %g, %g) = %g, %g, want %g, %g", c.a, c.b, c.p, x, xc, c.x, c.xc)
		}
	}
	// BASYM's Gaussian factor underflowing outright
	if v := BetaInc(1e15, 1e15, 0.4999); v != 0 {
		t.Errorf("BetaInc(1e15, 1e15, 0.4999) = %g", v)
	}
}

// Round trips over a grid reaching the branches of the inverse.
func TestIncBetaInverseGrid(t *testing.T) {
	ps := []float64{1e-300, 1e-20, 1e-5, 0.01, 0.3, 0.5, 0.7, 0.99, 1 - 1e-10}
	ab := []float64{1e-200, 1e-5, 0.1, 0.5, 1, 2.5, 20, 1e3, 1e8}
	for _, a := range ab {
		for _, b := range ab {
			for _, p := range ps {
				x, xc := BetaIncInv(a, b, p), BetaIncCInv(a, b, p)
				if !(x >= 0 && x <= 1) || !(xc >= 0 && xc <= 1) {
					t.Errorf("inverse(%g, %g, %g) = %g, %g", a, b, p, x, xc)
				}
			}
		}
	}
}

func BenchmarkBetaInc(b *testing.B) {
	args := [][3]float64{{0.5, 0.5, 0.3}, {2, 3, 0.4}, {50, 60, 0.45}, {1e3, 2e3, 0.34}, {0.5, 1e5, 1e-6}, {30, 0.5, 0.9}, {1e5, 1e5, 0.5011}}
	for i := 0; i < b.N; i++ {
		for _, c := range args {
			sinkF = BetaInc(c[0], c[1], c[2])
		}
	}
}

func BenchmarkBetaIncInv(b *testing.B) {
	args := [][3]float64{{0.5, 0.5, 0.3}, {2, 3, 0.4}, {50, 60, 0.45}, {1e3, 2e3, 0.01}, {0.5, 1e5, 0.6}, {30, 0.5, 0.9}, {1e5, 1e5, 0.5011}}
	for i := 0; i < b.N; i++ {
		for _, c := range args {
			sinkF = BetaIncInv(c[0], c[1], c[2])
		}
	}
}
