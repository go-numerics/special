package special

import (
	"math/big"
	"testing"
)

// TestBigCosReduces holds the argument reduction of the oracle's bigCos (see
// its comment): cos(a + 2 pi k) must equal cos(a), and cos at multiples of pi
// and at pi/2 must be +-1 and 0, to far below a float64 ulp, for arguments as
// large as the Bessel oracles pass (v pi with |v| <= 40.5, and x up to 2000).
// The unreduced 200-term series failed every case with |a| >= 40 pi.
func TestBigCosReduces(t *testing.T) {
	ivOracleInit()
	tiny := bnew().SetMantExp(bf(1), -280) // well below 2^-53, above 320-bit noise at |a| ~ 2^11
	near := func(name string, got, want *big.Float) {
		t.Helper()
		d := bnew().Sub(got, want)
		if d.Abs(d).Cmp(tiny) > 0 {
			t.Errorf("%s: got %s, want %s (difference %s)", name, got.Text('g', 20), want.Text('g', 20), d.Text('g', 5))
		}
	}
	for _, a := range []float64{0.3, 1, 2.5, -1.7} {
		base := bigCos(bf(a))
		for _, k := range []float64{20, 40, -40, 318} { // up to 2000 / (2 pi)
			shifted := bnew().Add(bf(a), bnew().Mul(bf(float64(2*k)), bigPi))
			near("cos(a + 2 pi k)", bigCos(shifted), base)
		}
	}
	for _, n := range []float64{40, 41, -39, 81} {
		want := bf(1)
		if int(n)%2 != 0 {
			want = bf(-1)
		}
		near("cos(n pi)", bigCos(bnew().Mul(bf(n), bigPi)), want)
		half := bnew().Mul(bf(n+0.5), bigPi)
		near("cos((n + 1/2) pi)", bigCos(half), bf(0))
	}
}
