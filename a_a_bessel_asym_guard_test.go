package special

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-numerics/special/dd"
)

// Guards of bessel_asym.go found by `mutsweep -sites all`. Every reference
// is outside the code under test: the bit string of the 2/pi table read one
// digit at a time, pi to 100 digits (Abramowitz-Stegun 3.1.1) in math/big,
// and the standard library's Jn/Yn (FreeBSD msun) for the Hankel phase.

// twoOverPiBit is binary digit k (k >= 1 the k-th after the point) of the
// table, zero for k <= 0 and for k past the table.
func twoOverPiBit(k int) uint64 {
	if k < 1 {
		return 0
	}
	w, b := (k-1)/64, uint((k-1)%64)
	if w >= len(besselTwoOverPiBits) {
		return 0
	}
	return besselTwoOverPiBits[w] >> (63 - b) & 1
}

// TestBesselTwoOverPiWordDigits (bessel_asym.go:22, :29): the word starting
// at digit s holds digits s..s+63, MSB first, digits s <= 0 being zero; s from
// far below the table (shift >= 64) to its last full word.
func TestBesselTwoOverPiWordDigits(t *testing.T) {
	last := 64 * (len(besselTwoOverPiBits) - 1) // last s whose second word is inside the table
	for s := -200; s <= last; s++ {
		var want uint64
		for i := 0; i < 64; i++ {
			want = want<<1 | twoOverPiBit(s+i)
		}
		if got := besselTwoOverPiWord(s); got != want {
			t.Fatalf("besselTwoOverPiWord(%d) = %#x, want %#x", s, got, want)
		}
	}
}

const piDigits100 = "3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798"

// bigTwoOverPi is 2/pi at 600 bits.
func bigTwoOverPi() *big.Float {
	pi, _, err := big.ParseFloat(piDigits100, 10, 600, big.ToNearestEven)
	if err != nil {
		panic(err)
	}
	return new(big.Float).SetPrec(600).Quo(big.NewFloat(2).SetPrec(600), pi)
}

// TestBesselReduceAgainstBig (bessel_asym.go:50, :73): x*(2/pi) = 4N + q + f
// with q = round(x*(2/pi)) mod 4 and f in [-1/2, 1/2), against pi from
// math/big. The 100 digits of pi bound the oracle to 2^-330 relative, so x up
// to 2^250 keeps 2^-80 absolute; subnormals take the e == 0 branch, x = 1
// (f = 0.6366 - 1) and x = 3 (0.9 - 1) the f >= 1/2 branch, x = 0.5 the other.
func TestBesselReduceAgainstBig(t *testing.T) {
	tp := bigTwoOverPi()
	xs := []float64{5e-324, 1e-310, 0x1p-1022, 0.5, 1, 3, 10, 355, 1e5, 1e10, 1e22, 0x1p200}
	for _, x := range xs {
		y := new(big.Float).SetPrec(600).Mul(new(big.Float).SetPrec(600).SetFloat64(x), tp)
		// r = round half up (y); f = y - r
		half := new(big.Float).SetPrec(600).Add(y, big.NewFloat(0.5))
		r, _ := half.Int(nil) // truncates toward zero; y > 0 so this is floor
		f := new(big.Float).SetPrec(600).Sub(y, new(big.Float).SetPrec(600).SetInt(r))
		wantQ := int(new(big.Int).And(r, big.NewInt(3)).Int64())
		q, got := besselReduce(x)
		if q != wantQ {
			t.Errorf("besselReduce(%g) q = %d, want %d", x, q, wantQ)
		}
		g := new(big.Float).SetPrec(600).SetFloat64(got.Hi)
		g.Add(g, new(big.Float).SetPrec(600).SetFloat64(got.Lo))
		d := new(big.Float).SetPrec(600).Sub(g, f)
		// besselReduce promises f to 2^-150 ABSOLUTELY, so for x < 2^-100 the
		// answer 0 is within contract and the subnormal branch (e == 0,
		// bessel_asym.go:50, an x*2^-1074 error of at most 2^-1000 in f) is
		// unobservable: the survivor is equivalent, and callers pass x >= 1.
		tol := 0x1p-80
		if dv, _ := d.Float64(); math.Abs(dv) > tol {
			t.Errorf("besselReduce(%g) f = %v + %v, want %v (abs err %g)", x, got.Hi, got.Lo, f, dv)
		}
	}
}

// TestLemmaSinCosQuarterNegativeN (bessel_asym.go:210, :22, :29): the fold
// `if ni < 0 { ni += 4 }` is value-equivalent, because besselSinCosQuarter
// selects its quadrant with n & 3 and Go's & on a negative int is the
// two's-complement residue, so n and n+4 give the same bits; and the result
// is sin/cos of (pi/2)(n+s) (math.Sincos reference). The guards of
// besselTwoOverPiWord (:22 sh >= 64, :29 b == 0) are equivalent likewise:
// Go defines a shift by >= the width as 0, so x>>64 == 0 and x<<0 | y>>64 == x
// (TestBesselTwoOverPiWordDigits pins the words themselves).
func TestLemmaSinCosQuarterNegativeN(t *testing.T) {
	for n := -9; n <= -1; n++ {
		for _, s := range []float64{-0.4, -0.1, 0, 0.25, 0.49} {
			sd := dd.F(s)
			a1, b1 := besselSinCosQuarter(n, sd)
			a2, b2 := besselSinCosQuarter(n+4*3, sd)
			if math.Float64bits(a1) != math.Float64bits(a2) || math.Float64bits(b1) != math.Float64bits(b2) {
				t.Errorf("n=%d s=%g: (%v, %v) != n+12: (%v, %v)", n, s, a1, b1, a2, b2)
			}
			ws, wc := math.Sincos(math.Pi / 2 * (float64(n) + s))
			if math.Abs(a1-ws) > 1e-13 || math.Abs(b1-wc) > 1e-13 {
				t.Errorf("n=%d s=%g: (%v, %v), want (%v, %v)", n, s, a1, b1, ws, wc)
			}
		}
	}
}

// TestBesselAsymNegativePhase (bessel_asym.go:210): the phase reduction
// T = f + q - (v mod 4) - 1/2 is negative whenever q + f < (v mod 4) + 1/2,
// and then Mod(n, 4) is negative and must be folded into 0..3. Checked as
// J_v and Y_v at large x against the standard library, over v = 0..7 and a
// run of x covering every quadrant of the phase.
func TestBesselAsymNegativePhase(t *testing.T) {
	checked := 0
	for v := 0; v <= 7; v++ {
		for i := 0; i < 80; i++ {
			x := 2000 + 0.4*float64(i)
			j, y, ok := besselAsymJY(float64(v), x, true, true)
			if !ok {
				continue
			}
			m := math.Sqrt(2 / (math.Pi * x))
			for _, c := range []struct {
				name      string
				got, want float64
			}{{"J", j, math.Jn(v, x)}, {"Y", y, math.Yn(v, x)}} {
				if math.Abs(c.want) < 0.05*m {
					continue // near a zero the stdlib is not the better oracle
				}
				if rel := math.Abs(c.got-c.want) / math.Abs(c.want); !(rel <= 1e-11) {
					t.Errorf("%s_%d(%g) = %v, stdlib %v (rel %g)", c.name, v, x, c.got, c.want, rel)
				}
				checked++
			}
		}
	}
	if checked < 500 {
		t.Errorf("only %d comparisons were made", checked)
	}
}
