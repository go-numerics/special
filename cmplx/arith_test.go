package cmplx

import (
	"math"
	"math/big"
	"testing"

	"github.com/go-numerics/special/dd"
)

func init() {
	implement("mul", func(a []float64) complex128 { return Mul(complex(a[0], a[1]), complex(a[2], a[3])) }, map[string]float64{"*": 1})
	implement("div", func(a []float64) complex128 { return Div(complex(a[0], a[1]), complex(a[2], a[3])) }, map[string]float64{"*": 1})
}

// exactPart returns a*b + c*d rounded once to float64 (math/big).
func exactPart(a, b, c, d float64) float64 {
	const p = 4400 // holds any sum of two float64 products exactly
	x := new(big.Float).SetPrec(p).Mul(big.NewFloat(a), big.NewFloat(b))
	y := new(big.Float).SetPrec(p).Mul(big.NewFloat(c), big.NewFloat(d))
	f, _ := x.Add(x, y).Float64()
	return f
}

// exactQuot returns (a*b + c*d) / (e² + f²) rounded to float64, to within
// the rounding of a 400-bit quotient.
func exactQuot(a, b, c, d, e, f float64) float64 {
	const p = 4400
	n := new(big.Float).SetPrec(p).Mul(big.NewFloat(a), big.NewFloat(b))
	n.Add(n, new(big.Float).SetPrec(p).Mul(big.NewFloat(c), big.NewFloat(d)))
	m := new(big.Float).SetPrec(p).Mul(big.NewFloat(e), big.NewFloat(e))
	m.Add(m, new(big.Float).SetPrec(p).Mul(big.NewFloat(f), big.NewFloat(f)))
	q, _ := new(big.Float).SetPrec(400).Quo(n, m).Float64()
	return q
}

// within1 reports whether got is exact or one float64 away from want.
func within1(got, want float64) bool {
	return same(got, want) || got == want || math.Nextafter(want, math.Inf(1)) == got || math.Nextafter(want, math.Inf(-1)) == got
}

// sameZ is sameC with the zeros compared by value, not by sign.
func sameZ(a, b complex128) bool {
	z := func(x float64) float64 {
		if x == 0 {
			return 0
		}
		return x
	}
	return sameC(complex(z(real(a)), z(imag(a))), complex(z(real(b)), z(imag(b))))
}

var arithSeeds = []float64{0, math.Copysign(0, -1), 1, -1, 0x1p-1074, -0x1p-1074, 0x1p-1022,
	math.MaxFloat64, -math.MaxFloat64, 1e300, 1e-300, math.Inf(1), math.Inf(-1), math.NaN(), 3, 0.1}

func FuzzMul(f *testing.F) {
	for i, s := range arithSeeds {
		f.Add(s, arithSeeds[(i+3)%len(arithSeeds)], arithSeeds[(i+7)%len(arithSeeds)], arithSeeds[(i+11)%len(arithSeeds)])
	}
	f.Fuzz(func(t *testing.T, a, b, c, d float64) {
		x, y := complex(a, b), complex(c, d)
		got := Mul(x, y)
		if !sameC(got, Mul(y, x)) {
			t.Fatalf("Mul(%v, %v) = %v, Mul(y, x) = %v", x, y, got, Mul(y, x))
		}
		// IEEE 754 sums of signed zeros (+0 + -0 = +0) break the conj
		// symmetry of the zeros of a product; the values must agree.
		if !sameZ(Mul(Conj(x), Conj(y)), Conj(got)) {
			t.Fatalf("Mul(conj) != conj Mul at %v, %v", x, y)
		}
		if finite(a) && finite(b) && finite(c) && finite(d) {
			if re := exactPart(a, c, -b, d); !within1(real(got), re) {
				t.Fatalf("Re Mul(%v, %v) = %g, exact %g", x, y, real(got), re)
			}
			if im := exactPart(a, d, b, c); !within1(imag(got), im) {
				t.Fatalf("Im Mul(%v, %v) = %g, exact %g", x, y, imag(got), im)
			}
			return
		}
		// Annex G: an infinite factor times a nonzero factor is an infinity.
		if (IsInf(x) && !IsNaN(y) && y != 0) || (IsInf(y) && !IsNaN(x) && x != 0) {
			if !IsInf(got) {
				t.Fatalf("Mul(%v, %v) = %v, want an infinity", x, y, got)
			}
		}
	})
}

func FuzzDiv(f *testing.F) {
	for i, s := range arithSeeds {
		f.Add(s, arithSeeds[(i+5)%len(arithSeeds)], arithSeeds[(i+2)%len(arithSeeds)], arithSeeds[(i+9)%len(arithSeeds)])
	}
	f.Fuzz(func(t *testing.T, a, b, c, d float64) {
		x, y := complex(a, b), complex(c, d)
		got := Div(x, y)
		if !sameZ(Div(Conj(x), Conj(y)), Conj(got)) {
			t.Fatalf("Div(conj) != conj Div at %v, %v", x, y)
		}
		if finite(a) && finite(b) && finite(c) && finite(d) && y != 0 {
			if re := exactQuot(a, c, b, d, c, d); !within1(real(got), re) {
				t.Fatalf("Re Div(%v, %v) = %g, exact %g", x, y, real(got), re)
			}
			if im := exactQuot(b, c, -a, d, c, d); !within1(imag(got), im) {
				t.Fatalf("Im Div(%v, %v) = %g, exact %g", x, y, imag(got), im)
			}
			return
		}
		switch {
		case y == 0 && !IsNaN(x) && x != 0:
			if !IsInf(got) {
				t.Fatalf("Div(%v, 0) = %v, want an infinity", x, got)
			}
		case IsInf(x) && finite(c) && finite(d):
			if !IsInf(got) {
				t.Fatalf("Div(%v, %v) = %v, want an infinity", x, y, got)
			}
		case IsInf(y) && finite(a) && finite(b):
			if real(got) != 0 || imag(got) != 0 {
				t.Fatalf("Div(%v, %v) = %v, want a zero", x, y, got)
			}
		}
	})
}

func TestArithAnnexG(t *testing.T) {
	c := func(re, im float64) complex128 { return complex(re, im) }
	for _, k := range []struct{ x, y, want complex128 }{
		// G.5.1p4: an infinity times a nonzero finite value is an infinity.
		{c(inf, nan), c(1, 1), c(inf, inf)},
		{c(nan, inf), c(2, 0), c(nan, inf)},
		{c(1, 1), c(nan, -inf), c(inf, -inf)},
		// A finite product that overflows next to a NaN part.
		{c(1e300, nan), c(1e300, 0), c(inf, nan)},
		{c(nan, 1e300), c(1e300, 0), c(nan, inf)},
		// Exact zeros keep IEEE signs.
		{c(nzero, 0), c(1, 0), c(nzero, 0)},
		{c(0, 0), c(nzero, nzero), c(0, nzero)},
		{c(nan, nan), c(1, 1), c(nan, nan)},
	} {
		if got := Mul(k.x, k.y); !sameC(got, k.want) {
			t.Errorf("Mul(%v, %v) = %v, want %v", k.x, k.y, got, k.want)
		}
	}
	for _, k := range []struct{ x, y, want complex128 }{
		{c(1, 1), c(0, 0), c(inf, inf)},
		{c(1, 1), c(nzero, 0), c(-inf, -inf)},
		{c(inf, nan), c(1, 1), c(inf, -inf)},
		{c(inf, 1), c(1, 1), c(inf, -inf)},
		{c(nan, inf), c(2, 0), c(nan, inf)},
		{c(1, 1), c(inf, nan), c(0, 0)},
		{c(1, 1), c(inf, 1), c(0, 0)},
		{c(nan, nan), c(0, 0), c(nan, nan)},
		{c(0, 0), c(1, 1), c(0, 0)},
		{c(0, nzero), c(1, 0), c(0, nzero)},
		{c(1, 1), c(nan, 1), c(nan, nan)},
	} {
		if got := Div(k.x, k.y); !sameC(got, k.want) {
			t.Errorf("Div(%v, %v) = %v, want %v", k.x, k.y, got, k.want)
		}
	}
	for _, k := range []struct{ x, want complex128 }{
		{c(1, -2), c(1, -2)},
		{c(inf, -2), c(inf, nzero)},
		{c(nan, -inf), c(inf, nzero)},
		{c(-inf, nan), c(inf, 0)},
		{c(nan, 1), c(nan, 1)},
	} {
		if got := Proj(k.x); !sameC(got, k.want) {
			t.Errorf("Proj(%v) = %v, want %v", k.x, got, k.want)
		}
	}
}

// TestArithDefects pins the cases the builtin operators get wrong.
func TestArithDefects(t *testing.T) {
	// Cancellation: (1+2^-30 + i) * (1-2^-30 + i) = (1-2^-60) - 1 + 2i.
	x, y := complex(1+0x1p-30, 1), complex(1-0x1p-30, 1)
	if got := Mul(x, y); real(got) != -0x1p-60 || imag(got) != 2 {
		t.Errorf("Mul cancellation = %v, want (-2^-60+2i)", got)
	}
	// Intermediate overflow: (1e300+1e300i)/(1e300+1e300i) = 1.
	if got := Div(complex(1e300, 1e300), complex(1e300, 1e300)); got != 1 {
		t.Errorf("Div huge = %v, want 1", got)
	}
	// Subnormal results are rounded once.
	if got := round(dd.DD{Hi: 1.5, Lo: 0}, -1074, 0); got != 0x1p-1073 {
		t.Errorf("round(1.5 * 2^-1074) = %g, want 2^-1073 (ties to even)", got)
	}
	if got := round(dd.DD{Hi: 2.5, Lo: 0x1p-60}, -1074, 0); got != 0x1.8p-1073 {
		t.Errorf("round((2.5+2^-60) * 2^-1074) = %g, want 3*2^-1074", got)
	}
	if got := round(dd.DD{Hi: 1.5, Lo: -0x1p-60}, -1074, 0); got != 0x1p-1074 {
		t.Errorf("round((1.5-2^-60) * 2^-1074) = %g, want 2^-1074", got)
	}
}
