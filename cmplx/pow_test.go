package cmplx

import (
	"math"
	std "math/cmplx"
	"testing"
)

func init() {
	register(accCase{fixture: "pow", std: func(a []float64) complex128 { return std.Pow(complex(a[0], a[1]), complex(a[2], a[3])) }})
	// Target (design-complex.md): exact for representable integer powers,
	// otherwise normwise (2 + |b log a|) ulp; measured per part here. Region
	// unit (|a| = 1 ± 2^-52..2^-10, |b| up to 1e8, |b log a| up to 3e5)
	// measures 2937 ulp, within that bound; the log and atan2 kernels carry
	// about 2^-65, which |b log a| amplifies (go-numerics/special#27).
	implement("pow", func(a []float64) complex128 { return Pow(complex(a[0], a[1]), complex(a[2], a[3])) },
		map[string]float64{"*": 2, "unit": 4500})
}

func TestPowSpecial(t *testing.T) {
	c := func(re, im float64) complex128 { return complex(re, im) }
	for _, k := range []struct{ x, y, want complex128 }{
		{c(0, 1), 2, -1},                     // i² exactly (math/cmplx: -1 + 1.2e-16i)
		{c(1, 1), 4, -4},                     // (1+i)⁴
		{c(1, 1), -2, c(0, -0.5)},            // (1+i)^-2 = -i/2
		{c(3, -4), 1, c(3, -4)},              // n = 1 returns x
		{c(nan, nan), 0, 1},                  // y = 0: 1, as NumPy
		{0, c(2, 5), 0},                      // 0^y = 0 for Re y > 0
		{c(2, 0), c(0.5, 0), math.Sqrt2},     // real square root
		{c(1e300, 0), 2, c(inf, 0)},          // overflow
		{c(0x1p-600, 0), 2, c(0, 0)},         // underflow
		{c(0x1p-537, 0), 2, c(0x1p-1074, 0)}, // subnormal result, exact
	} {
		if got := Pow(k.x, k.y); !sameC(got, k.want) {
			t.Errorf("Pow(%v, %v) = %v, want %v", k.x, k.y, got, k.want)
		}
	}
	if w := Pow(0, c(-1, 0)); !IsNaN(w) {
		t.Errorf("Pow(0, -1) = %v, want NaN (NumPy)", w)
	}
	// Integer-valued but outside (-100, 100): no conversion to int, the
	// exp-log path.
	if w := Pow(c(1, 0x1p-60), 1e300); IsNaN(w) {
		t.Errorf("Pow(1+2^-60 i, 1e300) = %v", w)
	}
	if w := Pow(c(inf, 0), 2); !IsInf(w) {
		t.Errorf("Pow(Inf, 2) = %v, want an infinity", w)
	}
	if w := Pow(std.Pow(1i, 2), 1); imag(w) == 0 {
		t.Log("math/cmplx Pow(i, 2) is now exact")
	}
}

func FuzzPow(f *testing.F) {
	for _, s := range arithSeeds {
		f.Add(s, 1.0, 2.0, 0.0)
		f.Add(0.5, s, -3.0, 0.0)
		f.Add(s, -1.0, 0.5, 1.5)
	}
	f.Fuzz(func(t *testing.T, a, b, c, d float64) {
		x, y := complex(a, b), complex(c, d)
		w := Pow(x, y)
		if !finite(a) || !finite(b) || !finite(c) || !finite(d) || x == 0 {
			return
		}
		if math.IsNaN(real(w)) || math.IsNaN(imag(w)) {
			t.Fatalf("Pow(%v, %v) = %v", x, y, w)
		}
		// Integer power 2 is the correctly rounded product.
		if p, m := Pow(x, 2), Mul(x, x); !within1(real(p), real(m)) || !within1(imag(p), imag(m)) {
			t.Fatalf("Pow(%v, 2) = %v, Mul = %v", x, p, m)
		}
		// Conjugate symmetry off the cut.
		if b != 0 && !sameZ(Pow(Conj(x), Conj(y)), Conj(w)) && !IsInf(w) {
			if d := Abs(Pow(Conj(x), Conj(y)) - Conj(w)); d > 1e-15*Abs(w) {
				t.Fatalf("Pow(conj) != conj Pow at %v, %v", x, y)
			}
		}
	})
}
