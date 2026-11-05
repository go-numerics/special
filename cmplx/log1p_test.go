package cmplx

import (
	"math"
	std "math/cmplx"
	"testing"
)

func init() {
	// The comparison column is the naive formula (NumPy's nc_log1p and
	// nc_expm1) through math/cmplx.
	register(accCase{fixture: "log1p", std: z2(func(z complex128) complex128 { return std.Log(1 + z) })})
	register(accCase{fixture: "expm1", std: z2(func(z complex128) complex128 { return std.Exp(z) - 1 })})
	// Target (design-complex.md): 2 ulp per part.
	implement("log1p", z2(Log1p), map[string]float64{"*": 2})
	implement("expm1", z2(Expm1), map[string]float64{"*": 2})
}

func TestLog1pSpecial(t *testing.T) {
	for _, z := range specialGrid() {
		x, y := real(z), imag(z)
		if !finite(x) || !finite(y) {
			if w, want := Log1p(z), Log(complex(1+x, y)); !sameC(w, want) {
				t.Errorf("Log1p(%v) = %v, want Log(1+z) = %v", z, w, want)
			}
			e := Exp(z)
			if w, want := Expm1(z), complex(real(e)-1, imag(e)); !sameC(w, want) {
				t.Errorf("Expm1(%v) = %v, want Exp(z)-1 = %v", z, w, want)
			}
		}
	}
	for _, z := range []complex128{0, complex(nzero, 0), complex(0, nzero), complex(nzero, nzero)} {
		if !sameC(Log1p(z), z) || !sameC(Expm1(z), z) {
			t.Errorf("Log1p/Expm1(%v) = %v, %v, want z", z, Log1p(z), Expm1(z))
		}
	}
	if w := Log1p(-1); !math.IsInf(real(w), -1) || imag(w) != 0 {
		t.Errorf("Log1p(-1) = %v, want -Inf", w)
	}
	if w := Log1p(complex(-3, nzero)); imag(w) != -math.Pi {
		t.Errorf("Log1p(-3 - i0) = %v, want Im -π (cut)", w)
	}
}

func TestLog1pSymmetry(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(complex128) complex128
	}{{"log1p", Log1p}, {"expm1", Expm1}} {
		checkSymmetry(t, c.name, c.f, noParity, fixturePoints(t, c.name))
		checkSymmetry(t, c.name, c.f, noParity, specialGrid())
	}
}

// TestLog1pDefects pins what the naive formulas (NumPy's) lose.
func TestLog1pDefects(t *testing.T) {
	if w := Log1p(complex(1e-20, 1e-20)); real(w) != 1e-20 || imag(w) != 1e-20 {
		t.Errorf("Log1p(1e-20+1e-20i) = %v", w)
	}
	if w := Expm1(complex(0, 1e-20)); real(w) != -5e-41 || imag(w) != 1e-20 {
		t.Errorf("Expm1(1e-20i) = %v, want -5e-41+1e-20i", w)
	}
	// |1+z|² - 1 = 2x + x² + y² cancels: z = -1e-8 + i sqrt(2e-8 - 1e-16).
	x := -1e-8
	y := math.Sqrt(-2*x - x*x)
	if w := Log1p(complex(x, y)); math.Abs(real(w)) > 1e-24 {
		t.Errorf("Log1p(%v) real part %g, want |.| < 1e-24", complex(x, y), real(w))
	}
}

func FuzzLog1p(f *testing.F) {
	for _, x := range arithSeeds {
		for _, y := range []float64{0, nzero, 1, -2, 1e-10, 1e300, inf, nan, 0x1p-1074} {
			f.Add(x, y)
		}
	}
	f.Add(-1e-8, 1.4142135623730951e-4)
	f.Fuzz(func(t *testing.T, x, y float64) {
		z := complex(x, y)
		for _, c := range []struct {
			name string
			f    func(complex128) complex128
		}{{"Log1p", Log1p}, {"Expm1", Expm1}} {
			w := c.f(z)
			if !sameC(c.f(Conj(z)), Conj(w)) {
				t.Fatalf("%s(conj %v) != conj", c.name, z)
			}
			if finite(x) && finite(y) && (math.IsNaN(real(w)) || math.IsNaN(imag(w))) {
				t.Fatalf("%s(%v) = %v", c.name, z, w)
			}
		}
		// Expm1(Log1p z) = z for small |z|, within conditioning.
		if a := Abs(z); finite(x) && finite(y) && a > 1e-300 && a < 0.25 {
			if d := Abs(Expm1(Log1p(z)) - z); d > 2e-15*a {
				t.Fatalf("Expm1(Log1p(%v)) off by %g", z, d/a)
			}
		}
	})
}
