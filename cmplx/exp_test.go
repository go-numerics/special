package cmplx

import (
	"math"
	std "math/cmplx"
	"testing"
)

func init() {
	// Targets (design-complex.md): 2 ulp per part. Region bigy (|Im z| in
	// 1e4..1e22) inherits the error of math.Sincos at large arguments,
	// up to 439 ulp where cos or sin is near a zero: an open defect
	// (go-numerics/special#21); the budget there is 1.5x the measurement.
	implement("exp", z2(Exp), map[string]float64{"*": 2, "bigy": 15})
	implement("cosh", z2(Cosh), map[string]float64{"*": 2, "bigy": 21})
	implement("sinh", z2(Sinh), map[string]float64{"*": 2, "bigy": 660})
	implement("cos", z2(Cos), map[string]float64{"*": 2, "bigy": 320})
	implement("sin", z2(Sin), map[string]float64{"*": 2, "bigy": 17})
}

// The C11 Annex G special values (G.6.3.1 cexp, G.6.2.4 ccosh, G.6.2.5
// csinh); each entry is also checked at its conjugate. y = 1 stands for a
// finite nonzero value, cis(1) for cos 1 + i sin 1.
var (
	cis1     = complex(math.Cos(1), math.Sin(1))
	infCis1  = complex(inf*math.Cos(1), inf*math.Sin(1))
	expTable = []gcase{
		{in: complex(0, 0), want: complex(1, 0)},
		{in: complex(nzero, 0), want: complex(1, 0)},
		{in: complex(1, inf), want: complex(nan, nan)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(inf, 0), want: complex(inf, 0)},
		{in: complex(-inf, 1), want: complex(0*real(cis1), 0*imag(cis1))},
		{in: complex(-inf, -3), want: complex(math.Copysign(0, math.Cos(-3)), math.Copysign(0, math.Sin(-3)))},
		{in: complex(inf, 1), want: infCis1},
		{in: complex(-inf, inf), want: complex(0, 0), anyRe: true, anyIm: true},
		{in: complex(inf, inf), want: complex(inf, nan), anyRe: true},
		{in: complex(-inf, nan), want: complex(0, 0), anyRe: true, anyIm: true},
		{in: complex(inf, nan), want: complex(inf, nan), anyRe: true},
		{in: complex(nan, 0), want: complex(nan, 0)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	coshTable = []gcase{
		{in: complex(0, 0), want: complex(1, 0)},
		{in: complex(0, inf), want: complex(nan, 0), anyIm: true},
		{in: complex(0, nan), want: complex(nan, 0), anyIm: true},
		{in: complex(1, inf), want: complex(nan, nan)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(inf, 0), want: complex(inf, 0)},
		{in: complex(inf, 1), want: infCis1},
		{in: complex(inf, inf), want: complex(inf, nan), anyRe: true},
		{in: complex(inf, nan), want: complex(inf, nan)},
		{in: complex(nan, 0), want: complex(nan, 0), anyIm: true},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	sinhTable = []gcase{
		{in: complex(0, 0), want: complex(0, 0)},
		{in: complex(0, inf), want: complex(0, nan), anyRe: true},
		{in: complex(0, nan), want: complex(0, nan), anyRe: true},
		{in: complex(1, inf), want: complex(nan, nan)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(inf, 0), want: complex(inf, 0)},
		{in: complex(inf, 1), want: infCis1},
		{in: complex(inf, inf), want: complex(inf, nan), anyRe: true},
		{in: complex(inf, nan), want: complex(inf, nan), anyRe: true},
		{in: complex(nan, 0), want: complex(nan, 0)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
)

// parity maps an Annex G table to -z for an even or odd function
// (G.6.2.4: ccosh is even, G.6.2.5: csinh is odd).
func parityTable(t []gcase, sign float64) []gcase {
	out := append([]gcase(nil), t...)
	for _, c := range t {
		out = append(out, gcase{-c.in, complex(sign*real(c.want), sign*imag(c.want)), c.anyRe, c.anyIm})
	}
	return out
}

func TestExpAnnexG(t *testing.T) {
	checkAnnexG(t, "Exp", Exp, expTable)
	checkAnnexG(t, "Cosh", Cosh, parityTable(coshTable, 1))
	checkAnnexG(t, "Sinh", Sinh, parityTable(sinhTable, -1))
	// G.6p7: Cos(z) = Cosh(iz), Sin(z) = -i Sinh(iz), on the whole grid.
	for _, z := range specialGrid() {
		if !sameC(Cos(z), Cosh(complex(-imag(z), real(z)))) {
			t.Errorf("Cos(%v) != Cosh(iz)", z)
		}
		w := Sinh(complex(-imag(z), real(z)))
		if !sameC(Sin(z), complex(imag(w), -real(w))) {
			t.Errorf("Sin(%v) != -i Sinh(iz)", z)
		}
	}
}

func TestExpSymmetry(t *testing.T) {
	for _, c := range []struct {
		name   string
		f      func(complex128) complex128
		parity int
	}{
		{"exp", Exp, noParity}, {"cosh", Cosh, even}, {"sinh", Sinh, odd}, {"cos", Cos, even}, {"sin", Sin, odd},
	} {
		checkSymmetry(t, c.name, c.f, c.parity, fixturePoints(t, c.name))
	}
}

// TestExpDefects pins the math/cmplx defects these functions fix.
func TestExpDefects(t *testing.T) {
	// Exp(710+1e-300i) = e^710 (cos + i sin 1e-300): the imaginary part is
	// 2.23e8, finite; math/cmplx returns +Inf for both parts.
	z := Exp(complex(710, 1e-300))
	if want := math.Exp(355) * (math.Exp(355) * 1e-300); !math.IsInf(real(z), 1) || math.Abs(imag(z)/want-1) > 1e-13 {
		t.Errorf("Exp(710+1e-300i) = %v, want +Inf + 2.2339e8i", z)
	}
	if !math.IsInf(imag(std.Exp(complex(710, 1e-300))), 1) {
		t.Log("math/cmplx Exp(710+1e-300i) no longer overflows")
	}
	// Cosh(710.4+1i) = 9.0e307 + 1.4e308i, both finite.
	z = Cosh(complex(710.4, 1))
	if math.IsInf(real(z), 0) || math.IsInf(imag(z), 0) || real(z) < 9e307 || imag(z) < 1.4e308 {
		t.Errorf("Cosh(710.4+1i) = %v, want 9.0e307+1.4e308i", z)
	}
	// Far beyond: every part overflows, with the sign of cos/sin.
	if z := Exp(complex(2e7, -2)); !math.IsInf(real(z), -1) || !math.IsInf(imag(z), -1) {
		t.Errorf("Exp(2e7-2i) = %v", z)
	}
	if z := Exp(complex(-2e7, 2)); real(z) != 0 || imag(z) != 0 || math.Signbit(imag(z)) || !math.Signbit(real(z)) {
		t.Errorf("Exp(-2e7+2i) = %v, want -0+0i", z)
	}
	if z := Exp(complex(-2e7, 0)); z != 0 {
		t.Errorf("Exp(-2e7) = %v", z)
	}
}

func FuzzExp(f *testing.F) {
	for _, x := range arithSeeds {
		for _, y := range []float64{0, nzero, 1, -2, 1e300, inf, nan, 0x1p-1074} {
			f.Add(x, y)
		}
	}
	f.Add(710.4, 1.0)
	f.Add(1454.0, 1e-310)
	f.Fuzz(func(t *testing.T, x, y float64) {
		z := complex(x, y)
		for _, c := range []struct {
			name   string
			f      func(complex128) complex128
			parity int
		}{
			{"Exp", Exp, noParity}, {"Cosh", Cosh, even}, {"Sinh", Sinh, odd}, {"Cos", Cos, even}, {"Sin", Sin, odd},
		} {
			w := c.f(z)
			if !finite(x) || !finite(y) {
				continue
			}
			// Finite in, never NaN out (Annex G: NaN only from NaN or an
			// infinite argument).
			if math.IsNaN(real(w)) || math.IsNaN(imag(w)) {
				t.Fatalf("%s(%v) = %v", c.name, z, w)
			}
			if !sameC(c.f(Conj(z)), Conj(w)) {
				t.Fatalf("%s(conj %v) != conj", c.name, z)
			}
			switch c.parity {
			case even:
				if !sameC(c.f(-z), w) {
					t.Fatalf("%s(-%v) != %s(z)", c.name, z, c.name)
				}
			case odd:
				if !sameC(c.f(-z), -w) {
					t.Fatalf("%s(-%v) != -%s(z)", c.name, z, c.name)
				}
			}
		}
		// |Exp(z)| = e^x within a few ulp where it is in range (two parts
		// of 2 ulp each, Hypot and Exp of 1 ulp each).
		if finite(x) && finite(y) && math.Abs(x) < 700 {
			if a, e := Abs(Exp(z)), math.Exp(x); math.Abs(a-e) > 1e-15*e {
				t.Fatalf("|Exp(%v)| = %g, e^x = %g", z, a, e)
			}
		}
	})
}
