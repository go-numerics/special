package cmplx

import (
	"github.com/go-numerics/special/internal/libm"
	"math"
	"math/big"
	std "math/cmplx"
	"testing"
)

func init() {
	// Target (design-complex.md): 3 ulp per part.
	b := map[string]float64{"*": 3}
	implement("tanh", z2(Tanh), b)
	implement("tan", z2(Tan), b)
	implement("cot", z2(Cot), b)
}

// C11 Annex G G.6.2.6 (ctanh); checked at each entry, its conjugate, and
// -z (ctanh is odd).
var tanhTable = []gcase{
	{in: complex(0, 0), want: complex(0, 0)},
	{in: complex(1, inf), want: complex(nan, nan)},
	{in: complex(0, inf), want: complex(nan, nan)},
	{in: complex(1, nan), want: complex(nan, nan)},
	{in: complex(inf, 1), want: complex(1, 0)},     // 1 + i0 sin 2
	{in: complex(inf, 2), want: complex(1, nzero)}, // sin 4 < 0
	{in: complex(inf, inf), want: complex(1, 0), anyIm: true},
	{in: complex(inf, nan), want: complex(1, 0), anyIm: true},
	{in: complex(nan, 0), want: complex(nan, 0)},
	{in: complex(nan, 1), want: complex(nan, nan)},
	{in: complex(nan, nan), want: complex(nan, nan)},
}

func TestTanhAnnexG(t *testing.T) {
	checkAnnexG(t, "Tanh", Tanh, parityTable(tanhTable, -1))
	for _, z := range specialGrid() {
		w := Tanh(complex(-imag(z), real(z)))
		if !sameC(Tan(z), complex(imag(w), -real(w))) {
			t.Errorf("Tan(%v) != -i Tanh(iz)", z)
		}
		if !finite(real(z)) || !finite(imag(z)) || z == 0 {
			if !sameC(Cot(z), Div(1, Tan(z))) {
				t.Errorf("Cot(%v) = %v, Div(1, Tan) = %v", z, Cot(z), Div(1, Tan(z)))
			}
		}
	}
	if z := Cot(0); !IsInf(z) {
		t.Errorf("Cot(0) = %v, want an infinity", z)
	}
}

func TestTanhSymmetry(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(complex128) complex128
	}{{"tanh", Tanh}, {"tan", Tan}, {"cot", Cot}} {
		checkSymmetry(t, c.name, c.f, odd, fixturePoints(t, c.name))
	}
}

// TestTanhDefects pins the math/cmplx defects these functions fix.
func TestTanhDefects(t *testing.T) {
	if z := Tanh(complex(400, 1)); real(z) != 1 || imag(z) != 0 {
		t.Errorf("Tanh(400+1i) = %v, want 1+0i", z)
	}
	if !math.IsNaN(real(std.Tanh(complex(400, 1)))) {
		t.Log("math/cmplx Tanh(400+1i) is no longer NaN")
	}
	if z := Tan(complex(1, 400)); real(z) != 0 || imag(z) != 1 {
		t.Errorf("Tan(1+400i) = %v, want 0+1i", z)
	}
	// Cot of a tiny argument is 1/z, finite.
	if z, w := Cot(complex(1e-200, 1e-200)), complex(5e199, -5e199); math.Abs(real(z)/real(w)-1) > 1e-15 || math.Abs(imag(z)/imag(w)-1) > 1e-15 {
		t.Errorf("Cot(1e-200+1e-200i) = %v, want %v", z, w)
	}
	// math.Tan returns y for 1e-8 < |y| < 1e-7, dropping y³/3 (15 ulp).
	y := 5e-8
	if want := y + y*y*y/3; tanSmall(y) != want || tan(-y) != -want {
		t.Errorf("tanSmall(%g) = %v, want %v", y, tanSmall(y), want)
	}
	if libm.Tan(y) != y {
		t.Log("math.Tan no longer truncates its series below 1e-7")
	}
	// tan near a pole: y = 3π/2 rounded, tan y = 1/(3π/2 - y) to 1 ulp.
	if got, want := tan(4.712388980384690), 5.443746451065123e15; math.Abs(got/want-1) > 4e-16 {
		t.Errorf("tan(fl(3π/2)) = %v, want %v", got, want)
	}
}

// TestPio2 checks the three-part π/2 against 110 digits.
func TestPio2(t *testing.T) {
	const digits = "1.57079632679489661923132169163975144209858469968755291048747229615390820314310449931401741267105853399107404326"
	want, _, err := big.ParseFloat(digits, 10, 400, big.ToNearestEven)
	if err != nil {
		t.Fatal(err)
	}
	got := new(big.Float).SetPrec(400).SetFloat64(pio2a)
	got.Add(got, big.NewFloat(pio2b)).Add(got, big.NewFloat(pio2c))
	d := new(big.Float).SetPrec(400).Sub(got, want)
	if d.Abs(d).Cmp(big.NewFloat(0x1p-155)) > 0 {
		t.Errorf("pio2a+pio2b+pio2c - π/2 = %g", d)
	}
}

func FuzzTanh(f *testing.F) {
	for _, x := range arithSeeds {
		for _, y := range []float64{0, nzero, 1, -2, 7.853981562768949, 1e300, inf, nan, 0x1p-1074} {
			f.Add(x, y)
		}
	}
	f.Fuzz(func(t *testing.T, x, y float64) {
		z := complex(x, y)
		for _, c := range []struct {
			name string
			f    func(complex128) complex128
		}{{"Tanh", Tanh}, {"Tan", Tan}, {"Cot", Cot}} {
			w := c.f(z)
			if !finite(x) || !finite(y) {
				continue
			}
			if (math.IsNaN(real(w)) || math.IsNaN(imag(w))) && !(c.name == "Cot" && z == 0) {
				t.Fatalf("%s(%v) = %v", c.name, z, w)
			}
			if !sameC(c.f(Conj(z)), Conj(w)) {
				t.Fatalf("%s(conj %v) != conj", c.name, z)
			}
			if !sameC(c.f(-z), -w) {
				t.Fatalf("%s(-%v) != -%s(z)", c.name, z, c.name)
			}
		}
		// |Re tanh| ≤ 1 + 1 ulp wherever |Im z| is far from the poles is not
		// a property; tanh·coth = 1 is, within a few ulp of each part.
		if finite(x) && finite(y) && Abs(z) > 1e-300 && math.Abs(x) < 20 && math.Abs(y) < 1e6 {
			p := Mul(Tan(z), Cot(z))
			if math.Abs(real(p)-1) > 1e-14 || math.Abs(imag(p)) > 1e-14 {
				t.Fatalf("Tan·Cot(%v) = %v", z, p)
			}
		}
	})
}
