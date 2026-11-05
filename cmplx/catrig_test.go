package cmplx

import (
	"math"
	std "math/cmplx"
	"testing"
)

func init() {
	for _, c := range []struct {
		name   string
		f, s   func(complex128) complex128
		budget float64
	}{
		// Target (design-complex.md): 2 ulp per part. Asinh, Asin, Acos and
		// Acosh measure up to 2.82 ulp near the branch points and in the
		// moderate region: msun's do_hard_work rounds |z±i|, A and A-1 in
		// float64 (go-numerics/special#26).
		{"asinh", Asinh, std.Asinh, 3}, {"asin", Asin, std.Asin, 3}, {"acos", Acos, std.Acos, 3},
		{"acosh", Acosh, std.Acosh, 3}, {"atanh", Atanh, std.Atanh, 2}, {"atan", Atan, std.Atan, 2},
	} {
		register(accCase{fixture: c.name, std: z2(c.s)})
		implement(c.name, z2(c.f), map[string]float64{"*": c.budget})
	}
}

var (
	asinhTable = []gcase{ // C11 G.6.2.2
		{in: complex(0, 0), want: complex(0, 0)},
		{in: complex(1, inf), want: complex(inf, math.Pi/2)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(inf, 1), want: complex(inf, 0)},
		{in: complex(inf, inf), want: complex(inf, math.Pi/4)},
		{in: complex(inf, nan), want: complex(inf, nan)},
		{in: complex(nan, 0), want: complex(nan, 0)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, inf), want: complex(inf, nan), anyRe: true},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	acosTable = []gcase{ // C11 G.6.1.1
		{in: complex(0, 0), want: complex(math.Pi/2, nzero)},
		{in: complex(nzero, 0), want: complex(math.Pi/2, nzero)},
		{in: complex(0, nan), want: complex(math.Pi/2, nan)},
		{in: complex(1, inf), want: complex(math.Pi/2, -inf)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(-inf, 1), want: complex(math.Pi, -inf)},
		{in: complex(inf, 1), want: complex(0, -inf)},
		{in: complex(-inf, inf), want: complex(3*math.Pi/4, -inf)},
		{in: complex(inf, inf), want: complex(math.Pi/4, -inf)},
		{in: complex(inf, nan), want: complex(nan, inf), anyIm: true},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, inf), want: complex(nan, -inf)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	acoshTable = []gcase{ // C11 G.6.2.1
		{in: complex(0, 0), want: complex(0, math.Pi/2)},
		{in: complex(nzero, 0), want: complex(0, math.Pi/2)},
		{in: complex(1, inf), want: complex(inf, math.Pi/2)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(-inf, 1), want: complex(inf, math.Pi)},
		{in: complex(inf, 1), want: complex(inf, 0)},
		{in: complex(-inf, inf), want: complex(inf, 3*math.Pi/4)},
		{in: complex(inf, inf), want: complex(inf, math.Pi/4)},
		{in: complex(inf, nan), want: complex(inf, nan)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, inf), want: complex(inf, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	atanhTable = []gcase{ // C11 G.6.2.3
		{in: complex(0, 0), want: complex(0, 0)},
		{in: complex(0, nan), want: complex(0, nan)},
		{in: complex(1, 0), want: complex(inf, 0)},
		{in: complex(1, inf), want: complex(0, math.Pi/2)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(inf, 1), want: complex(0, math.Pi/2)},
		{in: complex(inf, inf), want: complex(0, math.Pi/2)},
		{in: complex(inf, nan), want: complex(0, nan)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, inf), want: complex(0, math.Pi/2), anyRe: true},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
)

func TestCatrigAnnexG(t *testing.T) {
	checkAnnexG(t, "Asinh", Asinh, parityTable(asinhTable, -1))
	checkAnnexG(t, "Acos", Acos, acosTable)
	checkAnnexG(t, "Acosh", Acosh, acoshTable)
	checkAnnexG(t, "Atanh", Atanh, parityTable(atanhTable, -1))
	// G.6p7 relations on the grid.
	for _, z := range specialGrid() {
		w := Asinh(complex(imag(z), real(z)))
		if !sameC(Asin(z), complex(imag(w), real(w))) {
			t.Errorf("Asin(%v) != i conj Asinh(i conj z)", z)
		}
		w = Atanh(complex(imag(z), real(z)))
		if !sameC(Atan(z), complex(imag(w), real(w))) {
			t.Errorf("Atan(%v) != i conj Atanh(i conj z)", z)
		}
	}
}

func TestCatrigSymmetry(t *testing.T) {
	for _, c := range []struct {
		name   string
		f      func(complex128) complex128
		parity int
	}{
		{"asinh", Asinh, odd}, {"asin", Asin, odd}, {"atanh", Atanh, odd}, {"atan", Atan, odd},
		{"acos", Acos, noParity}, {"acosh", Acosh, noParity},
	} {
		checkSymmetry(t, c.name, c.f, c.parity, fixturePoints(t, c.name))
	}
}

// TestCatrigCut asserts the jump across each cut and pins math/cmplx's
// wrong quadrant for huge arguments.
func TestCatrigCut(t *testing.T) {
	for _, x := range []float64{1.5, 3, 1e10, 1e300} {
		up, down := Asin(complex(x, 0)), Asin(complex(x, nzero))
		if real(up) != real(down) || imag(up) <= 0 || imag(down) != -imag(up) {
			t.Errorf("Asin(%g ± i0) = %v, %v", x, up, down)
		}
		up, down = Acosh(complex(-x, 0)), Acosh(complex(-x, nzero))
		if imag(up) != math.Pi || imag(down) != -math.Pi {
			t.Errorf("Acosh(%g ± i0) = %v, %v", -x, up, down)
		}
		up, down = Atan(complex(0, x)), Atan(complex(nzero, x))
		if real(up) != math.Pi/2 || real(down) != -math.Pi/2 {
			t.Errorf("Atan(±0 + %gi) = %v, %v", x, up, down)
		}
	}
	// Asin(1e200+1e200i) = π/4 + i ln(2√2·1e200) = 0.785 + 461.5567i;
	// math/cmplx returns -0.785 - Inf i.
	if w := Asin(complex(1e200, 1e200)); math.Abs(real(w)-math.Pi/4) > 1e-15 || math.Abs(imag(w)/461.55673936964908-1) > 1e-15 {
		t.Errorf("Asin(1e200+1e200i) = %v", w)
	}
	// At the branch point i: asinh(i + x) = sqrt(x) + i(π/2 - sqrt(x)) + O(x).
	for _, x := range []float64{1e-40, 1e-20} {
		w, s := Asinh(complex(x, 1)), math.Sqrt(x)
		if math.Abs(real(w)/s-1) > 1e-15 || math.Abs(imag(w)-(math.Pi/2-s)) > 1e-15 {
			t.Errorf("Asinh(%g+1i) = %v, want %g + i(π/2-%g)", x, w, s, s)
		}
	}
	// Acos(1+1e-20i) = 1e-10 - 1e-10i; math/cmplx is off by 8e-8 relative.
	if w := Acos(complex(1, 1e-20)); math.Abs(real(w)/1e-10-1) > 1e-15 || math.Abs(imag(w)/-1e-10-1) > 1e-15 {
		t.Errorf("Acos(1+1e-20i) = %v", w)
	}
}

func FuzzCatrig(f *testing.F) {
	for _, x := range arithSeeds {
		for _, y := range []float64{0, nzero, 1, -2, 1e-10, 1e300, inf, nan, 0x1p-1074} {
			f.Add(x, y)
		}
	}
	f.Fuzz(func(t *testing.T, x, y float64) {
		z := complex(x, y)
		for _, c := range []struct {
			name string
			f    func(complex128) complex128
		}{{"Asinh", Asinh}, {"Asin", Asin}, {"Acos", Acos}, {"Acosh", Acosh}, {"Atanh", Atanh}, {"Atan", Atan}} {
			w := c.f(z)
			if !finite(x) || !finite(y) {
				continue // several entries leave a sign unspecified (G.6)
			}
			if !sameC(c.f(Conj(z)), Conj(w)) {
				t.Fatalf("%s(conj %v) != conj", c.name, z)
			}
			if math.IsNaN(real(w)) || math.IsNaN(imag(w)) {
				t.Fatalf("%s(%v) = %v", c.name, z, w)
			}
		}
		// Sinh(Asinh z) = z where both are well conditioned.
		if a := Abs(z); finite(x) && finite(y) && a > 1e-300 && a < 1e3 && math.Abs(Abs(complex(math.Abs(x), math.Abs(y))-complex(0, 1))) > 1e-3 {
			if d := Abs(Sinh(Asinh(z)) - z); d > 2e-14*a {
				t.Fatalf("Sinh(Asinh(%v)) off by %g", z, d/a)
			}
		}
	})
}
