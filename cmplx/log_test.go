package cmplx

import (
	"github.com/go-numerics/special/internal/libm"
	"math"
	"math/big"
	std "math/cmplx"
	"testing"

	"github.com/go-numerics/special/dd"
)

func init() {
	for _, c := range []struct {
		name   string
		f, s   func(complex128) complex128
		budget float64
	}{{"log", Log, std.Log, 1}, {"log10", Log10, std.Log10, 1.5}, {"log2", Log2, nil, 1.5}, {"sqrt", Sqrt, std.Sqrt, 1.5}} {
		s := c.s
		if s == nil {
			s = func(z complex128) complex128 { return std.Log(z) / complex(math.Ln2, 0) }
		}
		register(accCase{fixture: c.name, std: z2(s)})
		// Targets (design-complex.md): Log 1 ulp per part, Log10, Log2,
		// Sqrt 1.5 ulp.
		implement(c.name, z2(c.f), map[string]float64{"*": c.budget})
	}
}

var (
	logTable = []gcase{ // C11 G.6.3.2
		{in: complex(nzero, 0), want: complex(-inf, math.Pi)},
		{in: complex(0, 0), want: complex(-inf, 0)},
		{in: complex(1, inf), want: complex(inf, math.Pi/2)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(-inf, 1), want: complex(inf, math.Pi)},
		{in: complex(inf, 1), want: complex(inf, 0)},
		{in: complex(-inf, inf), want: complex(inf, libm.Atan2(inf, -inf))},
		{in: complex(inf, inf), want: complex(inf, math.Pi/4)},
		{in: complex(inf, nan), want: complex(inf, nan)},
		{in: complex(-inf, nan), want: complex(inf, nan)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, inf), want: complex(inf, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
	sqrtTable = []gcase{ // C11 G.6.4.2
		{in: complex(0, 0), want: complex(0, 0)},
		{in: complex(nzero, 0), want: complex(0, 0)},
		{in: complex(1, inf), want: complex(inf, inf)},
		{in: complex(nan, inf), want: complex(inf, inf)},
		{in: complex(-inf, inf), want: complex(inf, inf)},
		{in: complex(1, nan), want: complex(nan, nan)},
		{in: complex(-inf, 1), want: complex(0, inf)},
		{in: complex(inf, 1), want: complex(inf, 0)},
		{in: complex(-inf, nan), want: complex(nan, inf), anyIm: true},
		{in: complex(inf, nan), want: complex(inf, nan)},
		{in: complex(nan, 1), want: complex(nan, nan)},
		{in: complex(nan, nan), want: complex(nan, nan)},
	}
)

func TestLogAnnexG(t *testing.T) {
	checkAnnexG(t, "Log", Log, logTable)
	checkAnnexG(t, "Sqrt", Sqrt, sqrtTable)
	for _, z := range specialGrid() {
		l, l10, l2 := Log(z), Log10(z), Log2(z)
		for _, w := range []complex128{l10, l2} {
			if math.IsInf(real(l), 0) && real(w) != real(l) || math.IsNaN(real(l)) != math.IsNaN(real(w)) ||
				math.IsNaN(imag(l)) != math.IsNaN(imag(w)) ||
				// the sign bit of a NaN is not portable (riscv64 makes it canonical)
				!math.IsNaN(imag(l)) && math.Signbit(imag(l)) != math.Signbit(imag(w)) {
				t.Errorf("Log(%v) = %v but Log10/Log2 = %v", z, l, w)
			}
		}
	}
}

// TestLogCut asserts the jump across the negative real axis: Im Log is +π
// above (y = +0) and -π below (y = -0), Sqrt is +i|x|^½ above and -i below.
func TestLogCut(t *testing.T) {
	for _, x := range []float64{-0x1p-1074, -0.5, -1, -3, -1e300, -math.MaxFloat64} {
		up, down := Log(complex(x, 0)), Log(complex(x, nzero))
		if imag(up) != math.Pi || imag(down) != -math.Pi || real(up) != real(down) {
			t.Errorf("Log(%g ± i0) = %v, %v", x, up, down)
		}
		su, sd := Sqrt(complex(x, 0)), Sqrt(complex(x, nzero))
		if real(su) != 0 || imag(su) <= 0 || imag(sd) != -imag(su) {
			t.Errorf("Sqrt(%g ± i0) = %v, %v", x, su, sd)
		}
		// math.Atan2 loses the sign of a subnormal y below a huge x.
		if w := Log(complex(float64(x*1e290), -5e-324)); x < -1e-10 && imag(w) != -math.Pi {
			t.Errorf("Log(%g - 5e-324i) = %v, want Im -π", float64(x*1e290), w)
		}
	}
}

func TestLogSymmetry(t *testing.T) {
	for _, c := range []struct {
		name string
		f    func(complex128) complex128
	}{{"log", Log}, {"log10", Log10}, {"log2", Log2}, {"sqrt", Sqrt}} {
		checkSymmetry(t, c.name, c.f, noParity, fixturePoints(t, c.name))
		checkSymmetry(t, c.name, c.f, noParity, specialGrid())
	}
}

// TestLogDefects pins the math/cmplx defects fixed here.
func TestLogDefects(t *testing.T) {
	// Log(1+1e-10i) = 5e-21 + 1e-10i; math/cmplx returns 0 for the real
	// part (log(hypot) with hypot = 1).
	if z := Log(complex(1, 1e-10)); math.Abs(real(z)/5e-21-1) > 1e-15 || imag(z) != 1e-10 {
		t.Errorf("Log(1+1e-10i) = %v, want 5e-21+1e-10i", z)
	}
	if real(std.Log(complex(1, 1e-10))) != 0 {
		t.Log("math/cmplx Log(1+1e-10i) no longer loses its real part")
	}
	if got := arg(-5e-324, -1e130); got != -math.Pi {
		t.Errorf("arg(-1e130 - 5e-324i) = %v, want -π", got)
	}
	if libm.Atan2(-5e-324, -1e130) != math.Pi {
		t.Log("math.Atan2 no longer drops the sign of an underflowed y/x")
	}
}

// TestLogConstants checks the double-double constants against 60 digits.
func TestLogConstants(t *testing.T) {
	for _, c := range []struct {
		v      dd.DD
		digits string
	}{
		{log10eDD, "0.434294481903251827651128918916605082294397005803666566114453783"},
		{log2eDD, "1.44269504088896340735992468100189213742664595415298593413544940693"},
	} {
		want, _, _ := big.ParseFloat(c.digits, 10, 300, big.ToNearestEven)
		d := new(big.Float).SetPrec(300).Sub(c.v.Big(), want)
		if d.Abs(d).Cmp(new(big.Float).SetMantExp(want, -104)) > 0 {
			t.Errorf("%v differs from %s by %g", c.v, c.digits, d)
		}
	}
}

func FuzzLog(f *testing.F) {
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
		}{{"Log", Log}, {"Log10", Log10}, {"Log2", Log2}, {"Sqrt", Sqrt}} {
			w := c.f(z)
			if !sameC(c.f(Conj(z)), Conj(w)) {
				t.Fatalf("%s(conj %v) != conj", c.name, z)
			}
			if finite(x) && finite(y) && (math.IsNaN(real(w)) || math.IsNaN(imag(w))) {
				t.Fatalf("%s(%v) = %v", c.name, z, w)
			}
		}
		if !finite(x) || !finite(y) || z == 0 {
			return
		}
		// Exp(Log z) = z within the conditioning of Exp (an ulp of each part
		// of Log z is a relative error of up to |Log z|·2^-53 in z), and
		// Sqrt(z)² = z within a few ulp of |z|.
		if a := Abs(z); a > 1e-300 && a < 1e300 {
			if d := Abs(Exp(Log(z)) - z); d > 4e-16*a*(2+Abs(Log(z))) {
				t.Fatalf("Exp(Log(%v)) off by %g", z, d/a)
			}
			s := Sqrt(z)
			if d := Abs(Mul(s, s) - z); d > 8e-16*a {
				t.Fatalf("Sqrt(%v)² off by %g", z, d/a)
			}
			if real(s) < 0 {
				t.Fatalf("Sqrt(%v) = %v: negative real part", z, s)
			}
		}
	})
}
