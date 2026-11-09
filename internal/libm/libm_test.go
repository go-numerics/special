package libm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Each function against package math, which runs the same algorithm with
// fused multiply-adds (arm64, s390x, ...) or an assembly version (amd64,
// s390x): the two roundings agree to a few ulps — more where the algorithm
// cancels (Acos near 1, Asin, Tan near its poles, Pow's large results) —
// which catches a transcription error; the bits themselves are pinned
// across architectures by special's TestArchHash.

func ulpDiff(a, b float64) float64 {
	if a == b || (math.IsNaN(a) && math.IsNaN(b)) {
		return 0
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) || math.IsNaN(a) || math.IsNaN(b) {
		return math.Inf(1)
	}
	u := math.Abs(math.Nextafter(b, math.Inf(1)) - b)
	if u == 0 {
		u = math.SmallestNonzeroFloat64
	}
	return math.Abs(a-b) / u
}

var specials = []float64{
	math.Inf(-1), -1e308, -1e300, -1e30, -1000, -710, -100, -30, -21.5, -2,
	-1, -0.999999, -0.75, -0.5, -0.3, -1e-10, -1e-300, math.Copysign(0, -1), 0,
	5e-324, 1e-300, 1e-20, 1e-10, 0.25, 0.3, 0.5, 0.6, 0.7, 0.75, 0.9, 1,
	1.0000001, 1.5, 2, 2.5, 3, 10, 21.5, 22, 30, 100, 700, 709.7, 710, 1e3,
	1023.9, 1024, 1e5, 1 << 29, 1 << 30, 1e10, 1 << 53, 1 << 63, 1e300,
	1.7e308, math.MaxFloat64, math.Inf(1), math.NaN(),
}

// tol is the agreement required with package math, in ulps.
var tol = map[string]float64{
	"Exp": 2, "Exp2": 2, "Expm1": 2, "Log": 2, "Log1p": 2, "Sin": 2, "Cos": 2,
	"Atan": 2, "Atanh": 2, "Cosh": 2, "Sinh": 4, "Tan": 16, "Asin": 64,
	"Acos": 64, "Pow": 64, "Atan2": 4, "Hypot": 4,
}

func check1(t *testing.T, name string, f, ref func(float64) float64, lo, hi float64) {
	t.Helper()
	ulpDiff := ulpDiff
	if name == "Asin" || name == "Acos" {
		// Pi/2 - asin cancels and 1 - x*x loses digits near |x| = 1: the
		// two roundings differ by up to ~1e-14 absolutely there, so the
		// difference is measured in units of ulp(1).
		ulpDiff = func(a, b float64) float64 {
			if a == b || (math.IsNaN(a) && math.IsNaN(b)) {
				return 0
			}
			return math.Abs(a-b) / 0x1p-52
		}
	}
	rng := rand.New(rand.NewPCG(1, 2))
	bad := 0
	for i := 0; i < 20000; i++ {
		x := lo + float64((hi-lo)*rng.Float64())
		if d := ulpDiff(f(x), ref(x)); d > tol[name] && bad < 3 {
			t.Errorf("%s(%v) = %v, math %v (%g ulp)", name, x, f(x), ref(x), d)
			bad++
		}
	}
	for _, x := range specials {
		got, want := f(x), ref(x)
		if d := ulpDiff(got, want); d > tol[name] {
			t.Errorf("%s(%v) = %v, math %v", name, x, got, want)
		}
	}
}

func TestUnary(t *testing.T) {
	for _, c := range []struct {
		name   string
		f, ref func(float64) float64
		lo, hi float64
	}{
		{"Exp", Exp, math.Exp, -745, 709},
		{"Exp2", Exp2, math.Exp2, -1074, 1023},
		{"Expm1", Expm1, math.Expm1, -40, 709},
		{"Log", Log, math.Log, 0, 1e300},
		{"Log1p", Log1p, math.Log1p, -1, 1e10},
		{"Sin", Sin, math.Sin, -1e6, 1e6},
		{"Cos", Cos, math.Cos, -1e6, 1e6},
		{"Tan", Tan, math.Tan, -1e6, 1e6},
		{"Asin", Asin, math.Asin, -1, 1},
		{"Acos", Acos, math.Acos, -1, 1},
		{"Atan", Atan, math.Atan, -1e3, 1e3},
		{"Atanh", Atanh, math.Atanh, -1, 1},
		{"Sinh", Sinh, math.Sinh, -710, 710},
		{"Cosh", Cosh, math.Cosh, -710, 710},
	} {
		check1(t, c.name, c.f, c.ref, c.lo, c.hi)
		check1(t, c.name, c.f, c.ref, -1, 1)
	}
	// Huge arguments take the Payne–Hanek reduction.
	for _, x := range []float64{1 << 30, 1e20, 1e300, -1e300, 3.5e18} {
		for _, p := range []struct {
			n      string
			f, ref func(float64) float64
		}{{"Sin", Sin, math.Sin}, {"Cos", Cos, math.Cos}, {"Tan", Tan, math.Tan}} {
			if d := ulpDiff(p.f(x), p.ref(x)); d > tol[p.n] {
				t.Errorf("%s(%v) = %v, math %v", p.n, x, p.f(x), p.ref(x))
			}
		}
		s, c := Sincos(x)
		ms, mc := math.Sincos(x)
		if ulpDiff(s, ms) > 2 || ulpDiff(c, mc) > 2 {
			t.Errorf("Sincos(%v) = %v, %v; math %v, %v", x, s, c, ms, mc)
		}
	}
}

func TestSincos(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	xs := append([]float64{}, specials...)
	for i := 0; i < 20000; i++ {
		xs = append(xs, 2e6*rng.Float64()-1e6)
	}
	for _, x := range xs {
		s, c := Sincos(x)
		ms, mc := math.Sincos(x)
		if ulpDiff(s, ms) > 2 || ulpDiff(c, mc) > 2 {
			t.Errorf("Sincos(%v) = %v, %v; math %v, %v", x, s, c, ms, mc)
			return
		}
	}
}

func TestBinary(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 6))
	for _, c := range []struct {
		name   string
		f, ref func(a, b float64) float64
	}{
		{"Pow", Pow, math.Pow},
		{"Atan2", Atan2, math.Atan2},
		{"Hypot", Hypot, math.Hypot},
	} {
		for _, a := range specials {
			for _, b := range specials {
				if d := ulpDiff(c.f(a, b), c.ref(a, b)); d > tol[c.name] {
					t.Errorf("%s(%v, %v) = %v, math %v", c.name, a, b, c.f(a, b), c.ref(a, b))
				}
			}
		}
		for i := 0; i < 20000; i++ {
			a, b := 100*rng.Float64()-20, 60*rng.Float64()-30
			if d := ulpDiff(c.f(a, b), c.ref(a, b)); d > tol[c.name] {
				t.Errorf("%s(%v, %v) = %v, math %v", c.name, a, b, c.f(a, b), c.ref(a, b))
				break
			}
		}
	}
	// Pow: integer and half-integer exponents, odd/even negative bases,
	// results that overflow or underflow through the exponent accumulation.
	for _, xy := range [][2]float64{
		{-2, 3}, {-2, 4}, {-2, -3}, {2, 0.5}, {2, -0.5}, {-8, 1.0 / 3},
		{0, -3}, {math.Copysign(0, -1), -3}, {math.Copysign(0, -1), 3},
		{math.Copysign(0, -1), -2}, {math.Copysign(0, -1), 0.5}, {-1, math.Inf(1)},
		{2, 1500}, {2, -1500}, {1.5, 1e6}, {0.5, 1e6}, {10, 400}, {10, -400},
		{math.Inf(-1), 3}, {math.Inf(-1), -3}, {math.Inf(-1), 2}, {math.Inf(-1), -2},
		{1e-300, -2}, {3, 1 << 62}, {-3, 1 << 62}, {0.9, 1 << 62}, {1.1, -(1 << 62)},
		{2, 1<<63 + 0}, {-2, 1e19}, {1.0000001, 1e15}, {0.5, -1100},
	} {
		if d := ulpDiff(Pow(xy[0], xy[1]), math.Pow(xy[0], xy[1])); d > tol["Pow"] {
			t.Errorf("Pow(%v, %v) = %v, math %v", xy[0], xy[1], Pow(xy[0], xy[1]), math.Pow(xy[0], xy[1]))
		}
	}
}

func TestTrigReduceSmall(t *testing.T) {
	if j, z := trigReduce(0.5); j != 0 || z != 0.5 {
		t.Errorf("trigReduce(0.5) = %d, %v", j, z)
	}
}

func TestGlue(t *testing.T) {
	if Trunc(-2.5) != -2 || Floor(-2.5) != -3 || Signbit(1) || !Signbit(math.Copysign(0, -1)) {
		t.Error("glue")
	}
	if i, f := Modf(2.25); i != 2 || f != 0.25 {
		t.Error("Modf")
	}
	if Float64frombits(Float64bits(1.5)) != 1.5 || Copysign(2, -1) != -2 || Abs(-3) != 3 {
		t.Error("bits")
	}
}
