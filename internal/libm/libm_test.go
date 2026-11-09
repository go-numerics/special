package libm

import (
	"math"
	"math/rand/v2"
	"testing"
)

// Each function against package math on ordinary arguments: math runs the
// same algorithm with fused multiply-adds (arm64, s390x, ...) or an assembly
// version (amd64, s390x), so the two roundings agree to a few ulps — more
// where the algorithm cancels (Acos and Asin near ±1, Tan near its poles,
// Pow's large results) — which catches a transcription error. The bits
// themselves are pinned across architectures by special's TestArchHash.
// The extremes (overflow edges, subnormal arguments) are checked against
// fixed values instead, because math's assembly versions get some of them
// wrong (amd64 Log of a subnormal, s390x Sinh of large negative arguments).

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

// absDiff measures |a - b| in units of ulp(1), for the functions whose
// formula cancels near the ends of their domain.
func absDiff(a, b float64) float64 {
	if a == b || (math.IsNaN(a) && math.IsNaN(b)) {
		return 0
	}
	return float64(math.Abs(a-b) / 0x1p-52)
}

// tol is the agreement required with package math, in ulps.
var tol = map[string]float64{
	"Exp": 2, "Exp2": 2, "Expm1": 2, "Log": 2, "Log1p": 2, "Sin": 2, "Cos": 2,
	"Atan": 2, "Atanh": 2, "Cosh": 2, "Sinh": 4, "Tan": 16, "Asin": 64,
	"Acos": 64, "Pow": 64, "Atan2": 4, "Hypot": 4,
}

// ordinary are arguments every implementation handles alike.
var ordinary = []float64{
	-1000, -100, -30, -21.5, -2, -1, -0.999999, -0.75, -0.5, -0.3, -1e-10,
	math.Copysign(0, -1), 0, 1e-20, 1e-10, 0.25, 0.3, 0.5, 0.6, 0.7, 0.75,
	0.9, 1, 1.0000001, 1.5, 2, 2.5, 3, 10, 21.5, 22, 30, 100, 1e5, 1 << 29,
	math.Inf(-1), math.Inf(1), math.NaN(),
}

func check1(t *testing.T, name string, f, ref func(float64) float64, lo, hi float64) {
	t.Helper()
	diff := ulpDiff
	if name == "Asin" || name == "Acos" {
		diff = absDiff
	}
	rng := rand.New(rand.NewPCG(1, 2))
	bad := 0
	for i := 0; i < 20000; i++ {
		x := lo + float64((hi-lo)*rng.Float64())
		if d := diff(f(x), ref(x)); d > tol[name] && bad < 3 {
			t.Errorf("%s(%v) = %v, math %v (%g ulp)", name, x, f(x), ref(x), d)
			bad++
		}
	}
	for _, x := range ordinary {
		if x < lo || x > hi {
			continue
		}
		if d := diff(f(x), ref(x)); d > tol[name] {
			t.Errorf("%s(%v) = %v, math %v", name, x, f(x), ref(x))
		}
	}
}

func TestUnary(t *testing.T) {
	inf := math.Inf(1)
	for _, c := range []struct {
		name   string
		f, ref func(float64) float64
		lo, hi float64
	}{
		{"Exp", Exp, math.Exp, -700, 700},
		{"Exp2", Exp2, math.Exp2, -1000, 1000},
		{"Expm1", Expm1, math.Expm1, -40, 700},
		{"Log", Log, math.Log, 1e-300, inf},
		{"Log1p", Log1p, math.Log1p, -1, inf},
		{"Sin", Sin, math.Sin, -1e6, 1e6},
		{"Cos", Cos, math.Cos, -1e6, 1e6},
		{"Tan", Tan, math.Tan, -1e6, 1e6},
		{"Asin", Asin, math.Asin, -1, 1},
		{"Acos", Acos, math.Acos, -1, 1},
		{"Atan", Atan, math.Atan, -inf, inf},
		{"Atanh", Atanh, math.Atanh, -1, 1},
		{"Sinh", Sinh, math.Sinh, -700, 700},
		{"Cosh", Cosh, math.Cosh, -700, 700},
	} {
		lo, hi := math.Max(c.lo, -1e6), math.Min(c.hi, 1e10)
		check1(t, c.name, c.f, c.ref, lo, hi)
		check1(t, c.name, c.f, c.ref, math.Max(c.lo, -1), math.Min(c.hi, 1))
	}
	// Payne–Hanek reduction for huge arguments (math is pure Go there on
	// every architecture but s390x; compared within the same tolerance).
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
		if s != Sin(x) || c != Cos(x) {
			t.Errorf("Sincos(%v) = %v, %v; Sin, Cos = %v, %v", x, s, c, Sin(x), Cos(x))
		}
	}
}

// TestExtremes pins the edge cases to their IEEE/C99 values.
func TestExtremes(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	sub := 5e-324
	for _, c := range []struct {
		name string
		got  float64
		want float64
	}{
		{"Exp(710)", Exp(710), inf},
		{"Exp(-746)", Exp(-746), 0},
		{"Exp(-745.1)", Exp(-745.1), 5e-324},
		{"Exp(NaN)", Exp(nan), nan},
		{"Exp2(1024)", Exp2(1024), inf},
		{"Exp2(-1075)", Exp2(-1075), 0},
		{"Expm1(710)", Expm1(710), inf},
		{"Expm1(-inf)", Expm1(-inf), -1},
		{"Log(sub)", Log(sub), -744.4400719213812},
		{"Log(0)", Log(0), -inf},
		{"Log(-1)", Log(-1), nan},
		{"Log1p(-1)", Log1p(-1), -inf},
		{"Log1p(-2)", Log1p(-2), nan},
		{"Sinh(710)", Sinh(710), inf}, // Go's pure-Go sinh: Exp(x)*0.5 overflows first
		{"Sinh(-1000)", Sinh(-1000), -inf},
		{"Cosh(-1000)", Cosh(-1000), inf},
		{"Pow(sub, 0.5)", Pow(sub, 0.5), 2.2227587494850775e-162},
		{"Pow(2, 1500)", Pow(2, 1500), inf},
		{"Pow(2, -1500)", Pow(2, -1500), 0},
		{"Pow(-8, 1/3)", Pow(-8, 1.0/3), nan},
		{"Pow(-2, 3)", Pow(-2, 3), -8},
		{"Pow(0, -3)", Pow(0, -3), inf},
		{"Pow(-0, -3)", Pow(math.Copysign(0, -1), -3), -inf},
		{"Pow(-1, inf)", Pow(-1, inf), 1},
		{"Atan2(-0, -1)", Atan2(math.Copysign(0, -1), -1), -math.Pi},
		{"Atanh(1)", Atanh(1), inf},
		{"Atanh(2)", Atanh(2), nan},
		{"Asin(2)", Asin(2), nan},
		{"Acos(2)", Acos(2), nan},
		{"Hypot(inf, nan)", Hypot(inf, nan), inf},
		{"Hypot(3, 4)", Hypot(3, 4), 5},
		{"Tan(inf)", Tan(inf), nan},
		{"Sin(inf)", Sin(inf), nan},
	} {
		if !(c.got == c.want || (math.IsNaN(c.got) && math.IsNaN(c.want))) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if Pow(2, 1023) != math.Ldexp(1, 1023) || Pow(0.5, 1074) != sub {
		t.Error("Pow at the ends of the exponent range")
	}
}

func TestSincos(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	xs := append([]float64{}, ordinary...)
	for i := 0; i < 20000; i++ {
		xs = append(xs, float64(2e6*rng.Float64())-1e6)
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
		for _, a := range ordinary {
			for _, b := range ordinary {
				if d := ulpDiff(c.f(a, b), c.ref(a, b)); d > tol[c.name] {
					t.Errorf("%s(%v, %v) = %v, math %v", c.name, a, b, c.f(a, b), c.ref(a, b))
				}
			}
		}
		for i := 0; i < 20000; i++ {
			a, b := float64(100*rng.Float64())-20, float64(60*rng.Float64())-30
			if d := ulpDiff(c.f(a, b), c.ref(a, b)); d > tol[c.name] {
				t.Errorf("%s(%v, %v) = %v, math %v", c.name, a, b, c.f(a, b), c.ref(a, b))
				break
			}
		}
	}
	// Pow: integer and half-integer exponents, negative bases, results
	// that overflow or underflow through the exponent accumulation.
	for _, xy := range [][2]float64{
		{-2, 4}, {-2, -3}, {2, 0.5}, {2, -0.5}, {math.Copysign(0, -1), 3},
		{math.Copysign(0, -1), -2}, {math.Copysign(0, -1), 0.5}, {1.5, 1e6},
		{0.5, 1e6}, {10, 400}, {10, -400}, {math.Inf(-1), 3}, {math.Inf(-1), -3},
		{math.Inf(-1), 2}, {math.Inf(-1), -2}, {1e-300, -2}, {3, 1 << 62},
		{-3, 1 << 62}, {0.9, 1 << 62}, {1.1, -(1 << 62)}, {2, 1 << 63},
		{-2, 1e19}, {1.0000001, 1e15}, {0.5, -1100},
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
