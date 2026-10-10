package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/internal/dd"
)

// Special values and the branches the accuracy fixtures do not reach.

func same(a, b float64) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.IsNaN(a) && math.IsNaN(b)
	}
	return a == b && math.Signbit(a) == math.Signbit(b)
}

func TestBesselSpecialValues(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	cases := []struct {
		name      string
		got, want float64
	}{
		{"J0(NaN)", J0(nan), nan},
		{"J0(-Inf)", J0(-inf), 0},
		{"J1(-Inf)", J1(-inf), 0},
		{"J1(NaN)", J1(nan), nan},
		{"Y0(-1)", Y0(-1), nan},
		{"Y1(0)", Y1(0), -inf},
		{"Y1(Inf)", Y1(inf), 0},
		{"Y1(5e-324)", Y1(5e-324), -inf},
		{"Jn(3,NaN)", Jn(3, nan), nan},
		{"Jn(-3,0)", Jn(-3, 0), math.Copysign(0, -1)},
		{"Jn(4,Inf)", Jn(4, inf), 0},
		{"Yn(-3,0)", Yn(-3, 0), inf},
		{"Yn(4,Inf)", Yn(4, inf), 0},
		{"Yn(2,-1)", Yn(2, -1), nan},
		{"Jv(Inf,1)", Jv(inf, 1), 0},
		{"Jv(Inf,Inf)", Jv(inf, inf), nan},
		{"Jv(Inf,-1)", Jv(inf, -1), nan},
		{"Jv(-Inf,1)", Jv(math.Inf(-1), 1), nan},
		{"Jv(0.5,-1)", Jv(0.5, -1), nan},
		{"Jv(2.5,0)", Jv(2.5, 0), 0},
		{"Jv(-0.5,0)", Jv(-0.5, 0), inf},
		{"Jv(-1.5,0)", Jv(-1.5, 0), -inf},
		{"Jv(2.5,Inf)", Jv(2.5, inf), 0},
		{"Yv(Inf,1)", Yv(inf, 1), -inf},
		{"Yv(Inf,Inf)", Yv(inf, inf), nan},
		{"Yv(-Inf,1)", Yv(math.Inf(-1), 1), nan},
		{"Yv(2.5,0)", Yv(2.5, 0), -inf},
		{"Yv(-0.5,0)", Yv(-0.5, 0), 0},
		{"Yv(-1.25,0)", Yv(-1.25, 0), inf},
		{"Yv(-2.25,0)", Yv(-2.25, 0), -inf},
		{"Yv(2.5,Inf)", Yv(2.5, inf), 0},
		{"I0(-Inf)", I0(-inf), inf},
		{"I1(-Inf)", I1(-inf), -inf},
		{"I0e(NaN)", I0e(nan), nan},
		{"I1e(-Inf)", I1e(-inf), math.Copysign(0, -1)},
		{"I1e(NaN)", I1e(nan), nan},
		{"K0(-1)", K0(-1), nan},
		{"K1(0)", K1(0), inf},
		{"K0e(0)", K0e(0), inf},
		{"K1e(-1)", K1e(-1), nan},
		{"In(2,0)", In(2, 0), 0},
		{"In(-3,-1)", In(-3, -1), -In(3, 1)},
		{"Iv(Inf,1)", Iv(inf, 1), 0},
		{"Iv(Inf,Inf)", Iv(inf, inf), nan},
		{"Iv(-Inf,1)", Iv(math.Inf(-1), 1), nan},
		{"Iv(0.5,-1)", Iv(0.5, -1), nan},
		{"Iv(3,-Inf)", Iv(3, math.Inf(-1)), math.Inf(-1)},
		{"Ive(2,-Inf)", Ive(2, math.Inf(-1)), 0},
		{"Ive(3,-Inf)", Ive(3, math.Inf(-1)), math.Copysign(0, -1)},
		{"Iv(-0.5,0)", Iv(-0.5, 0), inf},
		{"Iv(-1.5,0)", Iv(-1.5, 0), -inf},
		{"Iv(2.5,0)", Iv(2.5, 0), 0},
		{"Iv(0,0)", Iv(0, 0), 1},
		{"Ive(1,-2)", Ive(1, -2), -I1e(2)},
		{"Kn(0,0)", Kn(0, 0), inf},
		{"Kv(2,Inf)", Kv(2, inf), 0},
		{"Kv(Inf,Inf)", Kv(inf, inf), nan},
		{"Kv(Inf,1)", Kv(inf, 1), inf},
		{"Kv(NaN,1)", Kv(nan, 1), nan},
		{"Kve(1,-1)", Kve(1, -1), nan},
	}
	for _, c := range cases {
		if !same(c.got, c.want) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
}

func TestBesselIntegerOrders(t *testing.T) {
	// The integer-order entry points agree with the general ones.
	for _, n := range []int{-7, -2, 2, 3, 12, 60} {
		for _, x := range []float64{0.3, 2.5, 17, 140} {
			if a, b := In(n, x), Iv(float64(n), x); a != b {
				t.Errorf("In(%d,%g) = %g, Iv = %g", n, x, a, b)
			}
			if a, b := Kn(n, x), Kv(float64(n), x); a != b {
				t.Errorf("Kn(%d,%g) = %g, Kv = %g", n, x, a, b)
			}
		}
	}
	// K_n for n > 50 goes through the general path; check K_{n+1} =
	// 2n/x K_n + K_{n-1} there.
	x := 30.0
	k49, k50, k51 := Kv(49, x), Kv(50, x), Kv(51, x)
	if r := (2*50/x*k50 + k49) / k51; math.Abs(r-1) > 1e-14 {
		t.Errorf("K recurrence at n=50: ratio %v", r)
	}
	if v := Kve(60, 30); math.Abs(v/(Kv(60, 30)*math.Exp(30))-1) > 1e-14 {
		t.Errorf("Kve(60,30) = %g", v)
	}
}

func TestBesselHugeOrder(t *testing.T) {
	// Beyond besselMaxOrder only the series and the large-x expansion run.
	v := 2e7 + 0.5
	if j := Jv(v, 1); j != 0 {
		t.Errorf("Jv(2e7+0.5, 1) = %g, want 0 (underflow)", j)
	}
	if y := Yv(v, 1); !math.IsInf(y, -1) {
		t.Errorf("Yv(2e7+0.5, 1) = %g, want -Inf", y)
	}
	if j := Jv(v, 1e7); !math.IsNaN(j) {
		t.Errorf("Jv(2e7+0.5, 1e7) = %g, want NaN (not supported)", j)
	}
	if i := Iv(v, 1); i != 0 {
		t.Errorf("Iv(2e7+0.5, 1) = %g", i)
	}
	if k := Kv(v, 1); !math.IsInf(k, 1) {
		t.Errorf("Kv(2e7+0.5, 1) = %g", k)
	}
	if i := Iv(-v, 5000); !math.IsInf(i, 1) {
		t.Errorf("Iv(-2e7-0.5, 5000) = %g, want +Inf", i)
	}
	if k := Kv(v, 1e7); !math.IsNaN(k) {
		t.Errorf("Kv(2e7+0.5, 1e7) = %g, want NaN (not supported)", k)
	}
	if i := Iv(v, 1e7); !math.IsNaN(i) {
		t.Errorf("Iv(2e7+0.5, 1e7) = %g, want NaN (not supported)", i)
	}
}

func TestBesselNegativeOrderReflection(t *testing.T) {
	// I_{-20.3}(5): the power series of I_{20.3} applies, but the series of
	// I_{-20.3} would cancel, so the reflection formula runs (mpmath value).
	const want = 463785078.28438944
	if got := Iv(-20.3, 5); math.Abs(got-want) > 4*ulp(want) {
		t.Errorf("Iv(-20.3, 5) = %.17g, want %.17g", got, want)
	}
}

func TestBesselExponentClamp(t *testing.T) {
	// log-prefixes beyond +-1e5 saturate before any int conversion
	if m, e := besselSeries(9e6, 1e-300, 1, false); m != 0 || e != 0 {
		t.Errorf("besselSeries underflow = %g, %d", m, e)
	}
	if m, _ := besselSeries(-9e6-0.5, 1e-300, 1, false); !math.IsInf(m, 0) {
		t.Errorf("besselSeries overflow = %g", m)
	}
	if r := besselExpMul(dd.F(2e5), -1); !math.IsInf(r, -1) {
		t.Errorf("besselExpMul(2e5) = %g", r)
	}
	if r := besselExpMul(dd.F(-2e5), 1); r != 0 {
		t.Errorf("besselExpMul(-2e5) = %g", r)
	}
}

func TestBesselTinyArguments(t *testing.T) {
	// Y overflows for an integer order and a tiny argument.
	if y := Yn(3, 1e-300); !math.IsInf(y, -1) {
		t.Errorf("Yn(3,1e-300) = %g", y)
	}
	if y := Yn(2, 0x1p-950); !math.IsInf(y, -1) {
		t.Errorf("Yn(2,2^-950) = %g", y)
	}
	// Y_1.25 at 1e-200 is finite (about -2.5e249) and reached through
	// rescaled recurrences.
	y := Yv(1.25, 1e-200)
	if !(y < -1e249 && y > -1e250) {
		t.Errorf("Yv(1.25,1e-200) = %g", y)
	}
}

func TestBesselReduce(t *testing.T) {
	// x*(2/pi) = 4N + q + f against the float64 product for moderate x,
	// and the exact reduction of a few large arguments.
	for _, x := range []float64{5e-324, 1e-310, 1e-5, 0.7, 1, 3, 100, 12345.678} {
		q, f := besselReduce(x)
		got := float64(q) + f.Hi
		want := math.Mod(x*2/math.Pi, 4)
		if want-got > 2 {
			got += 4
		}
		if got-want > 2 {
			got -= 4
		}
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("besselReduce(%g) = %d + %g, want %g", x, q, f.Hi, want)
		}
	}
	// sin(x) = sin((q + f) pi/2) for huge x, against math.Sin (which
	// reduces exactly too).
	for _, x := range []float64{1e22, 3.7e150, math.MaxFloat64} {
		q, f := besselReduce(x)
		s, _ := besselSinCosQuarter(q, f)
		if w := math.Sin(x); math.Abs(s-w) > 1e-15 {
			t.Errorf("sin via besselReduce(%g) = %g, want %g", x, s, w)
		}
	}
}

func TestBesselCF1LongTail(t *testing.T) {
	// For x far above v CF1 converges only after O(x) levels; beyond
	// 100000 levels the tail is evaluated in float64.
	v, x := 10.5, 1e5
	fv, _ := besselCF1JY(v, x)
	j0, _, _ := besselAsymJY(v, x, true, false)
	j1, _, _ := besselAsymJY(v+1, x, true, false)
	if r := fv.Float() / (j1 / j0); math.Abs(r-1) > 1e-11 {
		t.Errorf("CF1 at x=1e5: ratio %v", r)
	}
}

func TestJnZeros(t *testing.T) {
	if z := JnZeros(3, 0); len(z) != 0 {
		t.Errorf("JnZeros(3,0) = %v", z)
	}
	a, b := JnZeros(-4, 6), JnZeros(4, 6)
	for i := range a {
		if a[i] != b[i] {
			t.Errorf("JnZeros(-4) differs from JnZeros(4) at %d", i)
		}
	}
	for n := 0; n <= 3; n++ {
		for _, z := range JnZeros(n, 40) {
			if math.Abs(Jn(n, z)) > 1e-15 {
				t.Errorf("J_%d(%g) = %g", n, z, Jn(n, z))
			}
		}
	}
}
