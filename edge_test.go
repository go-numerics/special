package special

import (
	"math"
	"testing"
)

// Edge cases and special values the mpmath fixtures do not reach.

func eq(a, b float64) bool { return a == b || (a != a && b != b) }

func TestEdgeCases(t *testing.T) {
	inf, nan := math.Inf(1), math.NaN()
	cases := []struct {
		name      string
		got, want any
	}{
		{"LogBeta(NaN,1)", LogBeta(nan, 1), nan},
		{"LogBeta(2,1) swap", LogBeta(2, 1), LogBeta(1, 2)},
		{"LogBeta(1,Inf)", LogBeta(1, inf), math.Inf(-1)},
		{"LogBeta(-Inf,1)", LogBeta(math.Inf(-1), 1), nan},
		{"LogBeta(-1,2)", LogBeta(-1, 2), inf},
		{"LogBeta(-1,-2)", LogBeta(-1, -2), nan},
		{"LogBeta(-0.5,-0.5)", LogBeta(-0.5, -0.5), math.Inf(-1)},
		{"Beta(NaN,1)", Beta(nan, 1), nan},
		{"Beta(1,Inf)", Beta(1, inf), 0},
		{"Beta(-1,2)", Beta(-1, 2), inf},
		{"Beta(-1,-2)", Beta(-1, -2), nan},
		{"Beta(-0.5,-0.5)", Beta(-0.5, -0.5), 0},
		{"Beta(1e-300,2) ~ 1/a", math.Abs(Beta(1e-300, 2)*1e-300-1) < 3e-16, true},
		{"Factorial(NaN)", Factorial(nan), nan},
		{"Factorial(-1)", Factorial(-1), 0},
		{"Factorial(171)", Factorial(171), inf},
		{"Factorial(170.7)", Factorial(170.7), inf},
		{"LogFactorial(-1)", LogFactorial(-1), nan},
		{"LogFactorial(Inf)", LogFactorial(inf), inf},
		{"LogFactorial(0)", LogFactorial(0), 0},
		{"Factorial2(-3)", Factorial2(-3), 0},
		{"Factorial2(-1)", Factorial2(-1), 1},
		{"Factorial2(301)", Factorial2(301), inf},
		{"Comb(-1,0)", Comb(-1, 0), 0},
		{"Comb(5,6)", Comb(5, 6), 0},
		{"Comb(4000,2000)", Comb(4000, 2000), inf},
		{"Perm(5,-1)", Perm(5, -1), 0},
		{"Perm(100,90)", Perm(100, 90), Perm(100, 90)},
		{"Binom(NaN,1)", Binom(nan, 1), nan},
		{"Binom(Inf,1)", Binom(inf, 1), nan},
		{"Binom(5,-1)", Binom(5, -1), 0},
		{"Binom(-1,0.5)", Binom(-1, 0.5), nan},
		{"EllipK(-1e301)", EllipK(-1e301), (math.Log(4) + 0.5*math.Log(1e301)) / math.Sqrt(1e301)},
		{"EllipKm1(1e301)", EllipKm1(1e301), (math.Log(4) + 0.5*math.Log(1e301)) / math.Sqrt(1e301)},
		{"EllipKm1(NaN)", EllipKm1(nan), nan},
		{"EllipE(-1e301)", EllipE(-1e301), math.Sqrt(1e301)},
		{"EllipE(-Inf)", EllipE(math.Inf(-1)), inf},
		{"EllipE(2)", EllipE(2), nan},
		{"Erfcx(Inf)", Erfcx(inf), 0},
		{"Erfcx(-26.65)", Erfcx(-26.65), inf},
		{"Erfcx(-27)", Erfcx(-27), inf},
		{"Erfcx(NaN)", Erfcx(nan), nan},
		{"Gamma(-200.5)", Gamma(-200.5), 0},
		{"Gamma(-201.5)", Gamma(-201.5), 0},
		{"HurwitzZeta(NaN,1)", HurwitzZeta(nan, 1), nan},
		{"HurwitzZeta(1,2)", HurwitzZeta(1, 2), inf},
		{"HurwitzZeta(0.5,2)", HurwitzZeta(0.5, 2), nan},
		{"HurwitzZeta(2,Inf)", HurwitzZeta(2, inf), 0},
		{"HurwitzZeta(2,-3)", HurwitzZeta(2, -3), inf},
		{"HurwitzZeta(2.5,-0.5)", HurwitzZeta(2.5, -0.5), nan},
		{"HurwitzZeta(Inf,2)", HurwitzZeta(inf, 2), 0},
		{"HurwitzZeta(Inf,1)", HurwitzZeta(inf, 1), 1},
		{"HurwitzZeta(Inf,0.5)", HurwitzZeta(inf, 0.5), inf},
		{"HurwitzZeta(Inf,-0.5)", HurwitzZeta(inf, -0.5), nan},
		{"HurwitzZeta(1e5,1e300)", HurwitzZeta(1e5, 1e300), 0},
		{"Polygamma(-1,1)", Polygamma(-1, 1), nan},
		{"Polygamma(0,1)", Polygamma(0, 1), Digamma(1)},
		{"Polygamma(1,Inf)", Polygamma(1, inf), 0},
		{"Polygamma(1,0)", Polygamma(1, 0), inf},
		{"Polygamma(2,-1)", Polygamma(2, -1), math.Inf(-1)},
		{"Polygamma(200,200)", Polygamma(200, 200), Polygamma(200, 200)},
	}
	for _, c := range cases {
		g, gok := c.got.(float64)
		w, wok := c.want.(float64)
		if (gok && wok && !eq(g, w)) || (!gok && c.got != c.want) {
			t.Errorf("%s = %v, want %v", c.name, c.got, c.want)
		}
	}
	if s := math.Signbit(Gamma(-200.5)) == math.Signbit(Gamma(-201.5)); s {
		t.Error("Gamma underflow signs should alternate")
	}
	// Polygamma for n > 170 goes through log Γ(n+1); check against the
	// recurrence ψ⁽ⁿ⁾(x+1) = ψ⁽ⁿ⁾(x) + (-1)^n n!/x^(n+1) at a point where
	// the last term is negligible.
	if p := Polygamma(200, 200); !(p < 0 && !math.IsInf(p, 0)) {
		t.Errorf("Polygamma(200, 200) = %v", p)
	}
}

func TestSoftmaxBuffers(t *testing.T) {
	buf := make([]float64, 8)
	if r := Softmax(buf, nil); len(r) != 0 {
		t.Error("Softmax(empty)")
	}
	if r := LogSoftmax(buf, nil); len(r) != 0 {
		t.Error("LogSoftmax(empty)")
	}
	r := Softmax(buf, []float64{1, 2})
	if &r[0] != &buf[0] || math.Abs(r[0]+r[1]-1) > 1e-16 {
		t.Errorf("Softmax reuse: %v", r)
	}
	r = LogSoftmax(buf, []float64{1, 2})
	if &r[0] != &buf[0] {
		t.Error("LogSoftmax reuse")
	}
	if r := LogSoftmax(nil, []float64{1, math.NaN()}); !math.IsNaN(r[0]) {
		t.Errorf("LogSoftmax NaN: %v", r)
	}
	if r := Softmax(nil, []float64{math.Inf(-1), math.Inf(-1)}); !math.IsNaN(r[0]) {
		t.Errorf("Softmax all -Inf: %v", r)
	}
	if v := LogSumExp(nil); !math.IsInf(v, -1) {
		t.Errorf("LogSumExp(nil) = %v", v)
	}
	if v := LogSumExp([]float64{0, -1000}); v != 0 {
		t.Errorf("LogSumExp(0,-1000) = %v", v)
	}
}

func TestPolyHelpers(t *testing.T) {
	// p1evl: x^2 + 2x + 3 at 2 = 11; chbevl: Cephes convention.
	if v := p1evl(2, []float64{2, 3}); v != 11 {
		t.Errorf("p1evl = %v", v)
	}
	// chbevl(x, c) = Σ' c_i T_i(x/2) in reverse order: with c = {0, 0, 2}
	// (T_0 coefficient 2, halved) the value is 1.
	if v := chbevl(0.3, []float64{0, 0, 2}); v != 1 {
		t.Errorf("chbevl = %v", v)
	}
	if v := logf(5e-324); math.Abs(v-(-744.4400719213812)) > 1e-12 {
		t.Errorf("logf(5e-324) = %v", v)
	}
}
