package nd

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

func arr(t *testing.T, v []float64, shape ...int) *ndarray.Array {
	t.Helper()
	a, err := ndarray.FromData(v, shape...)
	if err != nil {
		t.Fatal(err)
	}
	return a
}

func data(t *testing.T, a *ndarray.Array) []float64 {
	t.Helper()
	d, ok := ndarray.Data[float64](a.AsType(ndarray.Float64))
	if !ok {
		t.Fatal("not float64")
	}
	return d
}

func same(a, b float64) bool { return a == b || (a != a && b != b) }

var xs = []float64{-3.5, -1, -0.5, 0, 1e-300, 0.25, 0.5, 0.75, 1, 1.5, 2, 7.5, 30, math.NaN(), math.Inf(1), math.Inf(-1)}

func TestUnaryBitIdentical(t *testing.T) {
	cases := map[string]struct {
		nd func(*ndarray.Array) (*ndarray.Array, error)
		sc func(float64) float64
	}{
		"Gamma": {Gamma, special.Gamma}, "LogGamma": {LogGamma, special.LogGamma},
		"GammaSign": {GammaSign, special.GammaSign}, "Digamma": {Digamma, special.Digamma},
		"Erf": {Erf, special.Erf}, "Erfc": {Erfc, special.Erfc}, "Erfcx": {Erfcx, special.Erfcx},
		"ErfInv": {ErfInv, special.ErfInv}, "ErfcInv": {ErfcInv, special.ErfcInv},
		"Ndtr": {Ndtr, special.Ndtr}, "Ndtri": {Ndtri, special.Ndtri}, "LogNdtr": {LogNdtr, special.LogNdtr},
		"Factorial": {Factorial, special.Factorial}, "LogFactorial": {LogFactorial, special.LogFactorial},
		"Expit": {Expit, special.Expit}, "LogExpit": {LogExpit, special.LogExpit},
		"Log1pExp": {Log1pExp, special.Log1pExp}, "Logit": {Logit, special.Logit},
		"Zeta": {Zeta, special.Zeta}, "EllipK": {EllipK, special.EllipK},
		"EllipKm1": {EllipKm1, special.EllipKm1}, "EllipE": {EllipE, special.EllipE},
		"Factorial2": {Factorial2, func(x float64) float64 {
			if x != math.Trunc(x) || math.IsInf(x, 0) {
				return math.NaN()
			}
			return special.Factorial2(int(x))
		}},
	}
	a := arr(t, xs, 4, 4)
	for name, c := range cases {
		r, err := c.nd(a)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if got := r.Shape(); len(got) != 2 || got[0] != 4 || got[1] != 4 {
			t.Errorf("%s: shape %v", name, got)
		}
		for i, v := range data(t, r) {
			if w := c.sc(xs[i]); !same(v, w) {
				t.Errorf("%s(%v) = %v, scalar %v", name, xs[i], v, w)
			}
		}
	}
}

func TestBinaryBroadcast(t *testing.T) {
	cases := map[string]struct {
		nd func(a, b *ndarray.Array) (*ndarray.Array, error)
		sc func(a, b float64) float64
	}{
		"Beta": {Beta, special.Beta}, "LogBeta": {LogBeta, special.LogBeta}, "Binom": {Binom, special.Binom},
		"Xlogy": {Xlogy, special.Xlogy}, "Xlog1py": {Xlog1py, special.Xlog1py}, "HurwitzZeta": {HurwitzZeta, special.HurwitzZeta},
		"Polygamma": {Polygamma, func(n, x float64) float64 {
			if !isInt(n) {
				return math.NaN()
			}
			return special.Polygamma(int(n), x)
		}},
		"Comb": {Comb, func(n, k float64) float64 {
			if !isInt(n) || !isInt(k) {
				return math.NaN()
			}
			return special.Comb(int(n), int(k))
		}},
		"Perm": {Perm, func(n, k float64) float64 {
			if !isInt(n) || !isInt(k) {
				return math.NaN()
			}
			return special.Perm(int(n), int(k))
		}},
	}
	col := []float64{0.5, 1, 2.5, 3}
	row := []float64{-1, 0, 0.5, 2, 4.5}
	a := arr(t, col, 4, 1)
	b := arr(t, row, 5)
	for name, c := range cases {
		r, err := c.nd(a, b)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if s := r.Shape(); len(s) != 2 || s[0] != 4 || s[1] != 5 {
			t.Fatalf("%s: shape %v", name, s)
		}
		d := data(t, r)
		for i, x := range col {
			for j, y := range row {
				if w := c.sc(x, y); !same(d[i*5+j], w) {
					t.Errorf("%s(%v, %v) = %v, scalar %v", name, x, y, d[i*5+j], w)
				}
			}
		}
	}
}

func TestMap3AndDTypes(t *testing.T) {
	a := arr(t, []float64{1, 2}, 2)
	b := arr(t, []float64{3}, 1)
	c := arr(t, []float64{4, 5, 6, 7}, 2, 2)
	r, err := Map3(func(x, y, z float64) float64 { return x*100 + y*10 + z }, a, b, c)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{134, 235, 136, 237}
	for i, v := range data(t, r) {
		if v != want[i] {
			t.Errorf("Map3[%d] = %v want %v", i, v, want[i])
		}
	}
	// Float32 in -> Float32 out; integers -> Float64; all-Float32 binary
	// -> Float32; mixed -> Float64.
	f32 := a.AsType(ndarray.Float32)
	if r, _ := Gamma(f32); r.DType() != ndarray.Float32 {
		t.Errorf("Gamma(float32) dtype %v", r.DType())
	}
	if r, _ := Gamma(a.AsType(ndarray.Int32)); r.DType() != ndarray.Float64 {
		t.Errorf("Gamma(int32) dtype %v", r.DType())
	}
	if r, _ := Beta(f32, f32); r.DType() != ndarray.Float32 {
		t.Errorf("Beta(f32,f32) dtype %v", r.DType())
	}
	if r, _ := Beta(f32, a); r.DType() != ndarray.Float64 {
		t.Errorf("Beta(f32,f64) dtype %v", r.DType())
	}
	cplx := a.AsType(ndarray.Complex128)
	if _, err := Gamma(cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Gamma(complex) err %v", err)
	}
	if _, err := Beta(a, cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Beta(_, complex) err %v", err)
	}
	if _, err := Beta(a, arr(t, []float64{1, 2, 3}, 3)); !errors.Is(err, ErrShape) {
		t.Errorf("Beta shape mismatch err %v", err)
	}
	if _, err := LogSumExp(cplx); err == nil {
		t.Error("LogSumExp(complex) no error")
	}
	if _, err := Softmax(cplx); err == nil {
		t.Error("Softmax(complex) no error")
	}
	if _, err := LogSoftmax(cplx); err == nil {
		t.Error("LogSoftmax(complex) no error")
	}
}

func TestParallelAndReductions(t *testing.T) {
	n := 3*parallelMin + 7
	v := make([]float64, n)
	for i := range v {
		v[i] = -5 + 10*float64(i)/float64(n)
	}
	a := arr(t, v, n)
	r, err := Ndtr(a)
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range data(t, r) {
		if x != special.Ndtr(v[i]) {
			t.Fatalf("Ndtr[%d]", i)
		}
	}
	r2, err := Xlogy(a, arr(t, []float64{2}, 1))
	if err != nil {
		t.Fatal(err)
	}
	for i, x := range data(t, r2) {
		if x != special.Xlogy(v[i], 2) {
			t.Fatalf("Xlogy[%d]", i)
		}
	}
	small := arr(t, []float64{1, 2, 3}, 3)
	lse, err := LogSumExp(small)
	if err != nil || lse != special.LogSumExp([]float64{1, 2, 3}) {
		t.Errorf("LogSumExp = %v, %v", lse, err)
	}
	sm, err := Softmax(small.AsType(ndarray.Float32))
	if err != nil || sm.DType() != ndarray.Float32 {
		t.Errorf("Softmax: %v %v", sm, err)
	}
	ls, err := LogSoftmax(small)
	if err != nil || data(t, ls)[2] != special.LogSoftmax(nil, []float64{1, 2, 3})[2] {
		t.Errorf("LogSoftmax: %v %v", ls, err)
	}
}
