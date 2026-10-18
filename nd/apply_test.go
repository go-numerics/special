package nd

import (
	"errors"
	"math"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

// The direct loops of apply1/apply2 against Map1/Map2 (bit for bit), and
// their fallbacks and errors.
func TestApplyPaths(t *testing.T) {
	xs := []float64{0.5, 1.5, 3, 12, 40, 0.1}
	x, _ := ndarray.FromData(xs, 2, 3)
	one, _ := ndarray.FromData([]float64{1}, 1)
	two, _ := ndarray.FromData([]float64{2}, 1)
	row, _ := ndarray.FromData([]float64{0.5, 1.5, 3}, 1, 3)
	same := func(name string, got, want *ndarray.Array) {
		t.Helper()
		g, _ := ndarray.Data[float64](got)
		w, _ := ndarray.Data[float64](want)
		if len(g) != len(w) {
			t.Fatalf("%s: %d elements, want %d", name, len(g), len(w))
		}
		for i := range g {
			if math.Float64bits(g[i]) != math.Float64bits(w[i]) {
				t.Errorf("%s[%d] = %v, want %v", name, i, g[i], w[i])
			}
		}
	}
	for _, c := range []struct {
		name string
		a, b *ndarray.Array
	}{{"same", x, x}, {"scalar a", one, x}, {"scalar b", x, two}, {"broadcast", row, x}, {"rank", xs1(), x16()}} {
		got, err := Beta(c.a, c.b)
		if err != nil {
			t.Fatal(err)
		}
		want, _ := Map2(special.Beta, c.a, c.b)
		same("Beta "+c.name, got, want)
	}
	got, _ := Polygamma(one, x)
	want, _ := Map2(func(n, x float64) float64 { return special.Polygamma(int(n), x) }, one, x)
	same("Polygamma(1, x)", got, want)
	got, _ = Polygamma(two, x)
	want, _ = Map2(func(n, x float64) float64 { return special.Polygamma(int(n), x) }, two, x)
	same("Polygamma(2, x)", got, want)

	cplx := x.AsType(ndarray.Complex128)
	if _, err := Map1(math.Sin, cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Map1(complex) err %v", err)
	}
	if _, err := Map2(math.Atan2, cplx, x); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Map2(complex, _) err %v", err)
	}
	if _, err := Beta(x, cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Beta(_, complex) err %v", err)
	}
	if _, err := Beta(cplx, x); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("Beta(complex, _) err %v", err)
	}
	bad, _ := ndarray.FromData([]float64{1, 2}, 2)
	if _, err := Map2(math.Atan2, bad, x); !errors.Is(err, ErrShape) {
		t.Errorf("Map2 shape err %v", err)
	}
	if _, err := Beta(bad, x); !errors.Is(err, ErrShape) {
		t.Errorf("Beta shape err %v", err)
	}
}

// xs1 is a rank-1 array of the same six elements as the 2x3 x.
func xs1() *ndarray.Array {
	a, _ := ndarray.FromData([]float64{2, 2.5, 3, 3.5, 4, 4.5}, 6)
	return a
}

// x16 is a 1x6 array.
func x16() *ndarray.Array {
	a, _ := ndarray.FromData([]float64{0.5, 1.5, 3, 12, 40, 0.1}, 1, 6)
	return a
}
