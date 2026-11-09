package nd

import (
	"errors"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// Map2 and Map3 take a direct loop for operands of one shape or a single
// element and mapN for any other broadcast; every layout gives the bits of the
// scalar function applied element by element.
func TestMapDirectLayouts(t *testing.T) {
	f2 := func(x, y float64) float64 { return float64(x*3) + y }
	f3 := func(x, y, z float64) float64 { return float64(x*5) + float64(y*3) + z }
	seq := func(n int, k float64) []float64 {
		v := make([]float64, n)
		for i := range v {
			v[i] = float64(float64(i+1) * k)
		}
		return v
	}
	type lay struct {
		name       string
		sa, sb, sc []int
	}
	for _, l := range []lay{
		{"same shape", []int{3, 4}, []int{3, 4}, []int{3, 4}},
		{"single first", []int{1}, []int{12}, []int{12}},
		{"single last", []int{12}, []int{12}, []int{1, 1}},
		{"rank differs", []int{12}, []int{1, 12}, []int{12}},
		{"real broadcast", []int{3, 1}, []int{1, 4}, []int{3, 4}},
	} {
		a := arr(t, seq(size(l.sa), 1), l.sa...)
		b := arr(t, seq(size(l.sb), 0.5), l.sb...)
		c := arr(t, seq(size(l.sc), 0.25), l.sc...)
		got2, err := Map2(f2, a, b)
		if err != nil {
			t.Fatal(l.name, err)
		}
		want2, err := mapN(func(v []float64) float64 { return f2(v[0], v[1]) }, a, b)
		if err != nil {
			t.Fatal(l.name, err)
		}
		got3, err := Map3(f3, a, b, c)
		if err != nil {
			t.Fatal(l.name, err)
		}
		want3, err := mapN(func(v []float64) float64 { return f3(v[0], v[1], v[2]) }, a, b, c)
		if err != nil {
			t.Fatal(l.name, err)
		}
		for _, p := range [][2]*ndarray.Array{{got2, want2}, {got3, want3}} {
			g, w := data(t, p[0]), data(t, p[1])
			if len(g) != len(w) || !equalShape(p[0].Shape(), p[1].Shape()) {
				t.Fatalf("%s: shape %v want %v", l.name, p[0].Shape(), p[1].Shape())
			}
			for i := range g {
				if !same(g[i], w[i]) {
					t.Errorf("%s[%d]: %v want %v", l.name, i, g[i], w[i])
				}
			}
		}
	}
}

func equalShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func TestMapDirectDTypesAndErrors(t *testing.T) {
	f2 := func(x, y float64) float64 { return x + y }
	f3 := func(x, y, z float64) float64 { return x + y + z }
	a := arr(t, []float64{1, 2, 3}, 3)
	a32 := a.AsType(ndarray.Float32)
	r, err := Map2(f2, a32, a32)
	if err != nil || r.DType() != ndarray.Float32 {
		t.Errorf("Float32 pair: %v, %v", r, err)
	}
	r, err = Map3(f3, a32, a32, a32)
	if err != nil || r.DType() != ndarray.Float32 {
		t.Errorf("Float32 triple: %v, %v", r, err)
	}
	r, err = Map2(f2, a32, a)
	if err != nil || r.DType() != ndarray.Float64 {
		t.Errorf("mixed pair: %v, %v", r, err)
	}
	// an empty result through the direct loop (empty_test.go has the mapN path)
	empty := arr(t, nil, 0)
	if r, err := Map2(f2, empty, empty); err != nil || len(data(t, r)) != 0 {
		t.Errorf("empty Map2: %v, %v", r, err)
	}
	cplx := a.AsType(ndarray.Complex128)
	if _, err := Map2(f2, a, cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("complex Map2: %v", err)
	}
	if _, err := Map3(f3, a, a, cplx); !errors.Is(err, ndarray.ErrDType) {
		t.Errorf("complex Map3: %v", err)
	}
	two := arr(t, []float64{1, 2}, 2)
	if _, err := Map2(f2, a, two); !errors.Is(err, ErrShape) {
		t.Errorf("shape Map2: %v", err)
	}
	if _, err := Map3(f3, a, a, two); !errors.Is(err, ErrShape) {
		t.Errorf("shape Map3: %v", err)
	}
}
