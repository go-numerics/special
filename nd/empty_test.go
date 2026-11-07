package nd

import (
	"math/rand"
	"testing"

	"github.com/go-ndarray/ndarray"
)

// A broadcast with a zero extent has no elements: (0,3) with (1,3) is the
// non-direct path (the second operand is neither the result's shape nor a single
// element) and used to panic with a remainder by zero in mapN.
func TestMapEmptyBroadcast(t *testing.T) {
	f2 := func(x, y float64) float64 { return x + y }
	f3 := func(x, y, z float64) float64 { return x + y + z }
	e := arr(t, nil, 0, 3)
	row := arr(t, []float64{1, 2, 3}, 1, 3)
	col := arr(t, []float64{1}, 1, 1)
	for name, call := range map[string]func() ([]int, int, error){
		"(0,3)x(1,3)": func() ([]int, int, error) { r, err := Map2(f2, e, row); return shapeLen(t, r, err) },
		"(1,3)x(0,3)": func() ([]int, int, error) { r, err := Map2(f2, row, e); return shapeLen(t, r, err) },
		"(0,3)x(1,1)": func() ([]int, int, error) { r, err := Map2(f2, e, col); return shapeLen(t, r, err) },
		"(0,1)x(1,3)": func() ([]int, int, error) { r, err := Map2(f2, arr(t, nil, 0, 1), row); return shapeLen(t, r, err) },
		"triple":      func() ([]int, int, error) { r, err := Map3(f3, e, row, col); return shapeLen(t, r, err) },
	} {
		sh, n, err := call()
		if err != nil || n != 0 || len(sh) != 2 || sh[0] != 0 {
			t.Errorf("%s: shape %v, %d elements, err %v", name, sh, n, err)
		}
	}
}

func shapeLen(t *testing.T, r *ndarray.Array, err error) ([]int, int, error) {
	t.Helper()
	if err != nil {
		return nil, -1, err
	}
	return r.Shape(), len(data(t, r)), nil
}

// Random shapes with zero extents (and ones, for broadcasting): Map2 and Map3
// never panic, and either fail with ErrShape or return the broadcast shape with
// as many elements as it holds.
func TestMapRandomShapesWithZeroExtents(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	f2 := func(x, y float64) float64 { return x*2 + y }
	f3 := func(x, y, z float64) float64 { return x*2 + y - z }
	mk := func() ([]int, int) {
		sh := make([]int, 1+rng.Intn(3))
		n := 1
		for i := range sh {
			sh[i] = []int{0, 1, 1, 2, 3}[rng.Intn(5)]
			n *= sh[i]
		}
		return sh, n
	}
	for it := 0; it < 3000; it++ {
		s1, n1 := mk()
		s2, n2 := mk()
		s3, n3 := mk()
		a := arr(t, make([]float64, n1), s1...)
		b := arr(t, make([]float64, n2), s2...)
		c := arr(t, make([]float64, n3), s3...)
		want, _, werr := broadcast(s1, s2, s3)
		r2, err2 := Map2(f2, a, b)
		r3, err3 := Map3(f3, a, b, c)
		if (err3 != nil) != (werr != nil) {
			t.Fatalf("%v %v %v: Map3 err %v, broadcast err %v", s1, s2, s3, err3, werr)
		}
		if err3 == nil && (!equalShape(r3.Shape(), want) || len(data(t, r3)) != size(want)) {
			t.Fatalf("%v %v %v: Map3 shape %v want %v", s1, s2, s3, r3.Shape(), want)
		}
		want2, _, werr2 := broadcast(s1, s2)
		if (err2 != nil) != (werr2 != nil) {
			t.Fatalf("%v %v: Map2 err %v, broadcast err %v", s1, s2, err2, werr2)
		}
		if err2 == nil && (!equalShape(r2.Shape(), want2) || len(data(t, r2)) != size(want2)) {
			t.Fatalf("%v %v: Map2 shape %v want %v", s1, s2, r2.Shape(), want2)
		}
	}
}
