// Package nd applies the special functions of package special element-wise
// to go-ndarray arrays (github.com/go-ndarray/ndarray).
//
// Every real dtype is accepted. A Float32 input gives a Float32 result
// (computed in float64 and rounded once); Float64, integer and Bool inputs
// give Float64. Complex inputs are rejected with ndarray.ErrDType. Functions
// of several arguments broadcast them against each other as NumPy does. Large
// arrays are split across GOMAXPROCS goroutines.
//
// Each function documents nothing beyond its scalar counterpart in package
// special, whose accuracy and conventions it inherits exactly: the element
// results are bit-identical to calling the scalar function.
package nd

import (
	"errors"
	"fmt"
	"runtime"
	"sync"

	"github.com/go-ndarray/ndarray"
)

// ErrShape is returned (wrapped) when arguments cannot be broadcast together.
var ErrShape = errors.New("nd: shapes cannot be broadcast")

// parallelMin is the element count from which work is split across
// goroutines; below it the goroutine start-up costs more than it saves.
const parallelMin = 1 << 13

// operand returns a's elements as float64 in row-major order and whether the
// result should be Float32.
func operand(a *ndarray.Array) ([]float64, bool, error) {
	switch dt := a.DType(); {
	case dt.IsComplex():
		return nil, false, fmt.Errorf("%w: special functions take real arrays, got %v", ndarray.ErrDType, dt)
	case dt == ndarray.Float64:
		d, _ := ndarray.Data[float64](a)
		return d, false, nil
	case dt == ndarray.Float32:
		d, _ := ndarray.Data[float64](a.AsType(ndarray.Float64))
		return d, true, nil
	default:
		d, _ := ndarray.Data[float64](a.AsType(ndarray.Float64))
		return d, false, nil
	}
}

// result wraps out with the given shape, as Float32 when f32 is set.
func result(out []float64, shape []int, f32 bool) *ndarray.Array {
	r, _ := ndarray.FromData(out, shape...)
	if f32 {
		return r.AsType(ndarray.Float32)
	}
	return r
}

// parallel runs body over [0, n) in chunks.
func parallel(n int, body func(lo, hi int)) {
	w := runtime.GOMAXPROCS(0)
	if n < parallelMin || w == 1 {
		body(0, n)
		return
	}
	chunk := (n + w - 1) / w
	var wg sync.WaitGroup
	for lo := 0; lo < n; lo += chunk {
		hi := min(lo+chunk, n)
		wg.Add(1)
		go func() {
			defer wg.Done()
			body(lo, hi)
		}()
	}
	wg.Wait()
}

// Map1 applies f element-wise to a.
func Map1(f func(float64) float64, a *ndarray.Array) (*ndarray.Array, error) {
	x, f32, err := operand(a)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(x))
	parallel(len(x), func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = f(x[i])
		}
	})
	return result(out, a.Shape(), f32), nil
}

// broadcast computes the broadcast shape of the operands and, for each, the
// row-major strides (0 along broadcast axes) into its own data.
func broadcast(shapes ...[]int) ([]int, [][]int, error) {
	nd := 0
	for _, s := range shapes {
		nd = max(nd, len(s))
	}
	out := make([]int, nd)
	for i := range out {
		out[i] = 1
	}
	for _, s := range shapes {
		off := nd - len(s)
		for i, d := range s {
			switch o := out[off+i]; {
			case o == d || d == 1:
			case o == 1:
				out[off+i] = d
			default:
				return nil, nil, fmt.Errorf("%w: %v", ErrShape, shapes)
			}
		}
	}
	strides := make([][]int, len(shapes))
	for k, s := range shapes {
		st := make([]int, nd)
		off := nd - len(s)
		acc := 1
		for i := len(s) - 1; i >= 0; i-- {
			if s[i] != 1 {
				st[off+i] = acc
			}
			acc *= s[i]
		}
		strides[k] = st
	}
	return out, strides, nil
}

// mapN applies f to the broadcast operands.
func mapN(f func(v []float64) float64, args ...*ndarray.Array) (*ndarray.Array, error) {
	data := make([][]float64, len(args))
	shapes := make([][]int, len(args))
	f32 := false
	for i, a := range args {
		d, s32, err := operand(a)
		if err != nil {
			return nil, err
		}
		data[i], shapes[i] = d, a.Shape()
		f32 = f32 || s32
	}
	// The result is Float32 only if every floating operand is Float32 and
	// no operand forces Float64 (an integer, bool or Float64 array).
	for _, a := range args {
		if a.DType() != ndarray.Float32 {
			f32 = false
		}
	}
	shape, strides, err := broadcast(shapes...)
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	out := make([]float64, n)
	parallel(n, func(lo, hi int) {
		v := make([]float64, len(args))
		idx := make([]int, len(shape))
		// position idx at lo
		rem := lo
		for i := len(shape) - 1; i >= 0; i-- {
			idx[i] = rem % shape[i]
			rem /= shape[i]
		}
		for p := lo; p < hi; p++ {
			for k := range args {
				off := 0
				for i, c := range idx {
					off += c * strides[k][i]
				}
				v[k] = data[k][off]
			}
			out[p] = f(v)
			for i := len(shape) - 1; i >= 0; i-- {
				idx[i]++
				if idx[i] < shape[i] {
					break
				}
				idx[i] = 0
			}
		}
	})
	return result(out, shape, f32), nil
}

// Map2 applies f element-wise to the broadcast pair (a, b).
func Map2(f func(x, y float64) float64, a, b *ndarray.Array) (*ndarray.Array, error) {
	return mapN(func(v []float64) float64 { return f(v[0], v[1]) }, a, b)
}

// Map3 applies f element-wise to the broadcast triple (a, b, c).
func Map3(f func(x, y, z float64) float64, a, b, c *ndarray.Array) (*ndarray.Array, error) {
	return mapN(func(v []float64) float64 { return f(v[0], v[1], v[2]) }, a, b, c)
}

// apply1 is Map1 for the functions with an array variant in package
// special (special.ErfTo etc., generated by tools/geninline with the
// kernel expanded in the loop): each goroutine runs that one loop over its
// chunk, with no call per element.
func apply1(loop func(dst, x []float64), a *ndarray.Array) (*ndarray.Array, error) {
	x, f32, err := operand(a)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(x))
	parallel(len(x), func(lo, hi int) { loop(out[lo:hi], x[lo:hi]) })
	return result(out, a.Shape(), f32), nil
}

// apply2 is Map2 for the functions with an array variant: operands
// of the same shape, or one of them a single element, run the loop
// directly; any other broadcast goes through mapN with f.
func apply2(loop func(dst, a, b []float64, sa, sb int), f func(x, y float64) float64, a, b *ndarray.Array) (*ndarray.Array, error) {
	x, xf32, err := operand(a)
	if err != nil {
		return nil, err
	}
	y, yf32, err := operand(b)
	if err != nil {
		return nil, err
	}
	shape, _, err := broadcast(a.Shape(), b.Shape())
	if err != nil {
		return nil, err
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	sa, sb := 1, 1
	switch {
	case len(x) == n && len(y) == n && sameShape(a.Shape(), b.Shape()):
	case len(x) == 1 && len(y) == n:
		sa = 0
	case len(y) == 1 && len(x) == n:
		sb = 0
	default:
		return Map2(f, a, b)
	}
	out := make([]float64, n)
	parallel(n, func(lo, hi int) {
		loop(out[lo:hi], x[lo*sa:], y[lo*sb:], sa, sb)
	})
	return result(out, shape, xf32 && yf32 && a.DType() == ndarray.Float32 && b.DType() == ndarray.Float32), nil
}

func sameShape(p, q []int) bool {
	if len(p) != len(q) {
		return false
	}
	for i := range p {
		if p[i] != q[i] {
			return false
		}
	}
	return true
}
