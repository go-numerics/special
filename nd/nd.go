// Package nd applies the special functions of package special element-wise
// to go-ndarray arrays (github.com/go-ndarray/ndarray).
//
// Every real dtype is accepted. A Float32 input gives a Float32 result
// (computed in float64 and rounded once); Float64, integer and Bool inputs
// give Float64. Complex inputs are rejected with ndarray.ErrDType. Functions
// of several arguments broadcast them against each other as NumPy does. A call
// whose total work is worth it (about 40 us, estimated from the element count
// and a per-function cost class) is split across up to GOMAXPROCS goroutines
// that claim blocks of elements from a shared counter; a cheap function needs
// tens of thousands of elements, an expensive one a few dozen.
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
	"sync/atomic"

	"github.com/go-ndarray/ndarray"
)

// ErrShape is returned (wrapped) when arguments cannot be broadcast together.
var ErrShape = errors.New("nd: shapes cannot be broadcast")

// cost is the estimated time of one element, in nanoseconds, of the function
// a call applies. The parallel policy needs it because the break-even point
// is a quantity of work, not a count of elements: 8192 elements of GammaSign
// (3 ns each) are 26 us and gain nothing from 16 goroutines, while 64
// elements of BetaIncInv (2 us each) are already worth four.
//
// The class of each function of this package is taken from its measured cost
// (ns per element, docs/perf.json, one core): the table costs below lists
// them and TestCostsMatchMeasurement fails when one is more than a class away
// from the measurement, so the classes cannot drift. An estimate that is off
// by a factor of three costs little: the thresholds scale linearly with it and
// the policy only needs the order of magnitude.
type cost int

const (
	costTiny   cost = 3    // under 4 ns: GammaSign, Factorial2 (serial at 8192 elements, as measured)
	costCheap  cost = 6    // 4 to 8 ns: Expit, Ellip*, the scaled Bessel functions
	costLight  cost = 12   // 7 to 22 ns: Gamma, Erf, J0, I0
	costMedium cost = 40   // 22 to 70 ns: Beta, Kn
	costDear   cost = 120  // 70 to 250 ns: Binom, HurwitzZeta, GammaP, Jn
	costHeavy  cost = 500  // 250 ns to 0.9 us: BetaInc, Jv, Kv, GammaPInv
	costHuge   cost = 1500 // over 0.9 us: BetaIncInv
	// costUser is the class of a function the caller supplies to Map1, Map2
	// or Map3: its cost is unknown, so the threshold is the one these
	// functions have always had (8000 elements, about main's 8192).
	costUser cost = 5
)

// costClasses lists the classes in increasing order.
var costClasses = []cost{costTiny, costCheap, costLight, costMedium, costDear, costHeavy, costHuge}

// costs is the class of each function of this package, by name.
var costs = map[string]cost{
	"Beta": costMedium, "BetaInc": costHeavy, "BetaIncC": costHeavy,
	"BetaIncCInv": costHuge, "BetaIncInv": costHuge, "Binom": costDear,
	"Comb": costLight, "Digamma": costLight, "EllipE": costCheap,
	"EllipK": costCheap, "EllipKm1": costCheap, "Erf": costLight,
	"ErfInv": costCheap, "Erfc": costLight, "ErfcInv": costCheap,
	"Erfcx": costCheap, "Factorial": costLight, "Factorial2": costTiny,
	"Gamma": costLight, "GammaP": costDear, "GammaPInv": costHeavy,
	"GammaQ": costDear, "GammaQInv": costHeavy, "GammaSign": costTiny,
	"HurwitzZeta": costDear, "I0": costLight, "I0e": costCheap,
	"I1": costLight, "I1e": costCheap, "In": costDear, "Iv": costHeavy,
	"Ive": costHeavy, "J0": costLight, "J1": costLight, "Jn": costDear,
	"Jv": costHeavy, "K0": costLight, "K0e": costCheap, "K1": costLight,
	"K1e": costCheap, "Kn": costMedium, "Kv": costHeavy, "Kve": costHeavy,
	"LogBeta": costLight, "LogExpit": costLight, "LogFactorial": costLight,
	"LogGamma": costCheap, "Log1pExp": costLight, "LogNdtr": costCheap,
	"Logit": costCheap, "Expit": costCheap, "Ndtr": costLight,
	"Ndtri": costCheap, "Perm": costLight, "Polygamma": costLight,
	"Xlog1py": costLight, "Xlogy": costCheap, "Y0": costLight,
	"Y1": costLight, "Yn": costDear, "Yv": costHeavy, "Zeta": costLight,
}

// Work quanta of the parallel policy, in nanoseconds (see parallel).
const (
	// parallelWork is the total work below which a call stays on the
	// calling goroutine.
	parallelWork = 40_000
	// workerWork is the work each worker must be worth: a call uses at most
	// total/workerWork workers.
	workerWork = 8_000
	// blockWork is the smallest block a worker claims at a time.
	blockWork = 8_000
	// blocksPerWorker is the number of blocks per worker the call is cut
	// into, so that a slow core (an efficiency core) claims fewer of them.
	blocksPerWorker = 8
)

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

// parallel runs body over [0, n) for elements costing c each, on at most
// GOMAXPROCS goroutines.
func parallel(n int, c cost, body func(lo, hi int)) {
	parallelOn(runtime.GOMAXPROCS(0), n, c, body)
}

// parallelOn is parallel with the processor count given. Below parallelWork
// of total work, or with one processor, body runs once over [0, n) on the
// calling goroutine. Above, there are w' = min(w, work/workerWork) workers (the
// calling goroutine is one of them), and the range is cut into blocks of
// at least blockWork that the workers claim from an atomic counter: a worker
// on a slow core, or one that met expensive elements (BetaInc takes from 1 to
// 38 iterations), simply claims fewer blocks, where an equal split would make
// the whole call wait for it. Every element is computed by one call of the
// same function whatever the split, so results do not depend on it.
func parallelOn(w, n int, c cost, body func(lo, hi int)) {
	per := int(c)
	if w <= 1 || n < parallelWork/per {
		body(0, n)
		return
	}
	// n/(workerWork/per) rather than n*per/workerWork: no overflow on 32 bits.
	// Since parallelWork is twice workerWork, n >= parallelWork/per gives at
	// least 2 workers; the max keeps one worker correct all the same.
	floor := blockWork / per
	if c == costUser {
		// A user function's cost is unknown, so it keeps every processor and
		// an even cut, as it always did above the threshold.
		floor = 1
	} else {
		w = max(1, min(w, n/max(1, workerWork/per)))
	}
	block := max(floor, (n+w*blocksPerWorker-1)/(w*blocksPerWorker))
	var next atomic.Int64
	// A panic of body (a function the caller supplied) is recovered in the
	// worker it happens in, the other workers stop claiming blocks, and the
	// first value is raised again in the calling goroutine once all are done:
	// the caller can recover it exactly as it can on the serial path.
	var (
		mu       sync.Mutex
		panicked bool
		value    any
	)
	run := func() {
		defer func() {
			if r := recover(); r != nil {
				mu.Lock()
				if !panicked {
					panicked, value = true, r
				}
				mu.Unlock()
				next.Store(int64(n) + int64(block)) // nothing left to claim
			}
		}()
		// start is compared as int64: the counter runs past n by up to one
		// block per worker, which could wrap an int on 32 bits
		for start := next.Add(int64(block)) - int64(block); start < int64(n); start = next.Add(int64(block)) - int64(block) {
			lo := int(start)
			body(lo, min(lo+block, n))
		}
	}
	var wg sync.WaitGroup
	for range w - 1 {
		wg.Go(run)
	}
	run()
	wg.Wait()
	if panicked {
		panic(value)
	}
}

// Map1 applies f element-wise to a. From 8000 elements on, f is called from
// several goroutines at once, so it must be safe for that; a panic of f is
// raised again in the calling goroutine, as on a serial loop.
func Map1(f func(float64) float64, a *ndarray.Array) (*ndarray.Array, error) {
	return map1(costUser, f, a)
}

// map1 is Map1 for a function of the given cost.
func map1(c cost, f func(float64) float64, a *ndarray.Array) (*ndarray.Array, error) {
	x, f32, err := operand(a)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(x))
	parallel(len(x), c, func(lo, hi int) {
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
func mapN(c cost, f func(v []float64) float64, args ...*ndarray.Array) (*ndarray.Array, error) {
	data := make([][]float64, len(args))
	shapes := make([][]int, len(args))
	f32 := false
	for i, a := range args {
		d, s32, _ := operand(a) // Map2 and Map3 have validated the operands (direct)
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
	shape, strides, _ := broadcast(shapes...) // validated by direct
	n := size(shape)
	if n == 0 {
		// a zero extent: nothing to compute, and the index walk below would
		// take a remainder by that extent
		return result(nil, shape, f32), nil
	}
	out := make([]float64, n)
	parallel(n, c, func(lo, hi int) {
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

// direct reports whether every operand is either of the broadcast shape or a
// single element, the two cases that need no index arithmetic: it returns the
// operands' data, their strides (1, or 0 for a single element) and the result
// shape. ok is false for any other broadcast (mapN handles it); an error is a
// complex operand or shapes that do not broadcast.
func direct(args ...*ndarray.Array) (data [][]float64, st []int, shape []int, f32, ok bool, err error) {
	data = make([][]float64, len(args))
	shapes := make([][]int, len(args))
	f32 = true
	for i, a := range args {
		d, _, err := operand(a)
		if err != nil {
			return nil, nil, nil, false, false, err
		}
		data[i], shapes[i] = d, a.Shape()
		f32 = f32 && a.DType() == ndarray.Float32
	}
	shape, _, err = broadcast(shapes...)
	if err != nil {
		return nil, nil, nil, false, false, err
	}
	n := size(shape)
	st = make([]int, len(args))
	for i, d := range data {
		switch {
		case len(d) == n && len(shapes[i]) == len(shape):
			st[i] = 1
		case len(d) == 1:
			st[i] = 0
		default:
			return nil, nil, nil, false, false, nil
		}
	}
	return data, st, shape, f32, true, nil
}

// Map2 applies f element-wise to the broadcast pair (a, b). Operands of one
// shape, or one of them a single element, run one loop with no index
// arithmetic and no call through a second closure per element. As for Map1,
// f may be called concurrently and a panic of f reaches the caller.
func Map2(f func(x, y float64) float64, a, b *ndarray.Array) (*ndarray.Array, error) {
	return map2(costUser, f, a, b)
}

// map2 is Map2 for a function of the given cost.
func map2(c cost, f func(x, y float64) float64, a, b *ndarray.Array) (*ndarray.Array, error) {
	d, st, shape, f32, ok, err := direct(a, b)
	if err != nil {
		return nil, err
	}
	if !ok {
		return mapN(c, func(v []float64) float64 { return f(v[0], v[1]) }, a, b)
	}
	n := size(shape)
	x, y, sx, sy := d[0], d[1], st[0], st[1]
	out := make([]float64, n)
	parallel(n, c, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = f(x[i*sx], y[i*sy])
		}
	})
	return result(out, shape, f32), nil
}

// Map3 applies f element-wise to the broadcast triple (a, b, c), with the
// same direct loop as Map2 where the operands allow it. As for Map1, f may be
// called concurrently and a panic of f reaches the caller.
func Map3(f func(x, y, z float64) float64, a, b, c *ndarray.Array) (*ndarray.Array, error) {
	return map3(costUser, f, a, b, c)
}

// map3 is Map3 for a function of the given cost.
func map3(k cost, f func(x, y, z float64) float64, a, b, c *ndarray.Array) (*ndarray.Array, error) {
	d, st, shape, f32, ok, err := direct(a, b, c)
	if err != nil {
		return nil, err
	}
	if !ok {
		return mapN(k, func(v []float64) float64 { return f(v[0], v[1], v[2]) }, a, b, c)
	}
	n := size(shape)
	x, y, z, sx, sy, sz := d[0], d[1], d[2], st[0], st[1], st[2]
	out := make([]float64, n)
	parallel(n, k, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			out[i] = f(x[i*sx], y[i*sy], z[i*sz])
		}
	})
	return result(out, shape, f32), nil
}

// size is the element count of a shape.
func size(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}

// apply1 is Map1 for the functions with an array variant in package
// special (special.ErfTo etc., generated by tools/geninline with the
// kernel expanded in the loop): each goroutine runs that one loop over its
// chunk, with no call per element.
func apply1(c cost, loop func(dst, x []float64), a *ndarray.Array) (*ndarray.Array, error) {
	x, f32, err := operand(a)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(x))
	parallel(len(x), c, func(lo, hi int) { loop(out[lo:hi], x[lo:hi]) })
	return result(out, a.Shape(), f32), nil
}

// apply2 is Map2 for the functions with an array variant: operands
// of the same shape, or one of them a single element, run the loop
// directly; any other broadcast goes through mapN with f.
func apply2(c cost, loop func(dst, a, b []float64, sa, sb int), f func(x, y float64) float64, a, b *ndarray.Array) (*ndarray.Array, error) {
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
	case len(x) == n && len(y) == n && len(a.Shape()) == len(b.Shape()):
		// same rank and both of the broadcast size: the same shape
	case len(x) == 1 && len(y) == n:
		sa = 0
	case len(y) == 1 && len(x) == n:
		sb = 0
	default:
		return map2(c, f, a, b)
	}
	out := make([]float64, n)
	parallel(n, c, func(lo, hi int) {
		loop(out[lo:hi], x[lo*sa:], y[lo*sb:], sa, sb)
	})
	return result(out, shape, xf32 && yf32 && a.DType() == ndarray.Float32 && b.DType() == ndarray.Float32), nil
}
