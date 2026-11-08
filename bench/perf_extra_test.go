package bench

import (
	"encoding/binary"
	"fmt"
	"math"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
	"github.com/go-numerics/special/nd"
	"github.com/go-scikit/refbench"
)

// The rows of the functions whose arguments are not all float64 arrays (the
// integer-order Bessel functions, combinatorics) or that reduce an array (the
// log-sum-exp family): each carries its own Python setup and its own data bytes
// for TestDataParity. The reference is SciPy's function of the same name, a
// ufunc (yn, kn, jn, iv) or a Python wrapper (factorial, comb, perm, logsumexp,
// softmax, log_softmax, jn_zeros), whose per-call overhead is what a user pays.

type xrow struct {
	name, fn, ref, setup string
	data                 []byte // the bytes the Python setup must reproduce; nil: not checked
	mk                   func() func()
}

func f64bytes(v ...[]float64) []byte {
	var b []byte
	for _, s := range v {
		for _, x := range s {
			b = binary.LittleEndian.AppendUint64(b, math.Float64bits(x))
		}
	}
	return b
}

func i64bytes(v []int64) []byte {
	var b []byte
	for _, x := range v {
		b = binary.LittleEndian.AppendUint64(b, uint64(x))
	}
	return b
}

// ptsExpr is the Python expression of pts(lo, hi).
func ptsExpr(lo, hi float64) string {
	return fmt.Sprintf("%v + (%v - %v) * np.mod(np.arange(n, dtype=np.float64) * 0.6180339887498949, 1.0)", lo, hi, lo)
}

func arr(v []float64) *ndarray.Array {
	a, err := ndarray.FromData(v, len(v))
	if err != nil {
		panic(err)
	}
	return a
}

func floatsOf(v []int64) []float64 {
	f := make([]float64, len(v))
	for i, x := range v {
		f[i] = float64(x)
	}
	return f
}

var sinkS []float64

func xrows() []xrow {
	var rs []xrow
	dst := make([]float64, n)

	// Integer-order Bessel functions: order i mod 21, x in (0.1, 50).
	ord := make([]int64, n)
	for i := range ord {
		ord[i] = int64(i % 21)
	}
	ordF := arr(floatsOf(ord))
	xs := pts(0.1, 50)
	xa := arr(xs)
	bessel := fmt.Sprintf("[np.arange(n) %% 21, %s]", ptsExpr(0.1, 50))
	bdata := append(i64bytes(ord), f64bytes(xs)...)
	for _, b := range []struct {
		name, ref string
		f         func(int, float64) float64
		g         func(a, b *ndarray.Array) (*ndarray.Array, error)
	}{
		{"Jn", "sp.jn", special.Jn, nd.Jn},
		{"Yn", "sp.yn", special.Yn, nd.Yn},
		{"In", "sp.iv", special.In, nd.In},
		{"Kn", "sp.kn", special.Kn, nd.Kn},
	} {
		b := b
		rs = append(rs, xrow{name: b.name, fn: b.name, ref: b.ref, setup: bessel, data: bdata,
			mk: func() func() {
				return func() {
					for i, o := range ord {
						dst[i] = b.f(int(o), xs[i])
					}
					sink = dst[0]
				}
			}})
		rs = append(rs, xrow{name: "nd/" + b.name, fn: "nd." + b.name, ref: b.ref, setup: bessel, // the order is a float64 array on the Go side
			mk: func() func() {
				return func() {
					if _, err := b.g(ordF, xa); err != nil {
						panic(err)
					}
				}
			}})
	}

	// Factorials: non-integers in (0.1, 100) take the gamma path, the integers
	// 0..170 the table.
	fx := pts(0.1, 100)
	fi := make([]float64, n)
	for i, v := range fx {
		fi[i] = math.Floor(v * 1.7)
	}
	fxArr, fiArr := arr(fx), arr(fi)
	rs = append(rs,
		xrow{name: "Factorial", fn: "Factorial", ref: "sp.factorial", setup: "[" + ptsExpr(0.1, 100) + "]", data: f64bytes(fx),
			mk: func() func() {
				return func() {
					for i, v := range fx {
						dst[i] = special.Factorial(v)
					}
					sink = dst[0]
				}
			}},
		xrow{name: "nd/Factorial", fn: "nd.Factorial", ref: "sp.factorial", setup: "[" + ptsExpr(0.1, 100) + "]",
			mk: func() func() { return func() { mustND(nd.Factorial(fxArr)) } }},
		xrow{name: "Factorial integers", fn: "Factorial", ref: "sp.factorial", setup: "[np.floor((" + ptsExpr(0.1, 100) + ") * 1.7)]", data: f64bytes(fi),
			mk: func() func() {
				return func() {
					for i, v := range fi {
						dst[i] = special.Factorial(v)
					}
					sink = dst[0]
				}
			}},
		xrow{name: "nd/Factorial integers", fn: "nd.Factorial", ref: "sp.factorial", setup: "[np.floor((" + ptsExpr(0.1, 100) + ") * 1.7)]",
			mk: func() func() { return func() { mustND(nd.Factorial(fiArr)) } }},
	)

	// Double factorial: n = i mod 201 (n!! overflows past 300).
	df := make([]int64, n)
	for i := range df {
		df[i] = int64(i % 201)
	}
	dfArr := arr(floatsOf(df))
	rs = append(rs,
		xrow{name: "Factorial2", fn: "Factorial2", ref: "sp.factorial2", setup: "[np.arange(n) % 201]", data: i64bytes(df),
			mk: func() func() {
				return func() {
					for i, v := range df {
						dst[i] = special.Factorial2(int(v))
					}
					sink = dst[0]
				}
			}},
		xrow{name: "nd/Factorial2", fn: "nd.Factorial2", ref: "sp.factorial2", setup: "[np.arange(n) % 201]",
			mk: func() func() { return func() { mustND(nd.Factorial2(dfArr)) } }},
	)

	// Combinations and permutations: N = 20 + i mod 80, k = 7i mod 21 (k <= N).
	cn, ck := make([]int64, n), make([]int64, n)
	for i := range cn {
		cn[i], ck[i] = int64(i%80+20), int64(i*7%21)
	}
	cnArr, ckArr := arr(floatsOf(cn)), arr(floatsOf(ck))
	comb := "[np.arange(n) % 80 + 20, (np.arange(n) * 7) % 21]"
	cdata := append(i64bytes(cn), i64bytes(ck)...)
	for _, c := range []struct {
		name, ref string
		f         func(n, k int64) float64
		g         func(n, k *ndarray.Array) (*ndarray.Array, error)
	}{
		{"Comb", "sp.comb", special.Comb, nd.Comb},
		{"Perm", "sp.perm", special.Perm, nd.Perm},
	} {
		c := c
		rs = append(rs,
			xrow{name: c.name, fn: c.name, ref: c.ref, setup: comb, data: cdata,
				mk: func() func() {
					return func() {
						for i := range cn {
							dst[i] = c.f(cn[i], ck[i])
						}
						sink = dst[0]
					}
				}},
			xrow{name: "nd/" + c.name, fn: "nd." + c.name, ref: c.ref, setup: comb,
				mk: func() func() { return func() { mustND(c.g(cnArr, ckArr)) } }},
		)
	}

	// Zeros of Jn: SciPy's jn_zeros is a Python function over specfun.
	rs = append(rs, xrow{name: "JnZeros", fn: "JnZeros", ref: "lambda: sp.jn_zeros(2, 20)", setup: "[]",
		mk: func() func() { return func() { sinkS = special.JnZeros(2, 20) } }})

	// Reductions over n points in (-50, 50).
	rx := pts(-50, 50)
	rxArr := arr(rx)
	red := "[" + ptsExpr(-50, 50) + "]"
	rdata := f64bytes(rx)
	rs = append(rs,
		xrow{name: "LogSumExp", fn: "LogSumExp", ref: "sp.logsumexp", setup: red, data: rdata,
			mk: func() func() { return func() { sink = special.LogSumExp(rx) } }},
		xrow{name: "Softmax", fn: "Softmax", ref: "sp.softmax", setup: red, data: rdata,
			mk: func() func() { return func() { sinkS = special.Softmax(dst, rx) } }},
		xrow{name: "LogSoftmax", fn: "LogSoftmax", ref: "sp.log_softmax", setup: red, data: rdata,
			mk: func() func() { return func() { sinkS = special.LogSoftmax(dst, rx) } }},
		xrow{name: "nd/LogSumExp", fn: "nd.LogSumExp", ref: "sp.logsumexp", setup: red,
			mk: func() func() {
				return func() {
					v, err := nd.LogSumExp(rxArr)
					if err != nil {
						panic(err)
					}
					sink = v
				}
			}},
		xrow{name: "nd/Softmax", fn: "nd.Softmax", ref: "sp.softmax", setup: red,
			mk: func() func() { return func() { mustND(nd.Softmax(rxArr)) } }},
		xrow{name: "nd/LogSoftmax", fn: "nd.LogSoftmax", ref: "sp.log_softmax", setup: red,
			mk: func() func() { return func() { mustND(nd.LogSoftmax(rxArr)) } }},
	)
	return rs
}

func mustND(a *ndarray.Array, err error) {
	if err != nil {
		panic(err)
	}
	_ = a
}

// xcases turns the extra rows into cases; ref is nil unless a worker is given.
func xcases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, r := range xrows() {
		body := r.mk()
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(r.setup, r.ref, n)
		}
		out = append(out, refbench.Case{
			Name:      r.name,
			Func:      r.fn,
			Reference: "scipy.special " + r.ref + " (over " + fmt.Sprint(n) + " points)",
			Parity:    refbench.ParityNone,
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					body()
				}
			},
			Ref: ref,
		})
	}
	return out
}
