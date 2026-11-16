package special

import (
	"math"
	"math/rand/v2"
	"os"
	"strconv"
	"testing"

	"github.com/go-numerics/special/dd"
)

// The oracle of the log-sum-exp family: the double-double evaluation Softmax,
// LogSoftmax and LogSumExp used before softmax_fast.go (each x_i - m exact, each
// exponential to about 2^-54 through dd.ExpMant, the sum, the quotient and
// log1p in double-double), whose results are within about 2^-98 relative.

// smRefExp returns exp(x_i - m) in double-double for finite m = max x and the
// index of the first maximum.
func smRefExp(x []float64) (es []dd.DD, m float64, im int) {
	m, im = math.Inf(-1), -1
	for i, v := range x {
		if v > m {
			m, im = v, i
		}
	}
	es = make([]dd.DD, len(x))
	for i, v := range x {
		e, k := dd.ExpMant(sumDD(v, -m))
		es[i] = dd.Ldexp(e, k)
	}
	return es, m, im
}

func softmaxRef(x []float64) []dd.DD {
	es, _, _ := smRefExp(x)
	var s dd.DD
	for _, e := range es {
		s = dd.Add(s, e)
	}
	out := make([]dd.DD, len(x))
	for i, e := range es {
		out[i] = dd.Div(e, s)
	}
	return out
}

func logSumExpRef(x []float64) dd.DD {
	es, m, im := smRefExp(x)
	var s dd.DD
	for i, e := range es {
		if i != im {
			s = dd.Add(s, e)
		}
	}
	return dd.AddF(log1pDD(s), m)
}

func logSoftmaxRef(x []float64) []dd.DD {
	lse := logSumExpRef(x)
	out := make([]dd.DD, len(x))
	for i, v := range x {
		out[i] = dd.Sub(dd.F(v), lse)
	}
	return out
}

// inU is x in units of u = 2^-53 (an exact scaling by a power of two).
func inU(x float64) float64 { return math.Ldexp(x, 53) }

// relErr returns |got - want| / |want| for a double-double want, and the
// absolute error.
func relErr(got float64, want dd.DD) (rel, abs float64) {
	abs = math.Abs(dd.Sub(dd.F(got), want).Hi)
	w := math.Abs(want.Hi)
	if w == 0 {
		return 0, abs
	}
	return abs / w, abs
}

// smShapes are the inputs of the sweeps: each draws n values of one shape.
var smShapes = []struct {
	name string
	gen  func(r *rand.Rand) float64
}{
	{"moderate", func(r *rand.Rand) float64 { return float64(r.NormFloat64() * 3) }},
	{"wide", func(r *rand.Rand) float64 { return float64(-700 * r.Float64()) }},
	{"refbench", func(r *rand.Rand) float64 { return -50 + float64(100*r.Float64()) }},
	{"huge", func(r *rand.Rand) float64 { return 1e6 + r.NormFloat64() }},
	{"hugeneg", func(r *rand.Rand) float64 { return float64(-1e15 * (1 + r.Float64())) }},
	{"tiny", func(r *rand.Rand) float64 { return float64(r.NormFloat64() * 1e-9) }},
	{"dominant", func(r *rand.Rand) float64 {
		if r.IntN(50) == 0 {
			return 40 + r.Float64()
		}
		return r.NormFloat64()
	}},
	{"masked", func(r *rand.Rand) float64 {
		if r.IntN(3) == 0 {
			return math.Inf(-1)
		}
		return float64(5 * r.NormFloat64())
	}},
	{"ties", func(r *rand.Rand) float64 { return float64(r.IntN(3)) - 1 }},
}

// smSizes include the chunk edges (64, 512) of smSum.
var smSizes = []int{1, 2, 3, 4, 5, 7, 8, 63, 64, 65, 100, 511, 512, 513, 1000, 1024, 1025, 5000}

func smReps() int {
	if s := os.Getenv("SPECIAL_SOFTMAX_REPS"); s != "" {
		if n, err := strconv.Atoi(s); err == nil {
			return n
		}
	}
	return 8
}

// TestSoftmaxBound: the stated relative bound of Softmax against the
// double-double oracle on every shape and size, with the absolute allowance of
// the subnormal results. The largest observed error is logged.
func TestSoftmaxBound(t *testing.T) {
	r := rand.New(rand.NewPCG(1, 2))
	worst := 0.0
	for _, sh := range smShapes {
		for _, n := range smSizes {
			for rep := 0; rep < smReps(); rep++ {
				x := make([]float64, n)
				for i := range x {
					x[i] = sh.gen(r)
				}
				got := Softmax(nil, x)
				var ref []dd.DD
				if math.IsInf(floatMax(x), -1) {
					continue
				}
				ref = softmaxRef(x)
				for i := range x {
					rel, abs := relErr(got[i], ref[i])
					if ref[i].Hi >= 0x1p-1022 {
						worst = math.Max(worst, rel)
						if rel > softmaxErr {
							t.Fatalf("%s n=%d: Softmax[%d] = %v, want %v: relative error %g > %g (x_i = %v)", sh.name, n, i, got[i], ref[i].Hi, rel, softmaxErr, x[i])
						}
					} else if abs > float64(softmaxErr*ref[i].Hi)+0x1p-1073 {
						t.Fatalf("%s n=%d: Softmax[%d] = %v, want %v: absolute error %g (x_i = %v)", sh.name, n, i, got[i], ref[i].Hi, abs, x[i])
					}
				}
			}
		}
	}
	t.Logf("largest relative error %.3g = %.2f u (bound %.0f u)", worst, inU(worst), inU(softmaxErr))
}

func floatMax(x []float64) float64 {
	m := math.Inf(-1)
	for _, v := range x {
		m = math.Max(m, v)
	}
	return m
}

// TestLogSoftmaxBound: as TestSoftmaxBound for LogSoftmax.
func TestLogSoftmaxBound(t *testing.T) {
	r := rand.New(rand.NewPCG(3, 4))
	worst := 0.0
	for _, sh := range smShapes {
		for _, n := range smSizes {
			for rep := 0; rep < smReps(); rep++ {
				x := make([]float64, n)
				for i := range x {
					x[i] = sh.gen(r)
				}
				if math.IsInf(floatMax(x), -1) {
					continue
				}
				got := LogSoftmax(nil, x)
				ref := logSoftmaxRef(x)
				for i := range x {
					if math.IsInf(x[i], -1) {
						if !math.IsInf(got[i], -1) {
							t.Fatalf("%s n=%d: LogSoftmax of -Inf = %v", sh.name, n, got[i])
						}
						continue
					}
					rel, _ := relErr(got[i], ref[i])
					worst = math.Max(worst, rel)
					if rel > logSoftmaxErr {
						t.Fatalf("%s n=%d: LogSoftmax[%d] = %v, want %v: relative error %g > %g (x_i = %v)", sh.name, n, i, got[i], ref[i].Hi, rel, logSoftmaxErr, x[i])
					}
				}
			}
		}
	}
	t.Logf("largest relative error %.3g = %.2f u (bound %.0f u)", worst, inU(worst), inU(logSoftmaxErr))
}

// TestLogSumExpBound: as TestSoftmaxBound for LogSumExp; the inputs the guard
// sends to the double-double evaluation are held to 1 ulp instead.
func TestLogSumExpBound(t *testing.T) {
	r := rand.New(rand.NewPCG(5, 6))
	worst, fast, slow := 0.0, 0, 0
	for _, sh := range smShapes {
		for _, n := range smSizes {
			for rep := 0; rep < smReps(); rep++ {
				x := make([]float64, n)
				for i := range x {
					x[i] = sh.gen(r)
				}
				if math.IsInf(floatMax(x), -1) {
					continue
				}
				got := LogSumExp(x)
				ref := logSumExpRef(x)
				rel, _ := relErr(got, ref)
				if _, ok := logSumExpFast(x, floatMax(x)); ok {
					fast++
					worst = math.Max(worst, rel)
					if rel > logSumExpErr {
						t.Fatalf("%s n=%d: LogSumExp = %v, want %v: relative error %g > %g", sh.name, n, got, ref.Hi, rel, logSumExpErr)
					}
				} else {
					// the guarded path is judged by the math/big oracle, in
					// absolute terms (TestLogSumExpCancellationBig has the
					// derivation); the double-double oracle shares log1pDD
					// with it and could never fail it
					slow++
					if n <= 100 {
						o := lseOracle(x)
						l, _ := o.l.Float64()
						lf, _ := o.lse.Float64()
						bound := math.Ldexp(l, -84) + math.Ldexp(math.Abs(lf), -53)
						if abs, _ := lseErr(got, o.lse); abs > bound {
							t.Fatalf("%s n=%d: guarded LogSumExp = %v, want %v: absolute error %g > %g", sh.name, n, got, lf, abs, bound)
						}
					}
				}
			}
		}
	}
	t.Logf("fast path %d inputs (largest relative error %.3g = %.2f u, bound %.0f u), double-double path %d", fast, worst, inU(worst), inU(logSumExpErr), slow)
}
