package nd_test

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special/nd"
)

// parBenchSizes brackets the thresholds of the parallel policy.
var parBenchSizes = []int{1024, 4096, 8192, 16384, 65536}

// parBenchFns holds one cheap, one cheap-vectorised, one medium and two
// expensive functions; the benchmarks run under the GOMAXPROCS the caller
// sets (abbench -g, or go test -cpu).
var parBenchFns = []struct {
	name string
	args int
	lo   []float64
	hi   []float64
	run  func(a []*ndarray.Array) (*ndarray.Array, error)
}{
	{"GammaSign", 1, []float64{-10}, []float64{10}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.GammaSign(a[0]) }},
	{"Expit", 1, []float64{-10}, []float64{10}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.Expit(a[0]) }},
	{"Erf", 1, []float64{-3}, []float64{3}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.Erf(a[0]) }},
	{"Gamma", 1, []float64{0.5}, []float64{20}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.Gamma(a[0]) }},
	{"Binom", 2, []float64{1, 0}, []float64{50, 50}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.Binom(a[0], a[1]) }},
	{"HurwitzZeta", 2, []float64{1.5, 0.5}, []float64{10, 5}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.HurwitzZeta(a[0], a[1]) }},
	{"Beta", 2, []float64{0.5, 0.5}, []float64{30, 30}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.Beta(a[0], a[1]) }},
	{"EllipK", 1, []float64{0}, []float64{0.99}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.EllipK(a[0]) }},
	{"J0", 1, []float64{0}, []float64{50}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.J0(a[0]) }},
	{"BetaInc", 3, []float64{0.5, 0.5, 0.01}, []float64{30, 30, 0.99}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.BetaInc(a[0], a[1], a[2]) }},
	{"BetaIncInv", 3, []float64{0.5, 0.5, 0.01}, []float64{30, 30, 0.99}, func(a []*ndarray.Array) (*ndarray.Array, error) { return nd.BetaIncInv(a[0], a[1], a[2]) }},
}

func BenchmarkPar(b *testing.B) {
	for _, fn := range parBenchFns {
		for _, n := range parBenchSizes {
			rng := rand.New(rand.NewSource(1))
			args := make([]*ndarray.Array, fn.args)
			for k := range args {
				d := make([]float64, n)
				for i := range d {
					d[i] = math.FMA(fn.hi[k]-fn.lo[k], rng.Float64(), fn.lo[k])
				}
				args[k], _ = ndarray.FromData(d, n)
			}
			b.Run(fmt.Sprintf("%s/n=%d", fn.name, n), func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, err := fn.run(args); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}
