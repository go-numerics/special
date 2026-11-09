package special

import (
	"math"
	"testing"
)

// The refbench points of the Gamma and LogGamma rows: lo + (hi-lo) frac(i phi).
func gapsGolden(lo, hi float64) []float64 {
	x := make([]float64, gapsN)
	for i := range x {
		_, f := math.Modf(float64(i) * 0.6180339887498949)
		x[i] = lo + float64((hi-lo)*f)
	}
	return x
}

func BenchmarkGammaRef(b *testing.B) {
	g, l := gapsGolden(0.1, 170), gapsGolden(0.1, 1000)
	b.Run("Gamma", func(b *testing.B) { gapsSink(b, func(i int) float64 { return Gamma(g[i]) }) })
	b.Run("LogGamma", func(b *testing.B) { gapsSink(b, func(i int) float64 { return LogGamma(l[i]) }) })
	dst := make([]float64, gapsN)
	b.Run("GammaTo", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			GammaTo(dst, g)
		}
	})
	b.Run("LogGammaTo", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			LogGammaTo(dst, l)
		}
	})
}

// l1Like is the loop of the refbench scalar rows (bench/perf_test.go l1):
// one indirect call per element, results stored in out.
func l1Like(f func(float64) float64) func(x, out []float64) {
	return func(x, out []float64) {
		for i := range out {
			out[i] = f(x[i])
		}
	}
}

var gapsGammaFn = Gamma

func BenchmarkGammaHarness(b *testing.B) {
	g := gapsGolden(0.1, 170)
	out := make([]float64, gapsN)
	b.Run("direct", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			for i := range out {
				out[i] = Gamma(g[i])
			}
		}
	})
	run := l1Like(gapsGammaFn)
	b.Run("indirect", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			run(g, out)
		}
	})
}
