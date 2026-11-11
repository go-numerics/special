package special

import (
	"math"
	"testing"
)

// BenchmarkTail times the tail functions of special#33 on the refbench inputs
// (bench/perf_test.go: lo + (hi-lo)·frac(i·gold), 10000 points), as a scalar
// loop and as the array kernel. ns/op is per 10000 points.
// BenchmarkTail2 times the two-argument rows on the refbench inputs, which
// give both arguments the same sequence (bench/perf_test.go setup).
func BenchmarkTail2(b *testing.B) {
	x := make([]float64, 10000)
	for i := range x {
		x[i] = 0.1 + float64(999.9*math.Mod(float64(i)*0.6180339887498949, 1))
	}
	dst := make([]float64, 10000)
	b.Run("LogBeta", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			for i, v := range x {
				dst[i] = LogBeta(v, v)
			}
		}
	})
	b.Run("LogBetaTo", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			LogBetaTo(dst, x, x, 1, 1)
		}
	})
}

func BenchmarkTail(b *testing.B) {
	pts := func(lo, hi float64) []float64 {
		x := make([]float64, 10000)
		for i := range x {
			x[i] = lo + float64((hi-lo)*math.Mod(float64(i)*0.6180339887498949, 1))
		}
		return x
	}
	rows := []struct {
		name   string
		lo, hi float64
		f      func(float64) float64
		to     func(dst, x []float64)
	}{
		{"Ndtri", 1e-12, 0.999999, Ndtri, NdtriTo},
		{"NdtriC", 0.26, 0.74, Ndtri, NdtriTo},
		{"NdtriT", 1e-12, 0.25, Ndtri, NdtriTo},
		{"Erf", -5, 5, Erf, ErfTo},
		{"Erfc", -5, 25, Erfc, ErfcTo},
		{"Ndtr", -30, 10, Ndtr, NdtrTo},
		{"LogNdtr", -100, 10, LogNdtr, LogNdtrTo},
		{"ErfInv", -0.999, 0.999, ErfInv, ErfInvTo},
		{"ErfcInv", 1e-10, 1.999, ErfcInv, ErfcInvTo},
		{"Expit", -50, 50, Expit, ExpitTo},
		{"Logit", 1e-6, 0.999999, Logit, LogitTo},
	}
	dst := make([]float64, 10000)
	for _, r := range rows {
		x := pts(r.lo, r.hi)
		b.Run(r.name, func(b *testing.B) {
			for j := 0; j < b.N; j++ {
				for i, v := range x {
					dst[i] = r.f(v)
				}
			}
		})
		b.Run(r.name+"To", func(b *testing.B) {
			for j := 0; j < b.N; j++ {
				r.to(dst, x)
			}
		})
	}
}
