package dd

import (
	"math"
	"math/rand/v2"
	"testing"
)

var logSink float64

// BenchmarkLog times dd.Log over 4096 arguments: log-uniform over
// [1e-300, 1e300], and within 2^-7 of 1 (the cells next to 1).
func BenchmarkLog(b *testing.B) {
	r := rand.New(rand.NewPCG(3, 5))
	for _, tc := range []struct {
		name string
		gen  func() float64
	}{
		{"wide", func() float64 { return math.Exp((r.Float64() - 0.5) * 1380) }},
		{"near1", func() float64 { return 1 + (r.Float64()-0.5)*0x1p-6 }},
	} {
		xs := make([]float64, 4096)
		for i := range xs {
			xs[i] = tc.gen()
		}
		b.Run(tc.name, func(b *testing.B) {
			s := 0.0
			for b.Loop() {
				for _, x := range xs {
					s += Log(x).Hi
				}
			}
			logSink = s
		})
	}
}
