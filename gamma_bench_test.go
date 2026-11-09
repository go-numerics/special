package special

import "testing"

// benchPoints returns refbench's points lo + (hi-lo)·frac(i·0.618…).
func benchPoints(n int, lo, hi float64) []float64 {
	const g = 0.6180339887498949
	x := make([]float64, n)
	for i := range x {
		f := float64(float64(i) * g)
		f -= float64(int64(f))
		x[i] = lo + float64((hi-lo)*f)
	}
	return x
}

var gammaSink float64

// BenchmarkGamma times Gamma per region; "refbench" is the distribution
// of the refbench row (0.1, 170).
func BenchmarkGamma(b *testing.B) {
	for _, r := range []struct {
		name   string
		lo, hi float64
	}{
		{"refbench", 0.1, 170},
		{"tiny", 1e-20, 1e-17},
		{"0to1", 0.01, 1},
		{"1to10", 1, 10},
		{"10to171", 10, 171},
		{"m0.5to0", -0.5, -0.01},
		{"m34tom0.5", -34, -0.5},
		{"m190tom34", -190, -34},
	} {
		x := benchPoints(10000, r.lo, r.hi)
		b.Run(r.name, func(b *testing.B) {
			s := 0.0
			for i := 0; i < b.N; i++ {
				s += Gamma(x[i%len(x)])
			}
			gammaSink = s
		})
		b.Run(r.name+"/To", func(b *testing.B) {
			dst := make([]float64, len(x))
			b.SetBytes(0)
			for i := 0; i < b.N; i += len(x) {
				GammaTo(dst, x)
			}
		})
	}
}
