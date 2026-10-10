package special

import "testing"

// Representative arguments: small, moderate, near the order, large.
var (
	besselBenchX  = []float64{1e-5, 0.3, 1.7, 4.2, 9.9, 23.5, 77.0, 640.0, 1.5e5}
	besselBenchVX = [][2]float64{
		{0.3, 0.7}, {1.5, 4.2}, {2.7, 13.1}, {7.25, 3.3}, {12.5, 30.0},
		{-3.7, 8.8}, {45.5, 44.0}, {100.25, 160.0}, {0.75, 1e4},
	}
	besselSink float64
)

func BenchmarkJ0(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, x := range besselBenchX {
			besselSink += J0(x)
		}
	}
}

func BenchmarkK0(b *testing.B) {
	for i := 0; i < b.N; i++ {
		for _, x := range besselBenchX {
			besselSink += K0(x)
		}
	}
}

func benchVX(b *testing.B, f func(v, x float64) float64) {
	for i := 0; i < b.N; i++ {
		for _, a := range besselBenchVX {
			besselSink += f(a[0], a[1])
		}
	}
}

func BenchmarkJv(b *testing.B) { benchVX(b, Jv) }
func BenchmarkYv(b *testing.B) { benchVX(b, Yv) }
func BenchmarkIv(b *testing.B) { benchVX(b, Iv) }
func BenchmarkKv(b *testing.B) { benchVX(b, Kv) }
