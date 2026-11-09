package special

import (
	"fmt"
	"testing"
)

// The refbench inputs of the rows listed in #33 (bench/perf_extra_test.go):
// 10000 points, order i mod 21 and x on (0.1, 50) for the integer-order
// Bessel rows; comb arguments N = 20 + i mod 80, k = 7i mod 21.
const gapsN = 10000

func gapsX(lo, hi float64) []float64 {
	x := make([]float64, gapsN)
	for i := range x {
		x[i] = lo + (hi-lo)*float64(i)/float64(gapsN)
	}
	return x
}

func gapsSink(b *testing.B, f func(i int) float64) {
	var s float64
	for j := 0; j < b.N; j++ {
		for i := 0; i < gapsN; i++ {
			s += f(i)
		}
	}
	besselSink = s
}

func BenchmarkGaps(b *testing.B) {
	x := gapsX(0.1, 50)
	big := gapsX(0.1, 100)
	cn, ck := make([]int64, gapsN), make([]int64, gapsN)
	for i := range cn {
		cn[i], ck[i] = int64(i%80+20), int64(i*7%21)
	}
	bn, bk := gapsX(0.5, 40), gapsX(0, 40)
	red := gapsX(-50, 50)
	dst := make([]float64, gapsN)
	rows := []struct {
		name string
		f    func(i int) float64
	}{
		{"Jn", func(i int) float64 { return Jn(i%21, x[i]) }},
		{"Yn", func(i int) float64 { return Yn(i%21, x[i]) }},
		{"In", func(i int) float64 { return In(i%21, x[i]) }},
		{"Kn", func(i int) float64 { return Kn(i%21, x[i]) }},
		{"I1", func(i int) float64 { return I1(x[i]) }},
		{"I1e", func(i int) float64 { return I1e(x[i]) }},
		{"K1", func(i int) float64 { return K1(x[i]) }},
		{"K1e", func(i int) float64 { return K1e(x[i]) }},
		{"Iv", func(i int) float64 { return Iv(float64(i%21)+0.5, x[i]) }},
		{"Kv", func(i int) float64 { return Kv(float64(i%21)+0.5, x[i]) }},
		{"Comb", func(i int) float64 { return Comb(cn[i], ck[i]) }},
		{"Binom", func(i int) float64 { return Binom(bn[i], bk[i]) }},
		{"Factorial", func(i int) float64 { return Factorial(big[i]) }},
		{"Factorial2", func(i int) float64 { return Factorial2(i % 201) }},
		{"GammaSign", func(i int) float64 { return GammaSign(x[i] - 25) }},
		{"Xlog1py", func(i int) float64 { return Xlog1py(x[i], x[i]/50-0.5) }},
		{"BetaIncInv", func(i int) float64 { return BetaIncInv(1+float64(i%7), 2+float64(i%5), float64(i%99+1)/100) }},
		{"BetaInc", func(i int) float64 { return BetaInc(1+float64(i%7), 2+float64(i%5), float64(i%99+1)/100) }},
	}
	for _, r := range rows {
		b.Run(r.name, func(b *testing.B) { gapsSink(b, r.f) })
	}
	b.Run("Softmax", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			Softmax(dst, red)
		}
	})
	b.Run("LogSoftmax", func(b *testing.B) {
		for j := 0; j < b.N; j++ {
			LogSoftmax(dst, red)
		}
	})
	_ = fmt.Sprint
}
