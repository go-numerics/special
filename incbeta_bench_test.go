package special

import "testing"

// BenchmarkBetaIncBox is the refbench box of #33: a, b in (0.5, 50), x in
// (0.01, 0.99), 10000 points; ns/op / 10000 = ns per element.
func BenchmarkBetaIncBox(b *testing.B) {
	const n = 10000
	r := gapsRand(1)
	pa, pb, px := make([]float64, n), make([]float64, n), make([]float64, n)
	pp := make([]float64, n)
	for i := range pa {
		pa[i] = 0.5 + float64(r()*49.5)
		pb[i] = 0.5 + float64(r()*49.5)
		px[i] = 0.01 + float64(r()*0.98)
		pp[i] = 0.01 + float64(r()*0.98)
	}
	for _, f := range []struct {
		name string
		f    func(a, b, x float64) float64
		p    bool
	}{{"BetaInc", BetaInc, false}, {"BetaIncC", BetaIncC, false}, {"BetaIncInv", BetaIncInv, true}, {"BetaIncCInv", BetaIncCInv, true}} {
		b.Run(f.name, func(b *testing.B) {
			var s float64
			for j := 0; j < b.N; j++ {
				for i := 0; i < n; i++ {
					x := px[i]
					if f.p {
						x = pp[i]
					}
					s += f.f(pa[i], pb[i], x)
				}
			}
			besselSink = s
		})
	}
}
