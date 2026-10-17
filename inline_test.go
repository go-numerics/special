package special

import (
	"math"
	"math/rand"
	"testing"
)

// TestInlinedLoopsBitIdentical holds the generated array variants
// (zz_inline.go) to their scalar functions bit for bit, on the fixtures'
// arguments, special values and random arguments over every scale.
func TestInlinedLoopsBitIdentical(t *testing.T) {
	r := rand.New(rand.NewSource(7))
	pts := []float64{0, math.Copysign(0, -1), 1, -1, 0.5, 2, 1e-300, 5e-324, -5e-324,
		math.Inf(1), math.Inf(-1), math.NaN(), math.MaxFloat64, -math.MaxFloat64, 0x1p-1022}
	for i := 0; i < 30000; i++ {
		s := 1.0
		if r.Intn(2) == 0 {
			s = -1
		}
		pts = append(pts, s*math.Exp(r.Float64()*1400-700), s*r.Float64()*60, r.Float64())
	}
	unary := map[string]struct {
		vec    func(dst, x []float64)
		scalar func(float64) float64
	}{
		"gamma": {GammaTo, Gamma}, "loggamma": {LogGammaTo, LogGamma}, "digamma": {DigammaTo, Digamma},
		"erf": {ErfTo, Erf}, "erfc": {ErfcTo, Erfc}, "erfcx": {ErfcxTo, Erfcx},
		"erfinv": {ErfInvTo, ErfInv}, "erfcinv": {ErfcInvTo, ErfcInv},
		"ndtr": {NdtrTo, Ndtr}, "ndtri": {NdtriTo, Ndtri}, "log_ndtr": {LogNdtrTo, LogNdtr},
		"expit": {ExpitTo, Expit}, "logit": {LogitTo, Logit}, "log1pexp": {Log1pExpTo, Log1pExp},
		"zeta": {ZetaTo, Zeta}, "polygamma": {TrigammaTo, func(x float64) float64 { return Polygamma(1, x) }},
	}
	for name, f := range unary {
		var x []float64
		for _, v := range pts {
			// Polygamma's negative-x path sums ceil(-x) terms: keep it short.
			if name != "polygamma" || v > -1e4 {
				x = append(x, v)
			}
		}
		if name != "polygamma" {
			for _, row := range loadFixture(t, name) {
				x = append(x, row.args[0])
			}
		}
		dst := make([]float64, len(x))
		f.vec(dst, x)
		for i, v := range x {
			if w := f.scalar(v); math.Float64bits(w) != math.Float64bits(dst[i]) && !(w != w && dst[i] != dst[i]) {
				t.Errorf("%s(%v): array %v, scalar %v", name, v, dst[i], w)
				break
			}
		}
	}
	binary := map[string]struct {
		vec    func(dst, a, b []float64, sa, sb int)
		scalar func(float64, float64) float64
	}{
		"beta": {BetaTo, Beta}, "logbeta": {LogBetaTo, LogBeta}, "xlogy": {XlogyTo, Xlogy},
		"zeta_hurwitz": {HurwitzZetaTo, HurwitzZeta},
	}
	for name, f := range binary {
		var a, b []float64
		for _, row := range loadFixture(t, name) {
			a, b = append(a, row.args[0]), append(b, row.args[1])
		}
		for i := 0; i+1 < len(pts); i += 2 {
			if name == "zeta_hurwitz" && math.Abs(pts[i]) > 1e6 {
				continue // hurwitzPos sizes its term slice by s
			}
			a, b = append(a, pts[i]), append(b, pts[i+1])
			a, b = append(a, math.Abs(pts[i])), append(b, math.Abs(pts[i+1]))
		}
		dst := make([]float64, len(a))
		f.vec(dst, a, b, 1, 1)
		for i := range a {
			if w := f.scalar(a[i], b[i]); math.Float64bits(w) != math.Float64bits(dst[i]) && !(w != w && dst[i] != dst[i]) {
				t.Errorf("%s(%v, %v): array %v, scalar %v", name, a[i], b[i], dst[i], w)
				break
			}
		}
		// stride 0: the first operand repeated
		f.vec(dst[:10], a[:1], b, 0, 1)
		for i := 0; i < 10; i++ {
			if w := f.scalar(a[0], b[i]); math.Float64bits(w) != math.Float64bits(dst[i]) && !(w != w && dst[i] != dst[i]) {
				t.Errorf("%s stride 0 (%v, %v): array %v, scalar %v", name, a[0], b[i], dst[i], w)
			}
		}
	}
}
