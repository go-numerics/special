package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/dd"
)

// I1 and I1e on [1/8, 512) come from the octave interpolants; the
// double-double power series (besselI1SmallDD) is an independent oracle on
// [1/8, 20], good to ~2^-100.
func TestI1OctaveVsSeries(t *testing.T) {
	worst := 0.0
	for i := 0; i < 200000; i++ {
		x := 0.125 + 20*float64(i)/200000
		s := besselI1SmallDD(x)
		want := s.Float()
		em, ek := dd.ExpMant(dd.F(-x))
		wantE := dd.Ldexp(dd.Mul(s, em), ek).Float()
		for _, c := range []struct {
			name      string
			got, want float64
		}{{"I1", I1(x), want}, {"I1e", I1e(x), wantE}} {
			u := math.Abs(c.got-c.want) / (math.Nextafter(math.Abs(c.want), math.Inf(1)) - math.Abs(c.want))
			if u > worst {
				worst = u
			}
			if u > 1 {
				t.Fatalf("%s(%v) = %v, series %v: %.2f ulp", c.name, x, c.got, c.want, u)
			}
		}
	}
	t.Logf("worst difference to the series: %.3f ulp", worst)
}
