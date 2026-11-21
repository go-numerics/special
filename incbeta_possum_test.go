package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/dd"
)

// posSumRef is Σ d_i, d_0 = 1, d_{i+1} = d_i (a+b+i) x/(a+1+i), every
// operation in double-double, to terms below 2^-110 of the sum.
func posSumRef(a, b, x float64) dd.DD {
	ab, a1 := sumDD(a, b), sumDD(a, 1)
	d, s := dd.F(1), dd.F(1)
	for i := 0.0; i < 1e7; i++ {
		f := dd.Div(dd.MulF(dd.AddF(ab, i), x), dd.AddF(a1, i))
		d = dd.Mul(d, f)
		s = dd.Add(s, d)
		if d.Hi < 0x1p-110*s.Hi {
			break
		}
	}
	return s
}

// TestIncBetaPosSumBound holds ibetaPosSum, including its float64 tail summed
// four terms per division, within 2^-56 of a double-double reference. The
// a = 1e300 rows hold the guard (e3 < 2^200) that keeps the product of four
// denominators finite: without it that product overflows (from e ~ 2^256),
// q·p is 0·Inf and the sum is NaN. The rows on either side of 2^200 take
// the blocked and the per-term loop.
func TestIncBetaPosSumBound(t *testing.T) {
	worst := 0.0
	for _, a := range []float64{0.5, 3, 37.5, 1e3, 1e6, 0x1p190, 0x1p199, 0x1p201, 0x1p230, 1e300} {
		for _, b := range []float64{0.5, 2, 30} {
			for _, x := range []float64{0.01, 0.1, 0.5, 0.7, 0.9, 0.97} {
				got := ibetaPosSum(a, b, dd.F(x), 0, 1<<30, ibetaEps)
				ref := posSumRef(a, b, x)
				rel := math.Abs(dd.Sub(got, ref).Float() / ref.Hi)
				// written so that a NaN fails: q·p is 0·Inf once the
				// denominators overflow
				if !(rel <= 0x1p-56) {
					t.Errorf("a=%v b=%v x=%v: relative error %.3g, want <= 2^-56 (got %v, want %v)", a, b, x, rel, got.Float(), ref.Float())
					continue
				}
				worst = max(worst, rel)
			}
		}
	}
	t.Logf("worst relative error 2^%.1f", math.Log2(worst))
}
