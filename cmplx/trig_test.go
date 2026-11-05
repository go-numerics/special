package cmplx

import (
	"math"
	"testing"
)

func TestTrigReduce(t *testing.T) {
	// A fraction whose first word is zero keeps its relative accuracy.
	if f := fracDD([4]uint64{0, 1 << 63, 0, 0}); f.Hi != 0x1p-65 || f.Lo != 0 {
		t.Errorf("fracDD(2^-65) = %v", f)
	}
	if f := fracDD([4]uint64{0, 0, 0, 1}); f.Hi != 0x1p-256 {
		t.Errorf("fracDD(2^-256) = %v", f)
	}
	// sin(1e22), the classic test of a reduction (correctly rounded value).
	if s, _ := sincos(1e22); s != -0.8522008497671888 {
		t.Errorf("sin(1e22) = %v, want -0.8522008497671888", s)
	}
	for _, y := range []float64{0.5, -1, 3, 1e5, -1e16, 1e300, math.MaxFloat64} {
		s, c := sincos(y)
		if d := math.Abs(math.FMA(s, s, c*c) - 1); d > 4e-16 {
			t.Errorf("sin²+cos² - 1 = %g at %g", d, y)
		}
		if ts := tan(y); math.Abs(ts-s/c) > 4e-16*math.Abs(ts) {
			t.Errorf("tan(%g) = %v, sin/cos = %v", y, ts, s/c)
		}
	}
}
