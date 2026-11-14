package special

import (
	"math"
	"testing"
)

// ynBoundSane reports why e is not a possible error bound: a bound is a
// non-negative number, never NaN.
func ynBoundSane(e float64) string {
	switch {
	case math.IsNaN(e):
		return "NaN"
	case e < 0:
		return "negative"
	}
	return ""
}

// besselYZeros returns the zeros of Y_m in (lo, hi), by sign changes of
// math.Yn on a fine grid and bisection (the zeros are only needed to 1e-13, the
// callers probe a neighbourhood around them).
func besselYZeros(m int, lo, hi float64) []float64 {
	var zs []float64
	const step = 0.01
	a, fa := lo, math.Yn(m, lo)
	for x := lo + step; x <= hi; x += step {
		fx := math.Yn(m, x)
		if (fa < 0) != (fx < 0) {
			l, r := a, x
			for i := 0; i < 60; i++ {
				mid := float64((l + r) / 2)
				if (math.Yn(m, mid) < 0) == (fa < 0) {
					l = mid
				} else {
					r = mid
				}
			}
			zs = append(zs, float64((l+r)/2))
		}
		a, fa = x, fx
	}
	return zs
}

// TestYnCertificateBoundNeverNegative holds `ylow > 0` in ynRecurErr by an
// assertion. Next to a zero of Y_{n-1} the chain's own Y_{n-1} is smaller than
// the 2^-36 taken away from it; without the guard the Wronskian bound divides by
// a non-positive number, the sharper |J_n| bound becomes negative, and so does
// the certificate's error bound e (which then passes e <= jyAcceptY*ay, a wrong
// acceptance: mutsweep reported `ylow > 0` as held by nothing, special#77).
func TestYnCertificateBoundNeverNegative(t *testing.T) {
	// The zero of Y_2 that special#77 names.
	if _, e, ok := ynRecurErr(3, 3.3842417671495713); ok {
		if why := ynBoundSane(e); why != "" {
			t.Errorf("ynRecurErr(3, 3.3842417671495713): bound %g is %s", e, why)
		}
	}
	probed, accepted := 0, 0
	for n := 2; n <= 10; n++ {
		for _, z := range besselYZeros(n-1, 0.5, 40) {
			if !(float64(n) <= 40) {
				continue
			}
			for k := -40; k <= 40; k++ {
				for _, d := range []float64{float64(float64(k) * 1e-12), float64(float64(k) * 1e-13), float64(float64(k) * 0x1p-45)} {
					x := z + d
					_, e, ok := ynRecurErr(n, x)
					probed++
					if !ok {
						continue
					}
					accepted++
					if why := ynBoundSane(e); why != "" {
						t.Fatalf("ynRecurErr(%d, %.17g): bound %g is %s", n, x, e, why)
					}
				}
			}
		}
	}
	if probed == 0 || accepted == 0 {
		t.Fatalf("probed %d points, %d accepted: the sweep checks nothing", probed, accepted)
	}
}

// TestJnMillerFinalComparison holds `return rel <= jyAcceptJ` of jnMiller: a
// start pair of J_0 whose error is large makes rel exceed jyAcceptJ, and the
// value must be declined. No (n, x) in range gets there by itself (the chain
// overflows first for the n that would), so the start pair is handed in
// through jnMillerStart; the unmodified pair must still be accepted, which
// keeps the test from passing on a function that always declines.
func TestJnMillerFinalComparison(t *testing.T) {
	const n, x = 5, 3.0
	if _, _, ok := jnMiller(n, x); !ok {
		t.Fatal("jnMiller(5, 3) declines with the true start pair")
	}
	old := jnMillerStart
	defer func() { jnMillerStart = old }()
	jnMillerStart = func(x float64) (jyPair, jyPair) {
		j0, j1 := besselJ0Pair(x), besselJ1Pair(x)
		j0.eb, j1.eb = float64(0.5*math.Abs(j0.h)), float64(0.5*math.Abs(j1.h)) // relative 1/2 >> 2^-51
		return j0, j1
	}
	if r, e, ok := jnMiller(n, x); ok {
		t.Errorf("jnMiller(5, 3) accepts a start error of 1/2 of the value: r = %g, e = %g", r, e)
	}
}
