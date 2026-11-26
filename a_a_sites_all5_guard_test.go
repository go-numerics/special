package special

import (
	"math"
	"testing"
)

// Fifth chunk of `mutsweep -sites all` guards. References are the standard
// library's Jn/Yn (FreeBSD msun), DLMF 10.4.1 for negative orders, and
// exact IEEE scaling for the lemma.

// TestBesselJYPosNeedYFalse (bessel_jy.go:479): with needY false the Y slot
// is NaN (the contract of besselJYPos: a function not asked for is not
// computed), and J is J_v(x) (math.Jn). The orders and arguments are in the
// x < v region (CF1 at v, backward recurrence, Steed at u).
func TestBesselJYPosNeedYFalse(t *testing.T) {
	for _, c := range []struct {
		v int
		x float64
	}{{20, 10}, {30, 15}, {12, 8}, {40, 25}, {25, 20}} {
		J, Y := besselJYPos(float64(c.v), c.x, true, false)
		if !math.IsNaN(Y) {
			t.Errorf("besselJYPos(%d, %g, true, false) Y = %v, want NaN", c.v, c.x, Y)
		}
		want := math.Jn(c.v, c.x)
		if rel := math.Abs(J-want) / math.Abs(want); !(rel <= 1e-10) {
			t.Errorf("besselJYPos(%d, %g) J = %v, stdlib %v (rel %g)", c.v, c.x, J, want, rel)
		}
	}
}

// TestYnRecurNegativeOrder (jn_yn_fast.go:208): ynRecur(-n, x) answers like
// ynRecur(n, x) (the function certifies |n|; the sign (-1)^n of DLMF 10.4.1
// is the caller's), and is not declined as an order below 2.
func TestYnRecurNegativeOrder(t *testing.T) {
	for _, n := range []int{2, 3, 7, 20} {
		for _, x := range []float64{1.5, 10, 33.3} {
			rp, okp := ynRecur(n, x)
			rn, okn := ynRecur(-n, x)
			if okp != okn || math.Float64bits(rp) != math.Float64bits(rn) {
				t.Errorf("ynRecur(%d, %g) = (%v, %v), ynRecur(%d) = (%v, %v)", -n, x, rn, okn, n, rp, okp)
			}
			if okp {
				want := math.Yn(n, x)
				if rel := math.Abs(rp-want) / math.Abs(want); !(rel <= 1e-12) {
					t.Errorf("ynRecur(%d, %g) = %v, stdlib %v (rel %g)", n, x, rp, want, rel)
				}
			}
		}
	}
}

// TestLemmaPow2IsLdexpAtNormalExponents (gamma_fast.go:161 k >= -1021): for a
// normal result the scaling eh*pow2(k) is exact and equals math.Ldexp(eh, k)
// bit for bit, so which side of the k = -1021 test a value takes does not
// change it (the guard only picks the cheaper multiply).
func TestLemmaPow2IsLdexpAtNormalExponents(t *testing.T) {
	for _, eh := range []float64{1, 1.5, 1.9999999999999998, 1.2345678901234567, 0.5000000000000001} {
		for k := -1021; k <= -980; k++ {
			a, b := eh*pow2(k), math.Ldexp(eh, k)
			if math.Float64bits(a) != math.Float64bits(b) {
				t.Errorf("%v*pow2(%d) = %v, Ldexp = %v", eh, k, a, b)
			}
		}
	}
}
