package special

import (
	"math"
	"math/big"
	"testing"
)

// An oracle for K_v(x), v > 0 not an integer, 0 < x < 100, that shares
// nothing with the package or with mpmath: math/big at 320 bits through the
// helpers of i_oracle_test.go. For 0 < nu < 1 the two power series of
// K_nu = pi/(2 sin(nu pi)) (I_-nu - I_nu) (DLMF 10.27.4, 10.25.2), with
// Gamma(1 +- nu) from bigLogGamma (both arguments positive); the order
// v = n + u (n the nearest integer, |u| <= 1/2, a = |u|) by the upward
// recurrence K_{w+1} = K_{w-1} + (2w/x) K_w (DLMF 10.29.1) from K_u = K_a and
// K_{u+1} = K_{1-a} (u < 0) or K_{1-a} + (2a/x) K_a (u > 0). The cancellation
// of I_-nu - I_nu is about e^2x / (a ln) of the result: 320 bits hold orders
// down to a = 2^-100 at x = 100 with more than 60 bits to spare.
// TestKvOracleClosedForms holds it against the exact K_{1/2} and K_{3/2}.

// bigSinPi returns sin(pi a) for 0 < a < 1 (as cos(pi (1/2 - a)), |.| <= pi/2).
func bigSinPi(a float64) *big.Float {
	ivOracleInit()
	d := bnew().Sub(bf(0.5), bf(a))
	return bigCos(bnew().Mul(bigPi, d))
}

// bigKnu returns K_nu(x) for 0 < nu < 1, nu given exactly (a big.Float: 1 - a
// is not a float64) with sin(pi nu).
func bigKnu(nu, sinNu *big.Float, x float64) *big.Float {
	ivOracleInit()
	hx := bnew().Quo(bf(x), bf(2))
	lh := bigLog(hx)
	q := bnew().Mul(hx, hx)
	series := func(s float64) *big.Float { // I_{s nu}(x), s = +-1
		ord := bnew().Mul(bf(s), nu)
		lt := bnew().Mul(ord, lh)
		lt.Sub(lt, bigLogGamma(bnew().Add(bf(1), ord)))
		term := bigExp(lt)
		sum := bnew().Set(term)
		for k := 1; k < 5000; k++ {
			den := bnew().Mul(bf(float64(k)), bnew().Add(bf(float64(k)), ord))
			term.Mul(term, q)
			term.Quo(term, den)
			sum.Add(sum, term)
			if float64(k)*float64(k) > x*x && term.Cmp(bnew().SetMantExp(sum, -340)) < 0 {
				break
			}
		}
		return sum
	}
	d := bnew().Sub(series(-1), series(1))
	d.Mul(d, bigPi)
	return d.Quo(d, bnew().Mul(bf(2), sinNu))
}

// bigKpair returns K_u(x) and K_{u+1}(x) (unscaled) for 0 < |u| <= 1/2.
func bigKpair(u, x float64) (k0, k1 *big.Float) {
	a := math.Abs(u)
	sa := bigSinPi(a) // sin(pi (1 - a)) = sin(pi a)
	ka := bigKnu(bf(a), sa, x)
	k1a := bigKnu(bnew().Sub(bf(1), bf(a)), sa, x)
	k0 = ka
	if u < 0 {
		k1 = k1a
	} else {
		k1 = bnew().Add(k1a, bnew().Mul(bnew().Quo(bf(float64(2*a)), bf(x)), ka))
	}
	return
}

// bigKve returns K_v(x) e^x for non-integer v > 0 (v = n + u with |u| >= 2^-100).
func bigKve(v, x float64) *big.Float {
	ivOracleInit()
	n := math.Round(v)
	u := v - n
	k0, k1 := bigKpair(u, x)
	if n == 0 {
		return bnew().Mul(k0, bigExp(bf(x)))
	}
	for k := 1; k < int(n); k++ {
		w := bnew().Add(bf(u), bf(float64(k)))
		t := bnew().Quo(bnew().Mul(bf(2), w), bf(x))
		k0, k1 = k1, bnew().Add(k0, bnew().Mul(t, k1))
	}
	return bnew().Mul(k1, bigExp(bf(x)))
}

// TestKvOracleClosedForms: K_{1/2}(x) = sqrt(pi/(2x)) e^-x, K_{3/2} = K_{1/2}
// (1 + 1/x), K_{5/2} = K_{1/2} (1 + 3/x + 3/x^2), exactly.
func TestKvOracleClosedForms(t *testing.T) {
	ivOracleInit()
	for _, x := range []float64{0.01, 0.7, 2, 9.5, 23.9} {
		pre := bnew().Quo(bigPi, bnew().Mul(bf(2), bf(x)))
		pre.Sqrt(pre) // K_{1/2} e^x
		for _, c := range []struct {
			v    float64
			poly *big.Float
		}{
			{0.5, bf(1)},
			{1.5, bnew().Add(bf(1), bnew().Quo(bf(1), bf(x)))},
			{2.5, bnew().Add(bf(1), bnew().Add(bnew().Quo(bf(3), bf(x)), bnew().Quo(bf(3), bnew().Mul(bf(x), bf(x)))))},
		} {
			want := bnew().Mul(pre, c.poly)
			got := bigKve(c.v, x)
			rel := bnew().Quo(bnew().Abs(bnew().Sub(got, want)), want)
			// v = 1/2 reduces to a = 1/2: sin(pi/2) = 1 and K_{1-a} = K_{1/2}
			if rel.Cmp(bnew().SetMantExp(bf(1), -200)) > 0 {
				t.Errorf("bigKve(%g, %g) relative %v from the closed form", c.v, x, rel.Text('g', 4))
			}
		}
	}
}

// kveUlps returns |got - want| in ulps of want.
func kveUlps(got float64, want *big.Float) float64 { return ivOracleUlps(got, want) }

// relDD returns |(h + l) - want| / want.
func relDD(h, l float64, want *big.Float) float64 {
	got := bnew().Add(bf(h), bf(l))
	d := bnew().Sub(got, want)
	d.Quo(d, want)
	f, _ := d.Float64()
	return math.Abs(f)
}
