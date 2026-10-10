package special

import (
	"math"
	"math/big"
	"testing"
)

// sinPi/cosPi against a 200-bit reference built with math/big from the
// Taylor series of sin(πr) and cos(πr) on the reduced argument.
func bigSinCosPi(x float64) (float64, float64) {
	const prec = 200
	// pi to 200 bits
	pi, _ := new(big.Float).SetPrec(prec).SetString("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
	// reduce x mod 2 exactly
	r := math.Mod(x, 2)
	a := new(big.Float).SetPrec(prec).Mul(pi, new(big.Float).SetPrec(prec).SetFloat64(r))
	a2 := new(big.Float).SetPrec(prec).Mul(a, a)
	sin := new(big.Float).SetPrec(prec)
	cos := new(big.Float).SetPrec(prec)
	term := new(big.Float).SetPrec(prec).Set(a)
	cterm := new(big.Float).SetPrec(prec).SetFloat64(1)
	for k := 1; k < 80; k++ {
		sin.Add(sin, term)
		cos.Add(cos, cterm)
		term.Mul(term, a2)
		term.Quo(term, new(big.Float).SetFloat64(float64(-(2*k)*(2*k+1))))
		cterm.Mul(cterm, a2)
		cterm.Quo(cterm, new(big.Float).SetFloat64(float64(-(2*k-1)*(2*k))))
	}
	s, _ := sin.Float64()
	c, _ := cos.Float64()
	return s, c
}

func TestSinCosPi(t *testing.T) {
	worstS, worstC := 0.0, 0.0
	for i := 0; i < 20000; i++ {
		x := -8 + 16*float64(i)/20000 + 1e-7*float64(i%7)
		ws, wc := bigSinCosPi(x)
		if math.Abs(ws) > 1e-40 {
			worstS = math.Max(worstS, math.Abs(sinPi(x)-ws)/ulp(ws))
		}
		if math.Abs(wc) > 1e-40 {
			worstC = math.Max(worstC, math.Abs(cosPi(x)-wc)/ulp(wc))
		}
	}
	t.Logf("sinPi max %.3f ulp, cosPi max %.3f ulp", worstS, worstC)
	if worstS > 1 || worstC > 1 {
		t.Errorf("sinPi %.3f / cosPi %.3f ulp > 1", worstS, worstC)
	}
	for _, x := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if !math.IsNaN(sinPi(x)) || !math.IsNaN(cosPi(x)) {
			t.Errorf("sinPi/cosPi(%v) not NaN", x)
		}
	}
	if s := sinPi(-3); s != 0 {
		t.Errorf("sinPi(-3) = %v, want 0", s)
	}
	if c := cosPi(0.5); c != 0 {
		t.Errorf("cosPi(0.5) = %v", c)
	}
}
