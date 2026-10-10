package dd

import (
	"math"
	"math/big"
	"testing"
)

func bf(a DD) *big.Float {
	x := new(big.Float).SetPrec(300).SetFloat64(a.Hi)
	return x.Add(x, new(big.Float).SetPrec(300).SetFloat64(a.Lo))
}

func relErr(got DD, want *big.Float) float64 {
	d := new(big.Float).SetPrec(300).Sub(bf(got), want)
	r, _ := new(big.Float).Quo(d, want).Float64()
	return math.Abs(r)
}

func TestMulDivExact(t *testing.T) {
	p := F(1)
	exact := new(big.Float).SetPrec(2000).SetInt64(1)
	for k := 1; k <= 170; k++ {
		p = MulF(p, float64(k))
		exact.Mul(exact, new(big.Float).SetInt64(int64(k)))
		if e := relErr(p, exact); e > 1e-29 {
			t.Fatalf("k=%d rel err %g", k, e)
		}
	}
}
