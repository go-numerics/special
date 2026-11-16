package special

import (
	"math"
	"testing"

	"github.com/go-numerics/special/dd"
)

// softmaxDiv must return dd.Div(a, b).Float() for every pair: 2e7 random
// pairs shaped like Softmax's (a is a double-double exponential in (0, 1], b
// a sum of them), 2e5 under -race and coverage.
func TestSoftmaxDivEqualsReference(t *testing.T) {
	n := 20000000
	if testing.Short() || raceEnabled {
		n = 200000
	}
	r := gapsRand(7)
	for i := 0; i < n; i++ {
		x := -float64(r() * 40)
		m, k := dd.ExpMant(dd.F(x))
		a := dd.Ldexp(m, k)
		b := dd.Add(a, dd.F(float64(r()*float64(1+int(r()*4000)))))
		got, want := softmaxDiv(a, b), dd.Div(a, b).Float()
		if got != want {
			t.Fatalf("softmaxDiv(%v, %v) = %v, reference %v", a, b, got, want)
		}
	}
}

// A quotient exactly halfway between two floats cannot be certified and goes
// to the double-double division.
func TestSoftmaxDivFallback(t *testing.T) {
	a := dd.DD{Hi: 1.5, Lo: 0x1p-53} // 1.5 + half an ulp of 1.5
	b := dd.F(1)
	if got, want := softmaxDiv(a, b), dd.Div(a, b).Float(); got != want {
		t.Fatalf("softmaxDiv at a tie = %v, reference %v", got, want)
	}
	if _, ok := softmaxDivCertified(a, b); ok {
		t.Fatal("a tie was certified")
	}
	_ = math.Pi
}
