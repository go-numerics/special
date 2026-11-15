package special

import (
	"math"
	"math/rand"
	"testing"
)

// TestHankelPairCap holds the cap on the number of pairs of terms from both
// sides: with maxL = l (the number the answer used) it answers the same, with
// maxL = l - 1 it declines. Orders up to 10 only: the first call runs without
// a lowered cap, and if the cap were ever removed the test must reach its
// assertions, not a table bound. This file sorts before jv_hankel_fast_test.go
// for the same reason: Go runs the tests in file order.
func TestHankelPairCap(t *testing.T) {
	r := rand.New(rand.NewSource(3))
	checked := 0
	for i := 0; i < 400; i++ {
		v, x := hankelDraw(r)
		v = math.Mod(v, 10)
		if v == 0 {
			continue
		}
		for _, y := range []bool{false, true} {
			got, _, l, ok := hankelFastL(v, x, y, hankelMaxL)
			if !ok {
				continue
			}
			checked++
			again, _, l2, ok2 := hankelFastL(v, x, y, l)
			if !ok2 || again != got || l2 != l {
				t.Fatalf("v=%v x=%v: maxL = l = %d changed the answer (%v, ok %v, l %d; was %v)", v, x, l, again, ok2, l2, got)
			}
			if _, _, _, ok3 := hankelFastL(v, x, y, l-1); ok3 {
				t.Fatalf("v=%v x=%v: maxL = l - 1 = %d still answers", v, x, l-1)
			}
		}
	}
	if checked < 100 {
		t.Errorf("only %d answers", checked)
	}
}
