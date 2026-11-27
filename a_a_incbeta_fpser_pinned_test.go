package special

import "testing"

// TestBetaIncCFpserPinned pins the complement where a (after the swap for
// x > 1/2) parameter is below eps*min(1, other): the branch of ibetaFpser
// (incbeta.go, b0 < min(eps, eps*a0)). Forcing it off moves the last bit.
// The rows were found by comparing 150000 random inputs against that build;
// the expected bits are the series' own.
func TestBetaIncCFpserPinned(t *testing.T) {
	rows := []struct{ a, b, x, want float64 }{
		{0x1.678f6f46ea16ep-80, 0x1.856fd87c7289ap-10, 0x1.8db23756fe3dbp-01, 0x1.d7d878af6fd81p-71},
		{0x1.d8735ebced4b6p-83, 0x1.daeb94616181dp-01, 0x1.ffffffffaa5dap-01, 0x1.e35927367a8f7p-115},
		{0x1.c3bddcdd4ab26p-71, 0x1.42ccfe54b2bbep+05, 0x1.ffbb846410af7p-01, 0x1.7d8284c2f7b96p-516},
	}
	for i, r := range rows {
		if got := BetaIncC(r.a, r.b, r.x); got != r.want {
			t.Errorf("row %d: BetaIncC(%x, %x, %x) = %x, want %x", i+1, r.a, r.b, r.x, got, r.want)
		}
	}
}
