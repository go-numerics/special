package special

import "testing"

// TestGammaInversePinned pins inverses where a branch of the inversion moves
// the answer. Each row was found by comparing 150000 random (a, p) against a
// build with the branch forced the other way. Rows 1-2: igamInvBelowSubnormal
// decides, for the upper tail, between 0 and 2^-1074 (the root near 2^-1075);
// rows 3-5: DiDonato and Morris Eq 35 (w < 0.15 (a + 1)) against the
// general start, whose Newton run ends one ulp away or more.
func TestGammaInversePinned(t *testing.T) {
	rows := []struct {
		a, p  float64
		upper bool // GammaQInv instead of GammaPInv
		want  float64
	}{
		{0x1.6d2fed24ed3efp-21, 0x1.09598f2135485p-11, true, 0x1p-1074},
		{0x1.2984952727d22p-15, 0x1.f2aa1eaa95d47p-01, false, 0x1p-1074},
		{0x1.78a6979cfcfffp+03, 0x1.74fb56b2e6c8p-1028, false, 0x1.0fe5d4800668cp-85},
		{0x1.0610b4945205cp+00, 0x1.b5d86af38p-1040, false, 0x1.ca6df99ab4d9cp-1016},
		{0x1.03a99d3fd8994p+03, 0x1.42562cde2bcd2p-447, false, 0x1.d82f865d6cf26p-54},
	}
	for i, r := range rows {
		got := GammaPInv(r.a, r.p)
		if r.upper {
			got = GammaQInv(r.a, r.p)
		}
		if got != r.want {
			t.Errorf("row %d: inverse(%x, %x) = %x, want %x", i+1, r.a, r.p, got, r.want)
		}
	}
}
