package special

import "testing"

// TestIncBetaInversePinned pins inverses where the choice made by a branch of
// the inversion moves the last bit. Each row was found by comparing 150000
// random (a, b, p) against a build with the branch forced the other way: the
// answer differs by one ulp. The expected bits are those of the double-double
// path the branch selects; the monotone bracket test cannot see one ulp.
//
//	ibetaInv   a < 1e-3 || b < 1e-3  (rows 1-3: the side decided at 1/2)
//	ibetaInvSide  !sure && g >= 0 && u <= 0.5  (rows 4-6: root side marked sure)
//	ibetaInvGuess lu > log(1/2)  (rows 7-9: guess clamped to 1/2)
func TestIncBetaInversePinned(t *testing.T) {
	rows := []struct {
		a, b, p float64
		upper   bool // BetaIncCInv (q = p) instead of BetaIncInv
		want    float64
	}{
		{0x1.4827252736f46p-11, 0x1.cd25ad49bb5eap+04, 0x1.72bb63f3cad6cp-27, true, 0x1.0ee285e448c81p-02},
		{0x1.f1a4545f8bf25p+02, 0x1.05638f439cae4p-16, 0x1.2b2d7ca61eac4p-38, false, 0x1.751e7624ca1dp-03},
		{0x1.9facfe063527ap-19, 0x1.a7e527ac697fep+07, 0x1.4d28dd1b7d931p-40, true, 0x1.cb5a0ab5f8e47p-05},
		{0x1.5a4dd092fbd2ap-08, 0x1.18d5ec0716179p+17, 0x1.b40d102b449bbp-29, true, 0x1.582423bf32be4p-14},
		{0x1.165f292b6a2f5p-07, 0x1.64ac90d9418f4p+15, 0x1.fffff0b89ce6ap-01, false, 0x1.61e0afd8e574dp-13},
		{0x1.fe638be7adbbfp-01, 0x1.9bc6c81461205p+17, 0x1.106b2ac81e649p-16, true, 0x1.b666ae5a12f8p-15},
		{0x1.05f4fe5143172p+00, 0x1.44ec5809987d7p-12, 0x1.ffe5d2ea6ba2cp-01, true, 0x1.f31c4eaf1e6aap-02},
		{0x1.ada44c44d9d3ep-14, 0x1.0c76d4955b884p+00, 0x1.053a4c74bdd75p-14, true, 0x1.09b084463b399p-01},
		{0x1.1debe90e81cf9p+02, 0x1.f161f9125abb2p-14, 0x1.ffffd48b0f1acp-01, true, 0x1.d3e8f13c8ab1bp-02},
	}
	for i, r := range rows {
		got := BetaIncInv(r.a, r.b, r.p)
		if r.upper {
			got = BetaIncCInv(r.a, r.b, r.p)
		}
		if got != r.want {
			t.Errorf("row %d: inverse(%x, %x, %x) = %x, want %x", i+1, r.a, r.b, r.p, got, r.want)
		}
	}
}
