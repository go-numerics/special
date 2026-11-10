package special

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"math"
	"os"
	"strconv"
	"testing"
)

// TestNdtriKernelAgainstSlow compares the direct table kernel with the
// previous implementation (ndtriSlow, erfcinv based, kept for p < 2^-28):
// the two agree to 1 ulp everywhere, are bit-equal except for a rate of
// flips (stated in the PR and release notes; measured with
// NDTRI_SWEEP=<points>), and the odd symmetry is exact.
//
// Big sweep (NDTRI_SWEEP=2000000, 4e6 points: half uniform p on (0,1), half
// log-uniform p on [2^-30, 1]): 2345 flips (5.9e-4), the largest 1 ulp.
// The default sweep's output hash must be the same on every architecture.

const ndtriSweepHash = "c7275040c2cf12c626752578cc58e47f9ceaec0a1baff032cc788bc081041b92"

func TestNdtriKernelAgainstSlow(t *testing.T) {
	n := 20000
	if s := os.Getenv("NDTRI_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	flips, worst := 0, 0.0
	h := sha256.New()
	check := func(p float64) {
		a, b := Ndtri(p), ndtriSlow(p)
		binary.Write(h, binary.LittleEndian, a)
		if a != b {
			flips++
			if u := math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b); u > worst {
				worst = u
			}
		}
		if q := 1 - p; q != 1-q && 1-q == p && Ndtri(q) != -a {
			t.Fatalf("odd symmetry: Ndtri(%v) = %v, Ndtri(%v) = %v", p, a, q, Ndtri(q))
		}
	}
	x := uint64(0x9e3779b97f4a7c15)
	rnd := func() float64 {
		x ^= x << 13
		x ^= x >> 7
		x ^= x << 17
		return math.Ldexp(float64(x>>11), -53)
	}
	for i := 0; i < n; i++ {
		check(rnd())
		check(math.Ldexp(0.5+math.Ldexp(rnd(), -1), -1-int(math.Ldexp(rnd(), 5)*0.9375)))
	}
	t.Logf("%d points: %d flips (%.3g), worst %.2f ulp", 2*n, flips, float64(flips)/float64(2*n), worst)
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != ndtriSweepHash {
		// the same bits on every architecture (the 8 CI lanes run this)
		t.Fatalf("Ndtri sweep hash %s, want %s", got, ndtriSweepHash)
	}
	if worst > 1 {
		t.Fatalf("differs from the previous kernel by %v ulp", worst)
	}
	if Ndtri(0.5) != 0 || math.Signbit(Ndtri(0.5)) {
		t.Fatalf("Ndtri(1/2) = %v", Ndtri(0.5))
	}
}
