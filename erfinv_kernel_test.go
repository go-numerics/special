package special

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
)

// TestErfInvKernelAgainstPrev compares the direct table kernel of ErfInv and
// ErfcInv (erfcinv fitted in q, no logarithm, square root or division) with
// the previous kernel (erfcInvPrev, and erfcInvN for ErfInv): the two agree to
// 1 ulp everywhere and are bit-equal except for a rate of flips (stated in
// the PR and release notes; measured with ERFINV_SWEEP=<points>). The
// default sweep's output hash must be the same on every architecture.
const erfInvSweepHash = "35a535eadae7707240e615f2891d8430c4316e0b3697ec30ee6c12afb2138bc1"

func TestErfInvKernelAgainstPrev(t *testing.T) {
	n := 20000
	if s := os.Getenv("ERFINV_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	flips, total, worst := 0, 0, 0.0
	h := sha256.New()
	dump := func(name string) *os.File {
		if p := os.Getenv("FLIPS_FILE"); p != "" {
			f, _ := os.Create(p + "." + name)
			return f
		}
		return nil
	}
	fi, fc := dump("erfinv"), dump("erfcinv")
	cmp := func(w *os.File, x, a, b float64) {
		binary.Write(h, binary.LittleEndian, a)
		total++
		if a != b {
			flips++
			if w != nil {
				fmt.Fprintln(w, x, a, b)
			}
			if u := math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b); u > worst {
				worst = u
			}
		}
	}
	check := func(x float64) { // x in (0, 1)
		// ErfInv(1 - x) and ErfInv(-(1 - x)), the old path: erfcinv(x) by the log kernel
		if p := 1 - x; p > 0.5 && p < 1 {
			hh, ll := erfcInvN(1 - p) // 1 - p exact (p need not be 1 - x)
			cmp(fi, p, ErfInv(p), hh+ll)
			if ErfInv(-p) != -ErfInv(p) {
				t.Fatalf("odd symmetry at %v", p)
			}
		}
		cmp(fc, x, ErfcInv(x), erfcInvPrev(x))
		cmp(fc, 2-x, ErfcInv(2-x), erfcInvPrev(2-x))
	}
	s := uint64(0x9e3779b97f4a7c15)
	rnd := func() float64 {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return math.Ldexp(float64(s>>11), -53)
	}
	for i := 0; i < n; i++ {
		check(rnd())
		check(math.Ldexp(0.5+math.Ldexp(rnd(), -1), -1-int(math.Ldexp(rnd(), 6)*0.9375))) // log-uniform down to 2^-60
	}
	// the piece edges and their neighbours
	for i := range erfcInvDirPieces {
		for _, x := range []float64{erfcInvDirPieces[i].lo, erfcInvDirPieces[i].hi, erfcInvDirPieces[i].c} {
			for _, v := range []float64{math.Nextafter(x, 0), x, math.Nextafter(x, 1)} {
				if v > 0 && v < 0.5 {
					cmp(fc, v, ErfcInv(v), erfcInvPrev(v))
				}
			}
		}
	}
	t.Logf("%d points: %d flips (%.3g), worst %.2f ulp", total, flips, float64(flips)/float64(total), worst)
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != erfInvSweepHash {
		t.Fatalf("ErfInv/ErfcInv sweep hash %s, want %s", got, erfInvSweepHash)
	}
	if worst > 1 {
		t.Fatalf("differs from the previous kernel by %v ulp", worst)
	}
}

// TestErfcInvDirTable: the table is what the kernel assumes (degree <= 11,
// so that q[11..] are zero and erfcInvDirHL, which reads q[0..10], drops
// nothing), and its edges are the octave pieces of q.
func TestErfcInvDirTable(t *testing.T) {
	if len(erfcInvDirPieces) != erfcInvDirN {
		t.Fatalf("%d pieces, want %d", len(erfcInvDirPieces), erfcInvDirN)
	}
	for i := range erfcInvDirEst {
		for k := 11; k < 16; k++ {
			if erfcInvDirEst[i].q[k] != 0 {
				t.Fatalf("piece %d has a nonzero q[%d]", i, k)
			}
		}
		lo := erfcInvDirPieces[i].lo
		if got := math.Float64bits(lo)>>49 - erfcInvDirBase; got != uint64(i) {
			t.Fatalf("piece %d starts at %v, whose index is %d", i, lo, got)
		}
	}
	if ErfcInv(0.5-0x1p-54) <= 0 || ErfInv(1-0x1p-53) < 5.86 || ErfInv(1-0x1p-53) > 5.87 {
		t.Fatalf("ends: %v %v", ErfcInv(0.5-0x1p-54), ErfInv(1-0x1p-53))
	}
}

// TestErfcInvPrevSpecials: the kept previous kernel agrees with ErfcInv on the
// special values.
func TestErfcInvPrevSpecials(t *testing.T) {
	for _, q := range []float64{math.NaN(), -1, 3, 0, 2, 1, 1.5, 0.75, 0.5, 0x1p-1074} {
		a, b := ErfcInv(q), erfcInvPrev(q)
		if a != b && !(math.IsNaN(a) && math.IsNaN(b)) && math.Abs(a-b) > 1e-15*math.Abs(b) {
			t.Errorf("ErfcInv(%v) = %v, previous %v", q, a, b)
		}
	}
}
