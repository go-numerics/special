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

// logitPrev is Logit as of v0.9.0: p/(1-p) as hi + lo through its FMA
// residual and log of it from the log kernel (log1p of (2p-1)/(1-p) in the
// middle).
func logitPrev(p float64) float64 {
	if !(p > 0 && p < 1) {
		switch {
		case p == 0:
			return math.Inf(-1)
		case p == 1:
			return math.Inf(1)
		}
		return math.NaN()
	}
	qh := 1 - p
	bb := qh - 1
	ql := (1 - (qh - bb)) + (-p - bb) // 1 - p = qh + ql exactly
	if p < 0.25 || p > 0.75 {
		y := p / qh
		r := (math.FMA(-y, qh, p) - float64(y*ql)) / qh
		lh, ll := logHL(y)
		return lh + (ll + r/y)
	}
	num := float64(2*p) - 1
	th := num / qh
	tl := (math.FMA(-th, qh, num) - float64(th*ql)) / qh
	return log1pHLPrev(th, tl)
}

func log1pHLPrev(th, tl float64) float64 {
	if math.Abs(th) < 0x1p-20 {
		s := float64(th * th * math.FMA(th, math.FMA(th, -0.25, 1.0/3), -0.5))
		return th + (math.FMA(tl, -th, tl) + s)
	}
	wh := 1 + th
	bb := wh - 1
	wl := (1 - (wh - bb)) + (th - bb) + tl
	lh, ll := logHLn(wh)
	return lh + (ll + wl/wh)
}

// TestLogitKernelAgainstPrev compares the direct table kernel with the
// previous one: they agree to 1 ulp, are bit-equal except for a rate of flips
// (stated in the PR and release notes; LOGIT_SWEEP=<points>,
// FLIPS_FILE=<path> writes "p new old" lines for tools/ulpflips), and the
// odd symmetry about 1/2 is exact. The default sweep's hash must be the same
// on every architecture.
const logitSweepHash = "5bdf002873a16043e8e4dfaa32deb9bffb81a3c0df62273c64e3d09ae760e5e8"

func TestLogitKernelAgainstPrev(t *testing.T) {
	n := 20000
	if s := os.Getenv("LOGIT_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	var fw *os.File
	if p := os.Getenv("FLIPS_FILE"); p != "" {
		fw, _ = os.Create(p)
	}
	flips, total, worst := 0, 0, 0.0
	h := sha256.New()
	check := func(p float64) {
		a, b := Logit(p), logitPrev(p)
		binary.Write(h, binary.LittleEndian, a)
		total++
		if a != b {
			flips++
			if fw != nil {
				fmt.Fprintln(fw, p, a, b)
			}
			if u := math.Abs(a-b) / math.Abs(math.Nextafter(b, math.Inf(1))-b); u > worst {
				worst = u
			}
		}
		if q := 1 - p; q != 1-q && 1-q == p && Logit(q) != -a {
			t.Fatalf("odd symmetry: Logit(%v) = %v, Logit(%v) = %v", p, a, q, Logit(q))
		}
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
		check(0.5 + math.Ldexp(rnd()-0.5, -1-int(math.Ldexp(rnd(), 5))))                  // around 1/2
	}
	// the piece edges
	for i := range logitPieces {
		for _, x := range []float64{logitPieces[i].lo, logitPieces[i].hi} {
			for _, v := range []float64{math.Nextafter(x, 0), x, math.Nextafter(x, 1)} {
				check(v)
			}
		}
	}
	t.Logf("%d points: %d flips (%.3g), worst %.2f ulp", total, flips, float64(flips)/float64(total), worst)
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != logitSweepHash {
		t.Fatalf("Logit sweep hash %s, want %s", got, logitSweepHash)
	}
	if worst > 1 {
		t.Fatalf("differs from the previous kernel by %v ulp", worst)
	}
	if v := Logit(0.5); v != 0 || math.Signbit(v) {
		t.Fatalf("Logit(1/2) = %v", v)
	}
}

// TestLogitTable: the table is what the kernel assumes (degree <= 11 is not
// required: all 16 q are read), and its pieces are the octave pieces of q.
func TestLogitTable(t *testing.T) {
	if len(logitPieces) != ndtriN-8 {
		t.Fatalf("%d pieces", len(logitPieces))
	}
	for i := range logitPieces {
		if got := math.Float64bits(logitPieces[i].lo)>>49 - ndtriBase; got != uint64(i) {
			t.Fatalf("piece %d starts at %v, whose index is %d", i, logitPieces[i].lo, got)
		}
	}
}
