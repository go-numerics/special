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

// TestLogBetaTableAgainstOracle checks LogBeta where it reads the fitted table
// of log Γ (10 <= a <= b <= 64a, a + b < 2048; every point of the sweep is on
// that branch) against the double-double reference logBetaPos: the stated
// bound is 0.5 ulp plus 0.005 for the table's 2^-57 absolute error (measured
// 0.5004 on 1.2e6 points; a table that takes the low part of a + b through
// the slope a1 alone measures 0.5078). It also
// compares with the previous evaluation (the same combination from the Stirling
// series): at most 1 ulp apart, and a flip-rate ceiling (measured 1.8e-5 for
// the first draw of this sweep; stated in the PR and release notes;
// LOGBETA_SWEEP=<points>, FLIPS_FILE=<path> writes "a b new old" lines for
// tools/ulpflips). The default sweep's hash must be the same on every
// architecture; the bound is what catches a wrong table, not the hash.
const (
	logBetaSweepHash = "fd49b376d5751c6fdc174d4cfa110946a3d8ea5fb773712d94ee21ede89af1c8"
	logBetaBoundUlp  = 0.505
	logBetaFlipCeil  = 2e-4
)

func TestLogBetaTableAgainstOracle(t *testing.T) {
	n := 20000
	if s := os.Getenv("LOGBETA_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	var fw *os.File
	if p := os.Getenv("FLIPS_FILE"); p != "" {
		fw, _ = os.Create(p)
	}
	flips, total, worst, worstPrev := 0, 0, 0.0, 0.0
	var wa, wb float64
	h := sha256.New()
	check := func(a, b float64) {
		if a > b {
			a, b = b, a
		}
		if a < 10 || b > 64*a || float64(a+b) >= 2048 {
			t.Fatalf("(%v, %v) is not on the table branch", a, b)
		}
		got := LogBeta(a, b)
		s := a + b
		bb := s - a
		sl := (a - (s - bb)) + (b - bb)
		hi, lo := logBetaDirect(a, b, s, sl)
		prev := hi + lo
		ref := logBetaPos(a, b)
		ulp := math.Abs(math.Nextafter(ref.Hi, math.Inf(1)) - ref.Hi)
		if u := math.Abs((got-ref.Hi)-ref.Lo) / ulp; u > worst {
			worst, wa, wb = u, a, b
		}
		binary.Write(h, binary.LittleEndian, got)
		total++
		if got != prev {
			flips++
			if fw != nil {
				fmt.Fprintln(fw, a, b, got, prev)
			}
			if u := math.Abs(got-prev) / math.Abs(math.Nextafter(prev, math.Inf(1))-prev); u > worstPrev {
				worstPrev = u
			}
		}
		if got != LogBeta(b, a) {
			t.Fatalf("LogBeta(%v, %v) is not symmetric", a, b)
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
		a := float64(10 + float64(990*rnd())) // [10, 1000)
		top := min(float64(64*a), 2047.5-a)   // b <= 64a and a + b < 2048
		check(a, float64(a+float64(rnd()*(top-a))))
		c := float64(10 + float64(1013*rnd())) // the refbench shape a = b
		check(c, c)
		d := float64(1024 + float64(1000*rnd())) // a + b about d in [1024, 2024)
		e := float64(d * float64(0.3+float64(0.2*rnd())))
		check(e, d-e)
	}
	t.Logf("%d points: %d flips vs the previous evaluation (%.3g), worst %.2f ulp from it; worst %.4f ulp from the oracle at (%v, %v)",
		total, flips, float64(flips)/float64(total), worstPrev, worst, wa, wb)
	if worst > logBetaBoundUlp {
		t.Fatalf("worst error %.4f ulp against the double-double reference at (%v, %v), bound %v", worst, wa, wb, logBetaBoundUlp)
	}
	if worstPrev > 1 {
		t.Fatalf("differs from the previous evaluation by %v ulp", worstPrev)
	}
	if rate := float64(flips) / float64(total); rate > logBetaFlipCeil {
		t.Fatalf("flip rate %.3g against the previous evaluation exceeds %.3g", rate, logBetaFlipCeil)
	}
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != logBetaSweepHash {
		t.Fatalf("LogBeta sweep hash %s, want %s", got, logBetaSweepHash)
	}
}
