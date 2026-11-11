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

	"github.com/go-numerics/special/internal/libm"
)

// TestLogBetaTableAgainstPrev compares LogBeta, whose direct combination of
// three log-gammas reads the fitted table for a + b < 2048 (lgammaOctHL),
// with the previous evaluation (the same combination from the Stirling
// series, logGammaLargeHL2): they agree to 1 ulp and are bit-equal except for
// a rate of flips (stated in the PR and release notes; LOGBETA_SWEEP=<points>,
// FLIPS_FILE=<path> writes "a b new old" lines for tools/ulpflips). The
// default sweep's hash must be the same on every architecture.
const logBetaSweepHash = "cae314777ca4e6635f530f9590da1fe11974ce465a35c2dacfe707acdd1d6136"

func TestLogBetaTableAgainstPrev(t *testing.T) {
	n := 20000
	if s := os.Getenv("LOGBETA_SWEEP"); s != "" {
		n, _ = strconv.Atoi(s)
	}
	var fw *os.File
	if p := os.Getenv("FLIPS_FILE"); p != "" {
		fw, _ = os.Create(p)
	}
	flips, total, worst := 0, 0, 0.0
	h := sha256.New()
	check := func(a, b float64) {
		if a > b {
			a, b = b, a
		}
		if b > 64*a {
			return // not the direct combination
		}
		got := LogBeta(a, b)
		s := a + b
		bb := s - a
		sl := (a - (s - bb)) + (b - bb)
		hi, lo := logBetaDirect(a, b, s, sl)
		want := hi + lo
		binary.Write(h, binary.LittleEndian, got)
		total++
		if got != want {
			flips++
			if fw != nil {
				fmt.Fprintln(fw, a, b, got, want)
			}
			if u := math.Abs(got-want) / math.Abs(math.Nextafter(want, math.Inf(1))-want); u > worst {
				worst = u
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
		a := float64(10 * libm.Exp(float64(7*math.Ln2*rnd()))) // log-uniform on [10, 1280)
		check(a, float64(a*float64(1+float64(63*rnd()))))      // b <= 64a
		check(a, a)                                            // the refbench shape
		check(a, a+float64(10*rnd()))                          // a + b near the table's upper end for the larger a
		check(float64(1000*rnd())+10, float64(1000*rnd())+10)
	}
	t.Logf("%d points: %d flips (%.3g), worst %.2f ulp", total, flips, float64(flips)/float64(total), worst)
	if got := hex.EncodeToString(h.Sum(nil)); n == 20000 && got != logBetaSweepHash {
		t.Fatalf("LogBeta sweep hash %s, want %s", got, logBetaSweepHash)
	}
	if worst > 1 {
		t.Fatalf("differs from the previous evaluation by %v ulp", worst)
	}
}
