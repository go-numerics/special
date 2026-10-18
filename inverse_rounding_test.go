package special

import (
	"bufio"
	"embed"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"testing"
)

// The correct-rounding check of the incomplete beta and gamma inverses:
// points a, b log-uniform in [e^-1, e^6], p uniform in [0.01, 0.99], with
// the exact root from mpmath (tools/rootcheck). The error is that of the
// root, in ulps of the root; a correctly rounded root is within 0.5.

//go:embed testdata/roots/*.tsv
var rootsFS embed.FS

// rootBudget is the largest root error allowed, and rootAbove the number
// of points allowed above 0.5 ulp, per function: the points where the
// forward function cannot settle the rounding (its error bound exceeds
// the half-gap in p between the two floats around the root; see
// ibetaInvRound and igamInvRound).
var rootBudget = map[string]struct {
	max   float64
	above int
}{
	"betaincinv":   {0.5, 0},
	"betainccinv":  {0.5, 0},
	"gammaincinv":  {0.5, 0},
	"gammainccinv": {0.5, 0},
}

var rootFns = map[string]func(args []float64) float64{
	"betaincinv":   func(a []float64) float64 { return BetaIncInv(a[0], a[1], a[2]) },
	"betainccinv":  func(a []float64) float64 { return BetaIncCInv(a[0], a[1], a[2]) },
	"gammaincinv":  func(a []float64) float64 { return GammaPInv(a[0], a[1]) },
	"gammainccinv": func(a []float64) float64 { return GammaQInv(a[0], a[1]) },
}

func loadRoots(tb testing.TB, name string) [][]float64 {
	tb.Helper()
	f, err := rootsFS.Open("testdata/roots/" + name + ".tsv")
	if err != nil {
		tb.Fatal(err)
	}
	defer f.Close()
	var rows [][]float64
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if line == "" || line[0] == '#' {
			continue
		}
		var row []float64
		for _, c := range strings.Split(line, "\t") {
			v, err := strconv.ParseFloat(c, 64)
			if err != nil {
				tb.Fatalf("%s: %q: %v", name, line, err)
			}
			row = append(row, v)
		}
		rows = append(rows, row)
	}
	if err := sc.Err(); err != nil {
		tb.Fatal(err)
	}
	return rows
}

// TestInverseRootRounding measures the root error of the inverses and
// holds it to rootBudget.
func TestInverseRootRounding(t *testing.T) {
	names := make([]string, 0, len(rootFns))
	for k := range rootFns {
		names = append(names, k)
	}
	sort.Strings(names)
	for _, name := range names {
		fn := rootFns[name]
		rows := loadRoots(t, name)
		var errs []float64
		above := 0
		var worst []string
		for _, r := range rows {
			n := len(r)
			e := ulpErr(fn(r[:n-2]), r[n-2], r[n-1])
			errs = append(errs, e)
			if e > 0.5 {
				above++
				worst = append(worst, fmt.Sprintf("%v: %.3f ulp", r[:n-2], e))
			}
		}
		sort.Float64s(errs)
		mx := errs[len(errs)-1]
		t.Logf("%-13s n=%d max %.3f ulp median %.3f above 0.5: %d", name, len(errs), mx, errs[len(errs)/2], above)
		b := rootBudget[name]
		if mx > b.max+1e-9 || above > b.above {
			t.Errorf("%s: max %.3f ulp (budget %.2f), %d above 0.5 ulp (budget %d): %v", name, mx, b.max, above, b.above, worst)
		}
	}
}
