package nd

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
)

// The speed comparison against SciPy (docs/perf.md). It runs only with
// -perf.report=PATH, needs uv (https://docs.astral.sh/uv/) on PATH or at
// ~/.local/bin/uv, and interleaves the two sides round by round so that a
// change in the load of a shared host affects both alike; medians are
// reported.

var (
	perfReport = flag.String("perf.report", "", "write the Go vs SciPy speed table (markdown) to this file")
	perfN      = flag.Int("perf.n", 1_000_000, "points per call")
	perfRounds = flag.Int("perf.rounds", 7, "interleaved rounds per function")
	perfOnly   = flag.String("perf.only", "", "comma-separated subset of functions")
)

type perfCase struct {
	name   string
	ranges [][2]float64
	scalar func(v []float64) float64
	nd     func(a []*ndarray.Array) (*ndarray.Array, error)
}

func u1(f func(float64) float64, g func(*ndarray.Array) (*ndarray.Array, error), lo, hi float64, name string) perfCase {
	return perfCase{name, [][2]float64{{lo, hi}}, func(v []float64) float64 { return f(v[0]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return g(a[0]) }}
}

func u2(f func(x, y float64) float64, g func(a, b *ndarray.Array) (*ndarray.Array, error), r1, r2 [2]float64, name string) perfCase {
	return perfCase{name, [][2]float64{r1, r2}, func(v []float64) float64 { return f(v[0], v[1]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return g(a[0], a[1]) }}
}

// perfCases must match CASES in bench/scipy_server.py.
var perfCases = []perfCase{
	u1(special.Gamma, Gamma, 0.1, 170, "Gamma"),
	u1(special.LogGamma, LogGamma, 0.1, 1000, "LogGamma"),
	u1(special.Digamma, Digamma, 0.1, 100, "Digamma"),
	u2(special.Beta, Beta, [2]float64{0.1, 50}, [2]float64{0.1, 50}, "Beta"),
	u2(special.LogBeta, LogBeta, [2]float64{0.1, 1000}, [2]float64{0.1, 1000}, "LogBeta"),
	u1(special.Erf, Erf, -5, 5, "Erf"),
	u1(special.Erfc, Erfc, -5, 25, "Erfc"),
	u1(special.Erfcx, Erfcx, -5, 50, "Erfcx"),
	u1(special.ErfInv, ErfInv, -0.999, 0.999, "ErfInv"),
	u1(special.ErfcInv, ErfcInv, 1e-10, 1.999, "ErfcInv"),
	u1(special.Ndtr, Ndtr, -30, 10, "Ndtr"),
	u1(special.Ndtri, Ndtri, 1e-12, 0.999999, "Ndtri"),
	u1(special.LogNdtr, LogNdtr, -100, 10, "LogNdtr"),
	u1(special.Expit, Expit, -50, 50, "Expit"),
	u1(special.Logit, Logit, 1e-6, 0.999999, "Logit"),
	u1(special.Log1pExp, Log1pExp, -50, 50, "Log1pExp"),
	u2(special.Xlogy, Xlogy, [2]float64{0.1, 10}, [2]float64{0.1, 10}, "Xlogy"),
	u1(special.Zeta, Zeta, 1.01, 60, "Zeta"),
	u2(special.HurwitzZeta, HurwitzZeta, [2]float64{1.5, 20}, [2]float64{0.1, 20}, "HurwitzZeta"),
	u1(func(x float64) float64 { return special.Polygamma(1, x) },
		func(a *ndarray.Array) (*ndarray.Array, error) {
			return Polygamma(ndOne, a)
		}, 0.1, 100, "Polygamma1"),
	u2(special.GammaP, GammaP, [2]float64{0.5, 50}, [2]float64{0.1, 60}, "GammaP"),
	u2(special.GammaPInv, GammaPInv, [2]float64{0.5, 50}, [2]float64{1e-6, 0.999999}, "GammaPInv"),
	{"BetaInc", [][2]float64{{0.5, 50}, {0.5, 50}, {0.01, 0.99}},
		func(v []float64) float64 { return special.BetaInc(v[0], v[1], v[2]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return BetaInc(a[0], a[1], a[2]) }},
	u2(special.GammaQ, GammaQ, [2]float64{0.5, 50}, [2]float64{0.1, 60}, "GammaQ"),
	u2(special.GammaQInv, GammaQInv, [2]float64{0.5, 50}, [2]float64{1e-6, 0.999999}, "GammaQInv"),
	{"BetaIncC", [][2]float64{{0.5, 50}, {0.5, 50}, {0.01, 0.99}},
		func(v []float64) float64 { return special.BetaIncC(v[0], v[1], v[2]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return BetaIncC(a[0], a[1], a[2]) }},
	{"BetaIncInv", [][2]float64{{0.5, 50}, {0.5, 50}, {0.01, 0.99}},
		func(v []float64) float64 { return special.BetaIncInv(v[0], v[1], v[2]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return BetaIncInv(a[0], a[1], a[2]) }},
	{"BetaIncCInv", [][2]float64{{0.5, 50}, {0.5, 50}, {0.01, 0.99}},
		func(v []float64) float64 { return special.BetaIncCInv(v[0], v[1], v[2]) },
		func(a []*ndarray.Array) (*ndarray.Array, error) { return BetaIncCInv(a[0], a[1], a[2]) }},
	u1(special.J0, J0, 0, 100, "J0"),
	u1(special.Y0, Y0, 0.1, 100, "Y0"),
	u1(special.I0, I0, 0, 50, "I0"),
	u1(special.K0, K0, 0.01, 50, "K0"),
	u2(special.Jv, Jv, [2]float64{0, 20}, [2]float64{0.1, 50}, "Jv"),
	u2(special.Yv, Yv, [2]float64{0, 20}, [2]float64{0.1, 50}, "Yv"),
	u2(special.Kv, Kv, [2]float64{0, 20}, [2]float64{0.1, 50}, "Kv"),
	u1(special.EllipK, EllipK, -10, 0.999, "EllipK"),
	u1(special.EllipE, EllipE, -10, 0.999, "EllipE"),
}

const gold = 0.6180339887498949

// ndOne is the order n = 1 of Polygamma1, a single-element array.
var ndOne, _ = ndarray.FromData([]float64{1}, 1)

// perfPoints reproduces bench/scipy_server.py's points().
func perfPoints(n int, lo, hi float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		f := math.Mod(float64(i)*gold, 1)
		x[i] = lo + (hi-lo)*f
	}
	return x
}

func median(v []float64) float64 {
	s := append([]float64(nil), v...)
	sort.Float64s(s)
	return s[len(s)/2]
}

func uvPath() string {
	if p, err := exec.LookPath("uv"); err == nil {
		return p
	}
	home, _ := os.UserHomeDir()
	return home + "/.local/bin/uv"
}

func TestPerfReport(t *testing.T) {
	if *perfReport == "" {
		t.Skip("set -perf.report=PATH to run the SciPy speed comparison")
	}
	cmd := exec.Command(uvPath(), "run", "--with", "scipy==1.18.1", "--with", "numpy==2.5.4",
		"python", "../bench/scipy_server.py")
	cmd.Env = append(os.Environ(), "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1")
	in, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	outp, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr
	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		fmt.Fprintln(in, "quit")
		cmd.Wait()
	}()
	rd := bufio.NewScanner(outp)
	scipy := func(name string, n int) float64 {
		fmt.Fprintf(in, "%s %d\n", name, n)
		if !rd.Scan() {
			t.Fatalf("scipy server: %v", rd.Err())
		}
		v, err := strconv.ParseFloat(strings.TrimSpace(rd.Text()), 64)
		if err != nil {
			t.Fatalf("scipy server said %q", rd.Text())
		}
		return v
	}
	only := map[string]bool{}
	for _, s := range strings.Split(*perfOnly, ",") {
		if s != "" {
			only[s] = true
		}
	}
	n := *perfN
	load, _ := exec.Command("uptime").Output()
	var b strings.Builder
	fmt.Fprintf(&b, "<!-- generated by: cd nd && go test -run TestPerfReport -perf.report=../docs/perf-table.md -->\n\n")
	fmt.Fprintf(&b, "%d points per call, median of %d interleaved rounds; %s/%s, GOMAXPROCS=%d; host load at start: %s\n\n",
		n, *perfRounds, runtime.GOOS, runtime.GOARCH, runtime.GOMAXPROCS(0), strings.TrimSpace(string(load)))
	b.WriteString("| function | SciPy (ns/elem) | Go scalar loop (ns/elem) | Go nd 1 thread (ns/elem) | Go nd parallel (ns/elem) | scalar speed-up | nd 1-thread speed-up | parallel speed-up |\n")
	b.WriteString("|---|--:|--:|--:|--:|--:|--:|--:|\n")
	for _, c := range perfCases {
		if len(only) > 0 && !only[c.name] {
			continue
		}
		args := make([][]float64, len(c.ranges))
		arrs := make([]*ndarray.Array, len(c.ranges))
		for i, r := range c.ranges {
			args[i] = perfPoints(n, r[0], r[1])
			arrs[i], _ = ndarray.FromData(args[i], n)
		}
		out := make([]float64, n)
		v := make([]float64, len(args))
		var ts, tg, tp, t1 []float64
		scipy(c.name, n) // warm-up (and point generation)
		for round := 0; round < *perfRounds; round++ {
			ts = append(ts, scipy(c.name, n))
			t0 := time.Now()
			for i := range out {
				for k := range args {
					v[k] = args[k][i]
				}
				out[i] = c.scalar(v)
			}
			tg = append(tg, time.Since(t0).Seconds())
			t0 = time.Now()
			if _, err := c.nd(arrs); err != nil {
				t.Fatal(err)
			}
			tp = append(tp, time.Since(t0).Seconds())
			// The nd function on one thread: the Go counterpart of a SciPy
			// ufunc call (one loop over the array, no call per element
			// through a function value).
			procs := runtime.GOMAXPROCS(1)
			t0 = time.Now()
			if _, err := c.nd(arrs); err != nil {
				t.Fatal(err)
			}
			t1 = append(t1, time.Since(t0).Seconds())
			runtime.GOMAXPROCS(procs)
		}
		s, g, p, o := median(ts), median(tg), median(tp), median(t1)
		ns := func(x float64) float64 { return x / float64(n) * 1e9 }
		fmt.Fprintf(&b, "| %s | %.1f | %.1f | %.1f | %.1f | %.2fx | %.2fx | %.2fx |\n", c.name, ns(s), ns(g), ns(o), ns(p), s/g, s/o, s/p)
		t.Logf("%s: scipy %.1f, go %.1f, nd1 %.1f, nd %.1f ns/elem", c.name, ns(s), ns(g), ns(o), ns(p))
	}
	if err := os.WriteFile(*perfReport, []byte(b.String()), 0o644); err != nil {
		t.Fatal(err)
	}
}
