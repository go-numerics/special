package bench

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"flag"
	"fmt"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"testing"

	"github.com/go-ndarray/ndarray"
	"github.com/go-numerics/special"
	"github.com/go-numerics/special/nd"
	"github.com/go-scikit/refbench"
	"github.com/go-scikit/refbench/rbtest"
)

// The speed comparison with SciPy (docs/perf.md).
//
//	cd bench
//	go test -run TestPerf .                          CI mode: allocations, table, denominator
//	go test -run TestPerf -cpu 1 -perf.run .         measure (refuses at a 1-minute load of 8 or more)
//	go test -run TestPerf -perf.record .             allocation counts only (a deferred report)
//	go test -run TestDataParity -perf.parity .       both sides get the same bytes (needs uv)
//
// What is compared. Every row times one call over an array of points, on both
// sides: the Go body loops over n points calling the scalar function (the
// "scalar loop" rows) or calls the nd array function (the "nd/" rows, run on one
// goroutine because refbench sets GOMAXPROCS=1); the SciPy body is one ufunc
// call over the same n points, result array allocated by the call, as the Go
// side allocates its own. Per-call overhead of the ufunc is amortised over n
// points (about 1 µs against n·10 ns and up), so the ratio is per element.
// Arrays are n = 10 000 points (80 kB) so that both sides run from cache and a
// CI allocation count (51 calls of the slowest function) stays short.

const n = 10_000

const gold = 0.6180339887498949

// pts reproduces the Python setup: lo + (hi-lo)·frac(i·gold). The explicit
// float64 conversion forbids a fused multiply-add, so the bytes are the same
// on arm64 (TestDataParity compares SHA-256 digests with NumPy's).
func pts(lo, hi float64) []float64 {
	x := make([]float64, n)
	for i := range x {
		f := math.Mod(float64(i)*gold, 1)
		x[i] = lo + float64((hi-lo)*f)
	}
	return x
}

// setup is the Python expression that builds the same arguments.
func setup(r [][2]float64) string {
	s := "["
	for i, p := range r {
		if i > 0 {
			s += ", "
		}
		s += fmt.Sprintf("%v + (%v - %v) * np.mod(np.arange(n, dtype=np.float64) * 0.6180339887498949, 1.0)", p[0], p[1], p[0])
	}
	return s + "]"
}

var sink float64

type spec struct {
	name   string // the public function (and the case name)
	ref    string // the SciPy callable
	ranges [][2]float64
	scalar func(a [][]float64, out []float64) // one loop over the points
	nd     func(a []*ndarray.Array) error     // the nd array function; nil: none
}

func l1(f func(float64) float64) func([][]float64, []float64) {
	return func(a [][]float64, out []float64) {
		x := a[0]
		for i := range out {
			out[i] = f(x[i])
		}
	}
}

func l2(f func(x, y float64) float64) func([][]float64, []float64) {
	return func(a [][]float64, out []float64) {
		x, y := a[0], a[1]
		for i := range out {
			out[i] = f(x[i], y[i])
		}
	}
}

func l3(f func(x, y, z float64) float64) func([][]float64, []float64) {
	return func(a [][]float64, out []float64) {
		x, y, z := a[0], a[1], a[2]
		for i := range out {
			out[i] = f(x[i], y[i], z[i])
		}
	}
}

func n1(g func(*ndarray.Array) (*ndarray.Array, error)) func([]*ndarray.Array) error {
	return func(a []*ndarray.Array) error { _, err := g(a[0]); return err }
}

func n2(g func(a, b *ndarray.Array) (*ndarray.Array, error)) func([]*ndarray.Array) error {
	return func(a []*ndarray.Array) error { _, err := g(a[0], a[1]); return err }
}

func n3(g func(a, b, c *ndarray.Array) (*ndarray.Array, error)) func([]*ndarray.Array) error {
	return func(a []*ndarray.Array) error { _, err := g(a[0], a[1], a[2]); return err }
}

func r(lo, hi float64) [2]float64 { return [2]float64{lo, hi} }

func u1(name, ref string, lo, hi float64, f func(float64) float64, g func(*ndarray.Array) (*ndarray.Array, error)) spec {
	s := spec{name: name, ref: ref, ranges: [][2]float64{r(lo, hi)}, scalar: l1(f)}
	if g != nil {
		s.nd = n1(g)
	}
	return s
}

func u2(name, ref string, r1, r2 [2]float64, f func(x, y float64) float64, g func(a, b *ndarray.Array) (*ndarray.Array, error)) spec {
	s := spec{name: name, ref: ref, ranges: [][2]float64{r1, r2}, scalar: l2(f)}
	if g != nil {
		s.nd = n2(g)
	}
	return s
}

func u3(name, ref string, r1, r2, r3 [2]float64, f func(x, y, z float64) float64, g func(a, b, c *ndarray.Array) (*ndarray.Array, error)) spec {
	s := spec{name: name, ref: ref, ranges: [][2]float64{r1, r2, r3}, scalar: l3(f)}
	if g != nil {
		s.nd = n3(g)
	}
	return s
}

// ndOne is the order 1 of Polygamma1, a single-element array broadcast by nd.
var ndOne, _ = ndarray.FromData([]float64{1}, 1)

// specs: one row per public scalar function that SciPy has a ufunc for. The
// ranges are those of the earlier docs/perf.md (31 rows) and, for the rest, a
// representative interval of the function.
var specs = []spec{
	u1("Gamma", "sp.gamma", 0.1, 170, special.Gamma, nd.Gamma),
	u1("LogGamma", "sp.gammaln", 0.1, 1000, special.LogGamma, nd.LogGamma),
	u1("GammaSign", "sp.gammasgn", -20, 20, special.GammaSign, nd.GammaSign),
	u1("Digamma", "sp.digamma", 0.1, 100, special.Digamma, nd.Digamma),
	u2("Beta", "sp.beta", r(0.1, 50), r(0.1, 50), special.Beta, nd.Beta),
	u2("LogBeta", "sp.betaln", r(0.1, 1000), r(0.1, 1000), special.LogBeta, nd.LogBeta),
	u1("Erf", "sp.erf", -5, 5, special.Erf, nd.Erf),
	u1("Erfc", "sp.erfc", -5, 25, special.Erfc, nd.Erfc),
	u1("Erfcx", "sp.erfcx", -5, 50, special.Erfcx, nd.Erfcx),
	u1("ErfInv", "sp.erfinv", -0.999, 0.999, special.ErfInv, nd.ErfInv),
	u1("ErfcInv", "sp.erfcinv", 1e-10, 1.999, special.ErfcInv, nd.ErfcInv),
	u1("Ndtr", "sp.ndtr", -30, 10, special.Ndtr, nd.Ndtr),
	u1("Ndtri", "sp.ndtri", 1e-12, 0.999999, special.Ndtri, nd.Ndtri),
	u1("LogNdtr", "sp.log_ndtr", -100, 10, special.LogNdtr, nd.LogNdtr),
	u1("Expit", "sp.expit", -50, 50, special.Expit, nd.Expit),
	u1("LogExpit", "sp.log_expit", -50, 50, special.LogExpit, nd.LogExpit),
	u1("Logit", "sp.logit", 1e-6, 0.999999, special.Logit, nd.Logit),
	u1("Log1pExp", "sp.softplus", -50, 50, special.Log1pExp, nd.Log1pExp),
	u2("Xlogy", "sp.xlogy", r(0.1, 10), r(0.1, 10), special.Xlogy, nd.Xlogy),
	u2("Xlog1py", "sp.xlog1py", r(0.1, 10), r(0.1, 10), special.Xlog1py, nd.Xlog1py),
	u2("Binom", "sp.binom", r(0.5, 40), r(0, 40), special.Binom, nd.Binom),
	u1("Zeta", "sp.zeta", 1.01, 60, special.Zeta, nd.Zeta),
	u2("HurwitzZeta", "sp.zeta", r(1.5, 20), r(0.1, 20), special.HurwitzZeta, nd.HurwitzZeta),
	{name: "Polygamma", ref: "lambda x: sp.polygamma(1, x)", ranges: [][2]float64{r(0.1, 100)},
		scalar: l1(func(x float64) float64 { return special.Polygamma(1, x) }),
		nd:     func(a []*ndarray.Array) error { _, err := nd.Polygamma(ndOne, a[0]); return err }},
	u2("GammaP", "sp.gammainc", r(0.5, 50), r(0.1, 60), special.GammaP, nd.GammaP),
	u2("GammaQ", "sp.gammaincc", r(0.5, 50), r(0.1, 60), special.GammaQ, nd.GammaQ),
	u2("GammaPInv", "sp.gammaincinv", r(0.5, 50), r(1e-6, 0.999999), special.GammaPInv, nd.GammaPInv),
	u2("GammaQInv", "sp.gammainccinv", r(0.5, 50), r(1e-6, 0.999999), special.GammaQInv, nd.GammaQInv),
	u3("BetaInc", "sp.betainc", r(0.5, 50), r(0.5, 50), r(0.01, 0.99), special.BetaInc, nd.BetaInc),
	u3("BetaIncC", "sp.betaincc", r(0.5, 50), r(0.5, 50), r(0.01, 0.99), special.BetaIncC, nd.BetaIncC),
	u3("BetaIncInv", "sp.betaincinv", r(0.5, 50), r(0.5, 50), r(0.01, 0.99), special.BetaIncInv, nd.BetaIncInv),
	u3("BetaIncCInv", "sp.betainccinv", r(0.5, 50), r(0.5, 50), r(0.01, 0.99), special.BetaIncCInv, nd.BetaIncCInv),
	u1("J0", "sp.j0", 0, 100, special.J0, nd.J0),
	u1("J1", "sp.j1", 0, 100, special.J1, nd.J1),
	u1("Y0", "sp.y0", 0.1, 100, special.Y0, nd.Y0),
	u1("Y1", "sp.y1", 0.1, 100, special.Y1, nd.Y1),
	u1("I0", "sp.i0", 0, 50, special.I0, nd.I0),
	u1("I0e", "sp.i0e", 0, 50, special.I0e, nd.I0e),
	u1("I1", "sp.i1", 0, 50, special.I1, nd.I1),
	u1("I1e", "sp.i1e", 0, 50, special.I1e, nd.I1e),
	u1("K0", "sp.k0", 0.01, 50, special.K0, nd.K0),
	u1("K0e", "sp.k0e", 0.01, 50, special.K0e, nd.K0e),
	u1("K1", "sp.k1", 0.01, 50, special.K1, nd.K1),
	u1("K1e", "sp.k1e", 0.01, 50, special.K1e, nd.K1e),
	u2("Jv", "sp.jv", r(0, 20), r(0.1, 50), special.Jv, nd.Jv),
	u2("Yv", "sp.yv", r(0, 20), r(0.1, 50), special.Yv, nd.Yv),
	u2("Iv", "sp.iv", r(0, 20), r(0.1, 50), special.Iv, nd.Iv),
	u2("Ive", "sp.ive", r(0, 20), r(0.1, 50), special.Ive, nd.Ive),
	u2("Kv", "sp.kv", r(0, 20), r(0.1, 50), special.Kv, nd.Kv),
	u2("Kve", "sp.kve", r(0, 20), r(0.1, 50), special.Kve, nd.Kve),
	u1("EllipK", "sp.ellipk", -10, 0.999, special.EllipK, nd.EllipK),
	u1("EllipKm1", "sp.ellipkm1", 1e-6, 1, special.EllipKm1, nd.EllipKm1),
	u1("EllipE", "sp.ellipe", -10, 0.999, special.EllipE, nd.EllipE),
}

// kernel is a public array kernel (special.GammaTo, ...): one call over the
// points, the Go counterpart of a ufunc loop. spec names the row whose data and
// SciPy callable it shares.
type kernel struct {
	name string
	spec string
	k1   func(dst, x []float64)
	k2   func(dst, a, b []float64, sa, sb int)
}

var kernels = []kernel{
	{name: "GammaTo", spec: "Gamma", k1: special.GammaTo},
	{name: "LogGammaTo", spec: "LogGamma", k1: special.LogGammaTo},
	{name: "DigammaTo", spec: "Digamma", k1: special.DigammaTo},
	{name: "ErfTo", spec: "Erf", k1: special.ErfTo},
	{name: "ErfcTo", spec: "Erfc", k1: special.ErfcTo},
	{name: "ErfcxTo", spec: "Erfcx", k1: special.ErfcxTo},
	{name: "ErfInvTo", spec: "ErfInv", k1: special.ErfInvTo},
	{name: "ErfcInvTo", spec: "ErfcInv", k1: special.ErfcInvTo},
	{name: "NdtrTo", spec: "Ndtr", k1: special.NdtrTo},
	{name: "NdtriTo", spec: "Ndtri", k1: special.NdtriTo},
	{name: "LogNdtrTo", spec: "LogNdtr", k1: special.LogNdtrTo},
	{name: "ExpitTo", spec: "Expit", k1: special.ExpitTo},
	{name: "LogitTo", spec: "Logit", k1: special.LogitTo},
	{name: "Log1pExpTo", spec: "Log1pExp", k1: special.Log1pExpTo},
	{name: "ZetaTo", spec: "Zeta", k1: special.ZetaTo},
	{name: "TrigammaTo", spec: "Polygamma", k1: special.TrigammaTo},
	{name: "BetaTo", spec: "Beta", k2: special.BetaTo},
	{name: "LogBetaTo", spec: "LogBeta", k2: special.LogBetaTo},
	{name: "XlogyTo", spec: "Xlogy", k2: special.XlogyTo},
	{name: "HurwitzZetaTo", spec: "HurwitzZeta", k2: special.HurwitzZetaTo},
}

// worker is the SciPy process, started once per test binary run that times.
var worker *refbench.Worker

func startWorker(t testing.TB) *refbench.Worker {
	t.Helper()
	uv, err := exec.LookPath("uv")
	if err != nil {
		home, _ := os.UserHomeDir()
		uv = filepath.Join(home, ".local", "bin", "uv")
	}
	refs, err := filepath.Abs("refs.txt")
	if err != nil {
		t.Fatal(err)
	}
	w, err := refbench.StartPython(context.Background(), uv, refs)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { w.Close() })
	return w
}

// cases builds the registry. ref is nil unless a worker is given (the CI mode
// and -perf.record need none).
func cases(w *refbench.Worker) []refbench.Case {
	var out []refbench.Case
	for _, s := range specs {
		args := make([][]float64, len(s.ranges))
		arrs := make([]*ndarray.Array, len(s.ranges))
		for i, rg := range s.ranges {
			args[i] = pts(rg[0], rg[1])
			arrs[i], _ = ndarray.FromData(args[i], n)
		}
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(setup(s.ranges), s.ref, n)
		}
		dst := make([]float64, n)
		out = append(out, refbench.Case{
			Name:      s.name,
			Func:      s.name,
			Reference: "scipy.special " + s.ref + " (ufunc over " + fmt.Sprint(n) + " points)",
			Parity:    refbench.ParityNone, // SciPy's own errors reach 1e15 ulp (docs/accuracy.md); the ulp fixtures are the oracle
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					s.scalar(args, dst)
				}
				sink = dst[0]
			},
			Ref: ref,
		})
		if s.nd != nil {
			out = append(out, refbench.Case{
				Name:      "nd/" + s.name,
				Func:      "nd." + s.name,
				Reference: "scipy.special " + s.ref + " (ufunc over " + fmt.Sprint(n) + " points)",
				Parity:    refbench.ParityNone,
				Go: func(iters int) {
					for ; iters > 0; iters-- {
						if err := s.nd(arrs); err != nil {
							panic(err)
						}
					}
				},
				Ref: ref,
			})
		}
	}
	byName := map[string]spec{}
	for _, s := range specs {
		byName[s.name] = s
	}
	for _, k := range kernels {
		s := byName[k.spec]
		args := make([][]float64, len(s.ranges))
		for i, rg := range s.ranges {
			args[i] = pts(rg[0], rg[1])
		}
		var ref refbench.Timer
		if w != nil {
			ref = w.Timer(setup(s.ranges), s.ref, n)
		}
		dst := make([]float64, n)
		out = append(out, refbench.Case{
			Name:      k.name,
			Func:      k.name,
			Reference: "scipy.special " + s.ref + " (ufunc over " + fmt.Sprint(n) + " points)",
			Parity:    refbench.ParityNone,
			Go: func(iters int) {
				for ; iters > 0; iters-- {
					if k.k1 != nil {
						k.k1(dst, args[0])
					} else {
						k.k2(dst, args[0], args[1], 1, 1)
					}
				}
				sink = dst[0]
			},
			Ref: ref,
		})
	}
	return append(out, xcases(w)...)
}

var smokeFlag = flag.Bool("perf.smoke", false, "TestRefSmoke: run every SciPy call once")

// TestRefSmoke runs every reference call once: a typo in a SciPy expression is a
// failure here, not in a timing run.
func TestRefSmoke(t *testing.T) {
	if !*smokeFlag {
		t.Skip("set -perf.smoke (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, c := range cases(w) {
		if _, err := c.Ref(1); err != nil {
			t.Errorf("%s: %v", c.Name, err)
		}
	}
}

var parityFlag = flag.Bool("perf.parity", false, "TestDataParity: compare the data digests with NumPy's")

// running reports rbtest's -perf.run (the worker is started only then).
func running() bool {
	f := flag.Lookup("perf.run")
	return f != nil && f.Value.String() == "true"
}

func TestPerf(t *testing.T) {
	var w *refbench.Worker
	if running() {
		w = startWorker(t)
	}
	opt := refbench.Options{Repo: "go-numerics/special"}
	if w != nil {
		v, err := w.RefVersions()
		if err != nil {
			t.Fatal(err)
		}
		opt.RefVersions = v
	}
	rbtest.Perf(t, refbench.CIConfig{Root: "..", Cases: cases(w)}, opt)
}

// TestDataParity checks, for every row, that the Go points and NumPy's are the
// same bytes: a ratio between two functions on different data means nothing.
func TestDataParity(t *testing.T) {
	if !*parityFlag {
		t.Skip("set -perf.parity (starts the SciPy worker through uv)")
	}
	w := startWorker(t)
	for _, s := range specs {
		h := sha256.New()
		for _, rg := range s.ranges {
			for _, v := range pts(rg[0], rg[1]) {
				var b [8]byte
				binary.LittleEndian.PutUint64(b[:], math.Float64bits(v))
				h.Write(b[:])
			}
		}
		got, err := w.DataHash(setup(s.ranges), n)
		if err != nil {
			t.Fatalf("%s: %v", s.name, err)
		}
		if want := hex.EncodeToString(h.Sum(nil)); got != want {
			t.Errorf("%s: NumPy data %s, Go data %s", s.name, got, want)
		}
	}
	for _, r := range xrows() {
		if r.data == nil {
			continue
		}
		got, err := w.DataHash(r.setup, n)
		if err != nil {
			t.Fatalf("%s: %v", r.name, err)
		}
		if sum := sha256.Sum256(r.data); got != hex.EncodeToString(sum[:]) {
			t.Errorf("%s: NumPy data %s, Go data %x", r.name, got, sum)
		}
	}
}
