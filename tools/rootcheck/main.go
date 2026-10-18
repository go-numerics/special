// Command rootcheck measures the root error of the incomplete beta and gamma
// inverses against mpmath and writes the points with their exact roots to
// ../testdata/roots/<name>.tsv, which TestInverseRootRounding reads:
//
//	go -C tools run ./rootcheck [-n 300] [-seed 1] [name ...]
//
// The points are those of the correct-rounding check: a, b log-uniform in
// [e^-1, e^6], p uniform in [0.01, 0.99] (q for the complements), from a
// fixed PCG seed. The root is mpmath's findroot of the forward function at
// 200 bits (60 digits), started from the library's value, and verified to
// bracket: the forward function minus the target changes sign between the
// two floats around it. Each row carries the root as a double-double
// (hi + lo), so that the error is measured in ulps of the root itself,
// not as a residual.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"math"
	"math/big"
	"math/rand/v2"
	"os"
	"path/filepath"
	"sort"

	"github.com/go-numerics/special"
	"github.com/go-numerics/special/tools/internal/mporacle"
)

// prec is the oracle's working precision in bits (about 60 digits).
const prec = 200

type inverse struct {
	name  string
	beta  bool // two shape parameters (a, b) or one (a)
	fn    func(a, b, p float64) float64
	expr  string // the forward function minus the target, of t
	deriv string // its derivative in t (the density)
}

const (
	betaDensity  = "t**(a-1)*(1-t)**(b-1)/mp.beta(a, b)"
	gammaDensity = "mp.exp((a-1)*mp.log(t) - t - mp.loggamma(a))"
)

var inverses = []inverse{
	{name: "betaincinv", beta: true, fn: special.BetaIncInv,
		expr: "mp.betainc(a, b, 0, t, regularized=True) - p", deriv: betaDensity},
	{name: "betainccinv", beta: true, fn: special.BetaIncCInv,
		expr: "p - mp.betainc(a, b, t, 1, regularized=True)", deriv: betaDensity},
	{name: "gammaincinv", fn: func(a, _, p float64) float64 { return special.GammaPInv(a, p) },
		expr: "mp.gammainc(a, 0, t, regularized=True) - p", deriv: gammaDensity},
	{name: "gammainccinv", fn: func(a, _, p float64) float64 { return special.GammaQInv(a, p) },
		expr: "p - mp.gammainc(a, t, mp.inf, regularized=True)", deriv: gammaDensity},
}

func bf(x float64) *big.Float { return new(big.Float).SetPrec(53).SetFloat64(x) }

func main() {
	n := flag.Int("n", 300, "points per function")
	seed := flag.Uint64("seed", 1, "PCG seed")
	flag.Parse()
	want := map[string]bool{}
	for _, a := range flag.Args() {
		want[a] = true
	}
	o, err := mporacle.Start()
	if err != nil {
		log.Fatal(err)
	}
	defer o.Close()
	for _, inv := range inverses {
		if len(want) > 0 && !want[inv.name] {
			continue
		}
		r := rand.New(rand.NewPCG(*seed, 0x5eed))
		var rows []string
		var errs []float64
		above := 0
		for i := 0; i < *n; i++ {
			a := math.Exp(-1 + 7*r.Float64())
			b := math.Exp(-1 + 7*r.Float64())
			p := 0.01 + 0.98*r.Float64()
			if !inv.beta {
				b = 0
			}
			x := inv.fn(a, b, p)
			vars := map[string]*big.Float{"a": bf(a), "b": bf(b), "p": bf(p), "x0": bf(x)}
			// mp.re: mpmath's hypergeometric sums can leave an imaginary
			// part of 1e-90 on a real value
			// Newton's method from the library's value (within a few ulps
			// of the root), with the density as the derivative.
			root := o.Eval(prec, "mp.re(mp.findroot(lambda t: mp.re("+inv.expr+"), x0, df=lambda t: mp.re("+inv.deriv+
				"), solver='newton', tol=mpf(2)**-180, maxsteps=100, verify=False))", vars)
			hi := mporacle.Float64(root)
			lo := mporacle.Float64(new(big.Float).SetPrec(prec).Sub(root, bf(hi)))
			check(o, inv, vars, root, hi)
			e := math.Abs((x-hi)-lo) / ulp(hi)
			errs = append(errs, e)
			if e > 0.5 {
				above++
			}
			if inv.beta {
				rows = append(rows, fmt.Sprintf("%v\t%v\t%v\t%v\t%v", a, b, p, hi, lo))
			} else {
				rows = append(rows, fmt.Sprintf("%v\t%v\t%v\t%v", a, p, hi, lo))
			}
		}
		write(inv, rows)
		sort.Float64s(errs)
		fmt.Printf("%-13s n=%d  max %.3f ulp  median %.3f  p99 %.3f  above 0.5: %d\n",
			inv.name, len(errs), errs[len(errs)-1], errs[len(errs)/2], errs[len(errs)*99/100], above)
	}
}

// check verifies that the forward function minus the target changes sign
// between the floats on either side of the root (a findroot that wandered
// would not), and stops the run otherwise.
func check(o *mporacle.Oracle, inv inverse, vars map[string]*big.Float, root *big.Float, hi float64) {
	lo, up := math.Nextafter(hi, 0), math.Nextafter(hi, math.Inf(1))
	if root.Cmp(bf(hi)) > 0 {
		lo = hi
	} else {
		up = hi
	}
	g := func(t float64) int {
		v := map[string]*big.Float{"t": bf(t)}
		for k, x := range vars {
			v[k] = x
		}
		return o.Eval(prec, "mp.re("+inv.expr+")", v).Sign()
	}
	if gl, gu := g(lo), g(up); gl*gu > 0 {
		log.Fatalf("%s %v: root %v not bracketed by [%v, %v] (signs %d %d)", inv.name, vars, root, lo, up, gl, gu)
	}
}

func ulp(x float64) float64 {
	x = math.Abs(x)
	if x < 0x1p-1022 {
		return 0x1p-1074
	}
	_, e := math.Frexp(x)
	return math.Ldexp(1, e-53)
}

func write(inv inverse, rows []string) {
	path := filepath.Join("..", "testdata", "roots", inv.name+".tsv")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		log.Fatal(err)
	}
	f, err := os.Create(path)
	if err != nil {
		log.Fatal(err)
	}
	w := bufio.NewWriter(f)
	cols := "a\tb\tp\troot_hi\troot_lo"
	if !inv.beta {
		cols = "a\tp\troot_hi\troot_lo"
	}
	fmt.Fprintf(w, "# generated by: go -C tools run ./rootcheck (mpmath %s findroot at %d bits)\n# %s\n", mporacle.MpmathVersion, prec, cols)
	for _, r := range rows {
		fmt.Fprintln(w, r)
	}
	if err := w.Flush(); err != nil {
		log.Fatal(err)
	}
	if err := f.Close(); err != nil {
		log.Fatal(err)
	}
}
