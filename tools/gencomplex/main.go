// Command gencomplex writes the accuracy fixtures of the cmplx package,
// ../cmplx/testdata/c_<name>.tsv.gz, from mpmath (docs/design-complex.md,
// "Validation"):
//
//	go -C tools run ./gencomplex [name ...]
//
// Each row holds the region, the float64 arguments, and the real and
// imaginary parts of the exact value, each split as a double-double
// (hi = the correctly rounded float64, lo = the rest). Every point is
// evaluated at 192 and at 320 bits; a point whose two evaluations differ by
// more than 2^-80 relative in either part is dropped and counted in the
// header, so that an mpmath evaluation that silently lost precision never
// becomes the truth the tests are held to.
//
// Signed zeros: mpmath has none, so a zero argument is bound as ±2^-4000
// with the sign of the zero. mpmath then evaluates the limit from that side,
// which is what C11 Annex G G.6 asks of a function on its branch cut ("taking
// into account the sign of zero"); away from a cut the shift is far below
// the last bit of any float64 result.
//
// Sampling is seeded by the fixture name, so a rerun with the same mpmath
// reproduces the files byte for byte.
package main

import (
	"bytes"
	"compress/gzip"
	"fmt"
	"hash/fnv"
	"log"
	"math"
	"math/big"
	"math/rand/v2"
	"os"
	"strings"
	"sync"

	"github.com/go-numerics/special/tools/internal/mporacle"
)

const (
	prec1 = 192
	prec2 = 320
)

// region draws one point (2 or 4 float64 arguments).
type region struct {
	name string
	n    int
	gen  func(r *rand.Rand) []float64
}

type fixture struct {
	name  string
	vars  []string // argument names bound in expr
	expr  string   // mpmath expression of an mpc value
	doc   string
	regs  []region
	swapX bool // exchange the two arguments after drawing (cos/sin/tan: f(z) = g(iz))
}

// lu draws a magnitude log-uniformly in [lo, hi] with a random sign.
func lu(r *rand.Rand, lo, hi float64) float64 {
	a, b := math.Log(lo), math.Log(hi)
	v := math.Exp(a + (b-a)*r.Float64())
	if r.IntN(2) == 0 {
		v = -v
	}
	return v
}

// lu2 is lu for magnitudes given as powers of two (reaches the subnormals).
func lu2(r *rand.Rand, elo, ehi float64) float64 {
	v := math.Ldexp(1+r.Float64(), int(math.Floor(elo+(ehi-elo)*r.Float64())))
	if r.IntN(2) == 0 {
		v = -v
	}
	return v
}

func uni(r *rand.Rand, lo, hi float64) float64 { return lo + (hi-lo)*r.Float64() }

// polar draws |z| log-uniformly in [2^elo, 2^ehi] and arg uniformly.
func polar(elo, ehi float64) func(r *rand.Rand) []float64 {
	return func(r *rand.Rand) []float64 {
		m := math.Abs(lu2(r, elo, ehi))
		t := uni(r, -math.Pi, math.Pi)
		return []float64{m * math.Cos(t), m * math.Sin(t)}
	}
}

// axes puts one part at a signed zero.
func axes(r *rand.Rand) []float64 {
	z := math.Copysign(0, float64(r.IntN(2)*2-1))
	v := lu(r, 1e-3, 50)
	if r.IntN(2) == 0 {
		return []float64{z, v}
	}
	return []float64{v, z}
}

// The exponential family: e^x cis y and its hyperbolic forms.
func expRegions(signedX bool) []region {
	sx := func(r *rand.Rand, v float64) float64 {
		if signedX && r.IntN(2) == 0 {
			return -v
		}
		return v
	}
	return []region{
		{"tiny", 200, polar(-1074, -26)},
		{"moderate", 300, polar(-3, 3)},
		{"axes", 100, axes},
		{"wide", 300, func(r *rand.Rand) []float64 { return []float64{uni(r, -745, 709), lu(r, 1e-3, 1e4)} }},
		{"bigy", 150, func(r *rand.Rand) []float64 { return []float64{uni(r, -5, 5), lu(r, 1e4, 1e22)} }},
		{"overflow", 300, func(r *rand.Rand) []float64 {
			x := uni(r, 709, 711)
			if r.IntN(2) == 0 {
				x = uni(r, 1454, 1460)
			}
			return []float64{sx(r, x), lu(r, 1e-310, 3)}
		}},
	}
}

func tanhRegions() []region {
	return []region{
		{"tiny", 200, polar(-1074, -26)},
		{"moderate", 300, polar(-3, 3)},
		{"axes", 100, axes},
		{"wide", 300, func(r *rand.Rand) []float64 { return []float64{uni(r, -30, 30), lu(r, 1e-3, 1e4)} }},
		{"huge", 300, func(r *rand.Rand) []float64 {
			x := uni(r, 22, 800)
			if r.IntN(3) == 0 {
				x = uni(r, 354, 357)
			}
			return []float64{math.Copysign(x, float64(r.IntN(2)*2-1)), lu(r, 1e-300, 1e3)}
		}},
		{"pole", 200, func(r *rand.Rand) []float64 {
			k := float64(r.IntN(7) - 3)
			return []float64{lu(r, 1e-300, 0.1), (k+0.5)*math.Pi + lu(r, 1e-16, 1e-2)}
		}},
	}
}

// cutYs are the imaginary parts placed on and next to the negative real
// axis, the branch cut of the logarithm and the square root, on both sides.
var cutYs = []float64{0, math.Copysign(0, -1), 0x1p-1074, -0x1p-1074, 0x1p-50, -0x1p-50}

// The logarithm family and the square root.
func logRegions() []region {
	return []region{
		{"tiny", 200, polar(-1074, -26)},
		{"subnormal", 100, func(r *rand.Rand) []float64 {
			a := []float64{lu2(r, -1074, -1023), lu2(r, -1074, 10)}
			if r.IntN(2) == 0 {
				a[0], a[1] = a[1], a[0]
			}
			return a
		}},
		{"moderate", 300, polar(-3, 3)},
		{"band", 300, func(r *rand.Rand) []float64 {
			m, t := uni(r, 0.71, 1.73), uni(r, -math.Pi, math.Pi)
			return []float64{m * math.Cos(t), m * math.Sin(t)}
		}},
		{"unit", 300, func(r *rand.Rand) []float64 {
			if r.IntN(3) == 0 {
				return []float64{math.Copysign(1, float64(r.IntN(2)*2-1)), lu(r, 1e-300, 1e-5)}
			}
			m, t := 1+lu2(r, -52, -20), uni(r, -math.Pi, math.Pi)
			return []float64{m * math.Cos(t), m * math.Sin(t)}
		}},
		{"huge", 200, polar(500, 1023)},
		{"cut", 300, func(r *rand.Rand) []float64 {
			return []float64{-math.Abs(lu(r, 1e-300, 1e300)), cutYs[r.IntN(len(cutYs))]}
		}},
		{"axes", 100, axes},
	}
}

// Products and quotients: four parts a, b, c, d.
func arithRegions(div bool) []region {
	four := func(f func(r *rand.Rand) float64) func(r *rand.Rand) []float64 {
		return func(r *rand.Rand) []float64 { return []float64{f(r), f(r), f(r), f(r)} }
	}
	return []region{
		{"moderate", 400, four(func(r *rand.Rand) float64 { return lu2(r, -10, 10) })},
		{"cancel", 400, func(r *rand.Rand) []float64 {
			a, b, c := lu2(r, -10, 10), lu2(r, -10, 10), lu2(r, -10, 10)
			// ac - bd (product) or ac + bd (quotient numerator) nearly cancels.
			d := a * c / b * (1 + lu(r, 1e-16, 1e-3))
			if div {
				d = -d
			}
			return []float64{a, b, c, d}
		}},
		{"wide", 400, four(func(r *rand.Rand) float64 { return lu2(r, -1000, 1000) })},
		{"subnormal", 200, four(func(r *rand.Rand) float64 {
			if r.IntN(2) == 0 {
				return lu2(r, -1074, -1000)
			}
			return lu2(r, -10, 10)
		})},
		{"huge", 200, four(func(r *rand.Rand) float64 { return lu2(r, 500, 1023) })},
	}
}

func fixtures() []fixture {
	xy := []string{"x", "y"}
	abcd := []string{"a", "b", "c", "d"}
	z := "mp.mpc(x, y)"
	return []fixture{
		{name: "mul", vars: abcd, expr: "mp.mpc(a, b) * mp.mpc(c, d)", doc: "(a+ib)(c+id)", regs: arithRegions(false)},
		{name: "div", vars: abcd, expr: "mp.mpc(a, b) / mp.mpc(c, d)", doc: "(a+ib)/(c+id)", regs: arithRegions(true)},
		{name: "exp", vars: xy, expr: "mp.exp(" + z + ")", doc: "exp(x+iy)", regs: expRegions(false)},
		{name: "cosh", vars: xy, expr: "mp.cosh(" + z + ")", doc: "cosh(x+iy)", regs: expRegions(true)},
		{name: "sinh", vars: xy, expr: "mp.sinh(" + z + ")", doc: "sinh(x+iy)", regs: expRegions(true)},
		{name: "cos", vars: xy, expr: "mp.cos(" + z + ")", doc: "cos(x+iy) (regions of cosh, arguments exchanged)", regs: expRegions(true), swapX: true},
		{name: "sin", vars: xy, expr: "mp.sin(" + z + ")", doc: "sin(x+iy) (regions of sinh, arguments exchanged)", regs: expRegions(true), swapX: true},
		{name: "log", vars: xy, expr: "mp.log(" + z + ")", doc: "log(x+iy), principal branch", regs: logRegions()},
		{name: "log10", vars: xy, expr: "mp.log(" + z + ")/mp.log(10)", doc: "log10(x+iy)", regs: logRegions()},
		{name: "log2", vars: xy, expr: "mp.log(" + z + ")/mp.log(2)", doc: "log2(x+iy)", regs: logRegions()},
		{name: "sqrt", vars: xy, expr: "mp.sqrt(" + z + ")", doc: "sqrt(x+iy), principal branch", regs: logRegions()},
		{name: "tanh", vars: xy, expr: "mp.tanh(" + z + ")", doc: "tanh(x+iy)", regs: tanhRegions()},
		{name: "tan", vars: xy, expr: "mp.tan(" + z + ")", doc: "tan(x+iy) (regions of tanh, arguments exchanged)", regs: tanhRegions(), swapX: true},
		{name: "cot", vars: xy, expr: "mp.cot(" + z + ")", doc: "cot(x+iy) (regions of tanh, arguments exchanged)", regs: tanhRegions(), swapX: true},
	}
}

// bind returns x exactly as a big.Float, a signed zero as ±2^-4000.
func bind(x float64) *big.Float {
	if x == 0 {
		return new(big.Float).SetMantExp(big.NewFloat(math.Copysign(1, x)), -4000)
	}
	return new(big.Float).SetFloat64(x)
}

func split(v *big.Float) (float64, float64) {
	hi := mporacle.Float64(v)
	if math.IsInf(hi, 0) {
		return hi, 0
	}
	d := new(big.Float).SetPrec(prec2+64).Sub(v, new(big.Float).SetFloat64(hi))
	return hi, mporacle.Float64(d)
}

// agree reports whether two evaluations agree to 2^-80 relative.
func agree(a, b *big.Float) bool {
	if b.Sign() == 0 {
		return a.Sign() == 0
	}
	d := new(big.Float).SetPrec(prec2).Sub(a, b)
	d.Abs(d)
	t := new(big.Float).SetPrec(prec2).Abs(b)
	t.SetMantExp(t, -80)
	return d.Cmp(t) <= 0
}

type point struct {
	region string
	args   []float64
}

func (f fixture) points() []point {
	h := fnv.New64a()
	h.Write([]byte("go-numerics/special/cmplx:" + f.name))
	r := rand.New(rand.NewPCG(h.Sum64(), 0x5eed))
	seen := map[string]bool{}
	var pts []point
	for _, rg := range f.regs {
		for range rg.n {
			a := rg.gen(r)
			if f.swapX {
				a[0], a[1] = a[1], a[0]
			}
			key := rg.name + fmt.Sprint(a)
			if seen[key] {
				continue
			}
			seen[key] = true
			pts = append(pts, point{rg.name, a})
		}
	}
	return pts
}

func generate(f fixture) error {
	o, err := mporacle.Start()
	if err != nil {
		return err
	}
	defer o.Close()
	pts := f.points()
	var reqs []mporacle.Req
	for _, p := range pts {
		vars := map[string]*big.Float{}
		for i, n := range f.vars {
			vars[n] = bind(p.args[i])
		}
		for _, prec := range []uint{prec1, prec2} {
			reqs = append(reqs,
				mporacle.Req{Expr: "mp.re(" + f.expr + ")", Prec: prec, Vars: vars},
				mporacle.Req{Expr: "mp.im(" + f.expr + ")", Prec: prec, Vars: vars})
		}
	}
	vals, _, err := o.Batch(reqs)
	if err != nil {
		return fmt.Errorf("%s: %w", f.name, err)
	}
	var body bytes.Buffer
	dropped := 0
	for i, p := range pts {
		v := vals[4*i : 4*i+4]
		if !agree(v[0], v[2]) || !agree(v[1], v[3]) {
			dropped++
			continue
		}
		reh, rel := split(v[2])
		imh, iml := split(v[3])
		cols := []string{p.region}
		for _, a := range p.args {
			cols = append(cols, mporacle.Repr(a))
		}
		for _, x := range []float64{reh, rel, imh, iml} {
			cols = append(cols, mporacle.Repr(x))
		}
		body.WriteString(strings.Join(cols, "\t") + "\n")
	}
	var out bytes.Buffer
	gz, _ := gzip.NewWriterLevel(&out, gzip.BestCompression)
	fmt.Fprintf(gz, "# c_%s: %s; %d argument(s); columns: region, args, re hi, re lo, im hi, im lo\n", f.name, f.doc, len(f.vars))
	fmt.Fprintf(gz, "# oracle: mpmath %s at %d and %d bits (rows that disagree dropped: %d); signed zero arguments bound as +-2^-4000\n",
		mporacle.MpmathVersion, prec1, prec2, dropped)
	gz.Write(body.Bytes())
	if err := gz.Close(); err != nil {
		return err
	}
	log.Printf("c_%-6s %5d rows, dropped %d", f.name, len(pts)-dropped, dropped)
	return os.WriteFile("../cmplx/testdata/c_"+f.name+".tsv.gz", out.Bytes(), 0o644)
}

func main() {
	want := map[string]bool{}
	for _, a := range os.Args[1:] {
		want[a] = true
	}
	var wg sync.WaitGroup
	errs := make(chan error, 32)
	for _, f := range fixtures() {
		if len(want) > 0 && !want[f.name] {
			continue
		}
		wg.Go(func() {
			if err := generate(f); err != nil {
				errs <- err
			}
		})
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		log.Fatal(err)
	}
}
