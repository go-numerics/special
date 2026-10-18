// Command genfit writes the piecewise polynomial tables listed in jobs
// (../zz_<name>_coeffs.go), fitted to mpmath:
//
//	go -C tools run ./genfit [name ...]
//
// Method (that of testdata/gen/fit.py, which generates the older tables):
// on each piece [a, b] the function is interpolated at the n+1 Chebyshev
// nodes of the first kind at 200-bit precision, converted to the monomial
// basis in s = t - c around the midpoint c, and the degree is the smallest
// whose interpolation error, checked on 401 points against mpmath, is
// below 2^-64 relative. The constant and linear coefficients are also
// emitted as double-doubles (c0lo, c1lo). The function values, the
// cosines and the error's decimal rendering come from mpmath (mporacle);
// the rest is big.Float arithmetic at the same precision with correct
// rounding, operation for operation as in fit.py (mp.fsum is an exact sum
// rounded once).
package main

import (
	"bytes"
	"fmt"
	"log"
	"math/big"
	"os"

	"github.com/go-numerics/special/tools/internal/mporacle"
)

const prec = 200

type job struct {
	name, doc string
	// f returns the oracle expression (and its working precision) for the
	// value at t, mirroring fit.py's Python function (the variable is t).
	f     func(t *big.Float) (string, uint)
	edges []float64
}

var jobs = []job{
	{"erfcxPieces", "erfcxPieces: erfcx(x) = exp(x²) erfc(x) on [0.5, 8].",
		func(t *big.Float) (string, uint) { return "mp.exp(t * t) * mp.erfc(t)", prec + 60 },
		seq(0.5, 0.5, 16)},
	{"logErfcxPieces", "logErfcxPieces: log erfcx(x) on [0.5, 8] (widths 1/2 on [0.5, 2], 1 on [2, 4], 2 on [4, 8]).",
		func(t *big.Float) (string, uint) { return "mp.log(mp.exp(t * t) * mp.erfc(t))", prec + 60 },
		[]float64{0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0}},
	{"logErfcxTailPieces", "logErfcxTailPieces: H(u) = log(x erfcx(x)), u = 1/x², on [0, 1/64] (x >= 8).",
		func(t *big.Float) (string, uint) {
			if t.Sign() == 0 {
				return "-mp.log(mp.sqrt(mp.pi))", prec
			}
			return "(lambda x: mp.workprec(mp.prec + 200)(lambda: mp.log(x * mp.exp(x * x) * mp.erfc(x)))())(1 / mp.sqrt(t))", prec
		},
		[]float64{0.0, 1.0 / 128, 1.0 / 64}},
	{"digammaMidPieces", "digammaMidPieces: ψ(x) on [2, 10] (widths 1/4 on [2, 3], 1/2 on [3, 4], 1 on [4, 10]).",
		func(t *big.Float) (string, uint) { return "mp.digamma(t)", prec },
		[]float64{2.0, 2.25, 2.5, 2.75, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}},
	{"erfInvSmallPieces", "erfInvSmallPieces: R(t) = erfinv(√t)/√t on [0, 1/4]; erfinv(p) = p R(p²) for |p| <= 1/2.",
		func(t *big.Float) (string, uint) {
			if t.Sign() == 0 {
				return "mp.sqrt(mp.pi) / 2", prec
			}
			return "(lambda r: mp.workprec(mp.prec + 40)(lambda: mp.erfinv(r) / r)())(mp.sqrt(t))", prec
		},
		[]float64{0.0, 0.125, 0.25}},
	{"erfcInvPieces", "erfcInvPieces: X(w) = erfcinv(exp(-w²)) on [0.8125, 31.0625] (q = exp(-w²) < 1/2; widths 1/4, 1, 2, 4).",
		func(t *big.Float) (string, uint) {
			return "mp.findroot(lambda x: mp.log(mp.erfc(x)) + t * t, (t if t > 2 else mp.erfinv(1 - mp.exp(-t * t))), tol=mpf(2) ** -(mp.prec + 20))", prec + 40
		},
		append(seq(0.8125, 0.25, 10), 4.0625, 5.0625, 6.0625, 7.0625, 9.0625, 11.0625, 15.0625, 19.0625, 23.0625, 27.0625, 31.0625)},
}

// seq returns a + step*k, k = 0..n-1, in float64 as Python computes it.
func seq(a, step float64, n int) []float64 {
	var s []float64
	for k := 0; k < n; k++ {
		s = append(s, a+step*float64(k))
	}
	return s
}

var (
	o     *mporacle.Oracle
	cosMu = map[string]*big.Float{}
)

func nf() *big.Float { return new(big.Float).SetPrec(prec) }

// eval evaluates the job's function at many points (one pipelined batch).
func eval(j job, ts []*big.Float) []*big.Float {
	reqs := make([]mporacle.Req, len(ts))
	for i, t := range ts {
		e, p := j.f(t)
		reqs[i] = mporacle.Req{Expr: e, Prec: p, Vars: map[string]*big.Float{"t": t}}
	}
	v, _, err := o.Batch(reqs)
	if err != nil {
		log.Fatal(err)
	}
	return v
}

// cosines returns the expressions' values, cached.
func cosines(exprs []string) []*big.Float {
	var miss []mporacle.Req
	var keys []string
	for _, e := range exprs {
		if _, ok := cosMu[e]; !ok {
			miss = append(miss, mporacle.Req{Expr: e, Prec: prec})
			keys = append(keys, e)
		}
	}
	if len(miss) > 0 {
		v, _, err := o.Batch(miss)
		if err != nil {
			log.Fatal(err)
		}
		for i, k := range keys {
			cosMu[k] = v[i]
		}
	}
	out := make([]*big.Float, len(exprs))
	for i, e := range exprs {
		out[i] = cosMu[e]
	}
	return out
}

// fsum returns the exactly rounded sum (mp.fsum).
func fsum(xs []*big.Float) *big.Float {
	acc := new(big.Float).SetPrec(1 << 14)
	for _, x := range xs {
		acc.Add(acc, x)
	}
	return nf().Set(acc)
}

func mul(x, y *big.Float) *big.Float { return nf().Mul(x, y) }

// chebInterp is fit.py's cheb_interp: c and the monomial coefficients in
// s = t - c of the degree-n Chebyshev interpolant on [a, b].
func chebInterp(j job, a, b float64, n int) (*big.Float, []*big.Float) {
	af, bf := nf().SetFloat64(a), nf().SetFloat64(b)
	c := nf().Quo(nf().Add(af, bf), big.NewFloat(2))
	h := nf().Quo(nf().Sub(bf, af), big.NewFloat(2))
	var ne []string
	for k := 0; k <= n; k++ {
		ne = append(ne, fmt.Sprintf("mp.cos(mp.pi * (%d + mpf(1) / 2) / (%d + 1))", k, n))
	}
	nodes := cosines(ne)
	ts := make([]*big.Float, n+1)
	for k, x := range nodes {
		ts[k] = nf().Add(c, mul(h, x))
	}
	vals := eval(j, ts)
	var ce []string
	for jj := 0; jj <= n; jj++ {
		for k := 0; k <= n; k++ {
			ce = append(ce, fmt.Sprintf("mp.cos(mp.pi * %d * (%d + mpf(1) / 2) / (%d + 1))", jj, k, n))
		}
	}
	cs := cosines(ce)
	cc := make([]*big.Float, n+1)
	for jj := 0; jj <= n; jj++ {
		terms := make([]*big.Float, n+1)
		for k := 0; k <= n; k++ {
			terms[k] = mul(vals[k], cs[jj*(n+1)+k])
		}
		s := fsum(terms)
		cc[jj] = nf().Quo(nf().Mul(big.NewFloat(2), s), big.NewFloat(float64(n+1)))
	}
	cc[0] = nf().Quo(cc[0], big.NewFloat(2))
	mono := make([]*big.Float, n+1)
	for i := range mono {
		mono[i] = nf()
	}
	one := func(v int64) *big.Float { return nf().SetInt64(v) }
	tprev := []*big.Float{one(1)}
	tcur := []*big.Float{one(0), one(1)}
	for jj := 0; jj <= n; jj++ {
		var t []*big.Float
		switch jj {
		case 0:
			t = tprev
		case 1:
			t = tcur
		default:
			t = []*big.Float{one(0)}
			for _, v := range tcur {
				t = append(t, mul(big.NewFloat(2), v))
			}
			for i, v := range tprev {
				t[i] = nf().Sub(t[i], v)
			}
			tprev, tcur = tcur, t
		}
		for i, v := range t {
			mono[i] = nf().Add(mono[i], mul(cc[jj], v))
		}
	}
	co := make([]*big.Float, n+1)
	hp := nf().SetInt64(1)
	for i := range co {
		co[i] = nf().Quo(mono[i], hp) // h is a power of two here: h**i exact
		hp = mul(hp, h)
	}
	return c, co
}

func fit(j job, a, b float64) (*big.Float, []*big.Float, int, *big.Float) {
	tol := new(big.Float).SetMantExp(big.NewFloat(1), -64)
	af, bf := nf().SetFloat64(a), nf().SetFloat64(b)
	ts := make([]*big.Float, 401)
	for k := range ts {
		d := nf().Mul(nf().Sub(bf, af), big.NewFloat(float64(k)))
		ts[k] = nf().Add(af, nf().Quo(d, big.NewFloat(400)))
	}
	fvs := eval(j, ts)
	for n := 4; n < 40; n++ {
		c, co := chebInterp(j, a, b, n)
		worst := nf()
		for k, t := range ts {
			s := nf().Sub(t, c)
			// p = Σ co[i] s^i, summed at a higher precision (fit.py's
			// fsum of mp powers agrees to far below the 3 digits printed)
			p := new(big.Float).SetPrec(4 * prec)
			pw := new(big.Float).SetPrec(4 * prec).SetInt64(1)
			for _, ci := range co {
				p.Add(p, new(big.Float).SetPrec(4*prec).Mul(ci, pw))
				pw.Mul(pw, s)
			}
			e := nf().Quo(nf().Sub(nf().Set(p), fvs[k]), fvs[k])
			if e.Abs(e).Cmp(worst) > 0 {
				worst = e
			}
		}
		if worst.Cmp(tol) < 0 {
			return c, co, n, worst
		}
	}
	log.Fatalf("%s: no convergence on [%v, %v]", j.name, a, b)
	return nil, nil, 0, nil
}

func main() {
	var err error
	if o, err = mporacle.Start(); err != nil {
		log.Fatal(err)
	}
	defer o.Close()
	want := map[string]bool{}
	for _, a := range os.Args[1:] {
		want[a] = true
	}
	r := mporacle.Repr
	for _, j := range jobs {
		if len(want) > 0 && !want[j.name] {
			continue
		}
		var b bytes.Buffer
		w := func(format string, a ...any) { fmt.Fprintf(&b, format+"\n", a...) }
		w("// Code generated by go -C tools run ./genfit %s; DO NOT EDIT.", j.name)
		w("//")
		w("// %s", j.doc)
		w("// Chebyshev interpolants converted to monomials in s = t - c;")
		w("// interpolation error < 2^-64 relative on each piece, checked")
		w("// against mpmath %s at 200 bits.", mporacle.MpmathVersion)
		w("")
		w("package special")
		w("")
		w("var %s = [...]polyPiece{", j.name)
		for i := 0; i+1 < len(j.edges); i++ {
			a, bb := j.edges[i], j.edges[i+1]
			c, co, n, worst := fit(j, a, bb)
			ws := o.String(prec, "mpmath.nstr(w, 3)", map[string]*big.Float{"w": worst})
			w("\t// [%s, %s]: degree %d, max interpolation error %s", r(a), r(bb), n, ws)
			c0lo := mporacle.Float64(nf().Sub(co[0], nf().SetFloat64(mporacle.Float64(co[0]))))
			c1lo := mporacle.Float64(nf().Sub(co[1], nf().SetFloat64(mporacle.Float64(co[1]))))
			w("\t{lo: %s, hi: %s, c: %s, c0lo: %s, c1lo: %s, coef: []float64{", r(a), r(bb), r(mporacle.Float64(c)), r(c0lo), r(c1lo))
			for k := len(co) - 1; k >= 0; k-- {
				w("\t\t%s,", r(mporacle.Float64(co[k])))
			}
			w("\t}},")
		}
		w("}")
		if err := os.WriteFile("../zz_"+j.name+"_coeffs.go", b.Bytes(), 0o644); err != nil {
			log.Fatal(err)
		}
		fmt.Fprintln(os.Stderr, j.name, "written")
	}
}
