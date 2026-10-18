// The octave, cell and Temme-sum tables (formerly the separate command
// testdata/gen/fitgo, now jobs of genfit on the shared mporacle): the
// octave pieces of the complete elliptic integrals and of I0 e^-x and
// K0 e^x, the cells of J0, J1, Y0, Y1, and the straight-line sum of
// Temme's expansion of the incomplete gamma ratios.
//
//	go -C tools run ./genfit ellip-oct           (../zz_ellipOct_coeffs.go)
//	go -C tools run ./genfit octaves besselI0eOct (../zz_besselI0eOct_coeffs.go)
//	go -C tools run ./genfit octaves besselK0eOct (../zz_besselK0eOct_coeffs.go)
//	go -C tools run ./genfit bessel-cells        (../zz_besselCells_coeffs.go)
//	go -C tools run ./genfit temme-sum           (../zz_igamTemmeSum_gen.go)
//
// The arithmetic is big.Float at the precisions of the method, rounding
// as mpf arithmetic does; mpmath is the oracle for the function values,
// the zeros, π and the cosines (oct_oracle.go, through mporacle). The
// output is gofmt-formatted.
//
// The methods. Octave pieces: each octave [2^e, 2^(e+1)) of the variable
// (p = 1 - m for the elliptic integrals) is cut into 8 pieces of equal
// width; on each, the function is interpolated at Chebyshev nodes of the
// first kind at 200 bits with a fixed degree (near-minimax: within a
// factor 1 + (2/π) log n of the best uniform approximation, Rivlin,
// "Chebyshev Polynomials", 1990) on the piece widened by 2^-20 of its
// width, converted to monomials in s = x - c (c the centre, an exact
// float64), and checked on 201 points against mpmath; the worst relative
// interpolation error is printed in the header. The constant term is also
// emitted as a double-double (c0lo).
//
// Bessel cells: the expansion points are the zeros of the function, the
// zeros of its derivative (J. Harrison, ARITH 19, 2009) and the midpoints
// between them; on each cell f(z+h) = a0 + a1 h + h² R(h) with a0, a1 (and
// z at a zero of f) double-doubles and R a degree-11 Chebyshev interpolant
// in h, accepted where |h² ΔR| <= 2^-62 |f| on 81 points, the cell split
// otherwise (only near Y's logarithmic singularity at 0).
//
// Temme sum: reads igamTemmeCoef from incgamma_temme_coeffs.go and emits
// Σ_k C_k(η) u^k for |η| <= 0.34, a >= 20, each row cut at the degree where
// the terms left out sum to less than 2^-56 · 0.3 · 20^k and the rows cut
// where their bound falls below the same, every polynomial by Estrin's
// scheme.

package main

import (
	"bytes"
	"fmt"
	"go/format"
	"math/big"
	"os"
	"strings"
)

var out bytes.Buffer

func p(format string, args ...any) { fmt.Fprintf(&out, format+"\n", args...) }

// octJob runs one of the jobs above if args names one, writing its file,
// and reports whether it did.
func octJob(args []string) bool {
	var file string
	switch args[0] {
	case "ellip-oct":
		ellipOct()
		file = "zz_ellipOct_coeffs.go"
	case "octaves":
		if len(args) < 2 {
			check(fmt.Errorf("usage: genfit octaves besselI0eOct|besselK0eOct"))
		}
		octaves(args[1])
		file = "zz_" + args[1] + "_coeffs.go"
	case "bessel-cells":
		besselCells()
		file = "zz_besselCells_coeffs.go"
	case "temme-sum":
		temmeSum()
		file = "zz_igamTemmeSum_gen.go"
	default:
		return false
	}
	src, err := format.Source(out.Bytes())
	check(err)
	check(os.WriteFile("../"+file, src, 0o644))
	out.Reset()
	return true
}

// generatedBy is the command named in the header of a generated file.
func generatedBy(job string) string { return "go -C tools run ./genfit " + job }

var c200 = ctx{200}

// chebInterpOct is fit.cheb_interp: the coefficients, as monomials in
// s = t - c, of the degree-n Chebyshev interpolant of f on [a, b].
func chebInterpOct(o *oracle, f func([]*big.Float) []*big.Float, a, b *big.Float, n int) (*big.Float, []*big.Float) {
	c := c200
	cm := c.quo(c.add(a, b), c.i(2))
	h := c.quo(c.sub(b, a), c.i(2))
	pi := o.one("pi", c.prec)
	nodes := o.eval(chebCos(o, c, pi, n+1, 1))
	xs := make([]*big.Float, n+1)
	for k, t := range nodes {
		xs[k] = c.add(cm, c.mul(h, t))
	}
	vals := f(xs)
	mono := chebMono(o, c, pi, vals, n)
	co := make([]*big.Float, n+1)
	for i := range mono {
		co[i] = c.quo(mono[i], c.pow(h, i))
	}
	return cm, co
}

// chebMono returns the Chebyshev coefficients of the interpolant through
// vals at the N = n+1 nodes, converted to monomials in t in [-1, 1].
func chebMono(o *oracle, c ctx, pi *big.Float, vals []*big.Float, n int) []*big.Float {
	N := n + 1
	var rs []req
	for j := 0; j < N; j++ {
		rs = append(rs, chebCos(o, c, pi, N, j)...)
	}
	cs := o.eval(rs)
	cc := make([]*big.Float, N)
	for j := 0; j < N; j++ {
		terms := make([]*big.Float, N)
		for k := 0; k < N; k++ {
			terms[k] = c.mul(vals[k], cs[j*N+k])
		}
		s := c.fsum(terms)
		cc[j] = c.quo(c.mul(c.i(2), s), c.i(int64(N)))
	}
	cc[0] = c.quo(cc[0], c.i(2))
	mono := make([]*big.Float, N)
	for i := range mono {
		mono[i] = new(big.Float).SetPrec(c.prec)
	}
	Tp := []*big.Float{c.i(1)}
	Tc := []*big.Float{c.i(0), c.i(1)}
	for j := 0; j < N; j++ {
		var T []*big.Float
		switch j {
		case 0:
			T = Tp
		case 1:
			T = Tc
		default:
			T = []*big.Float{c.i(0)}
			for _, v := range Tc {
				T = append(T, c.mul(c.i(2), v))
			}
			for i, v := range Tp {
				T[i] = c.sub(T[i], v)
			}
			Tp, Tc = Tc, T
		}
		for i, v := range T {
			mono[i] = c.add(mono[i], c.mul(cc[j], v))
		}
	}
	return mono
}

// worstRel returns max |(p(t) - f(t))/f(t)| over the 201 points
// a + (b-a)k/200, p the monomial polynomial in t - cm.
func worstRel(o *oracle, f func([]*big.Float) []*big.Float, a, b, cm *big.Float, co []*big.Float, worst *big.Float) *big.Float {
	c := c200
	ts := make([]*big.Float, 201)
	for k := range ts {
		ts[k] = c.add(a, c.quo(c.mul(c.sub(b, a), c.i(int64(k))), c.i(200)))
	}
	fv := f(ts)
	for k, t := range ts {
		s := c.sub(t, cm)
		terms := make([]*big.Float, len(co))
		for i := range co {
			terms[i] = c.mul(co[i], c.pow(s, i))
		}
		pv := c.fsum(terms)
		e := c.abs(c.quo(c.sub(pv, fv[k]), fv[k]))
		if e.Cmp(worst) > 0 {
			worst = e
		}
	}
	return worst
}

func evalFn(o *oracle, op string, prec uint) func([]*big.Float) []*big.Float {
	return func(xs []*big.Float) []*big.Float {
		rs := make([]req, len(xs))
		for i, x := range xs {
			rs[i] = req{op, prec, []any{x}}
		}
		return o.eval(rs)
	}
}

type piece struct {
	e, j int
	c    *big.Float
	co   []*big.Float
}

func emitPiece(pc piece, comment string) {
	c := c200
	c0 := pc.co[0]
	c0lo := fl(c.sub(c0, c.f(fl(c0))))
	p("\t// %s", comment)
	p("\t{c: %s, c0lo: %s, coef: [...]float64{", pyRepr(fl(pc.c)), pyRepr(c0lo))
	vs := make([]string, len(pc.co))
	for i := range pc.co {
		vs[i] = pyRepr(fl(pc.co[len(pc.co)-1-i]))
	}
	p("\t\t%s,", strings.Join(vs, ", "))
	p("\t}},")
}

// ellipOct (formerly fit_ellip_oct.py) emits the elliptic octave tables.
func ellipOct() {
	o := newOracle()
	const deg, emin, emax, nsub = 11, -10, 10, 8
	p("// Code generated by %s; DO NOT EDIT.", generatedBy("ellip-oct"))
	p("//")
	p("// Degree-%d interpolants of K(m) and E(m) on %d pieces per octave of", deg, nsub)
	p("// p = 1 - m, p in [2^%d, 2^%d); mpmath %s at 200 bits.", emin, emax+1, mpmathVersion)
	p("")
	p("package special")
	p("")
	for _, job := range []struct{ name, op, doc string }{
		{"ellipKOct", "ellipk", "K(m) = sum coef[i] s^(11-i), s = m - c."},
		{"ellipEOct", "ellipe", "E(m) = sum coef[i] s^(11-i), s = m - c."},
	} {
		c := c200
		f := evalFn(o, job.op, 200)
		worst := new(big.Float).SetPrec(200)
		var ps []piece
		for e := emin; e <= emax; e++ {
			p0 := new(big.Float).SetPrec(200).SetMantExp(c.i(1), e)
			for j := 0; j < nsub; j++ {
				plo := c.mul(p0, c.add(c.i(1), c.quo(c.i(int64(j)), c.i(nsub))))
				phi := c.mul(p0, c.add(c.i(1), c.quo(c.i(int64(j+1)), c.i(nsub))))
				pc := c.quo(c.add(plo, phi), c.i(2))
				cm := c.sub(c.i(1), pc)
				a, b := c.sub(c.i(1), phi), c.sub(c.i(1), plo)
				pad := c.mul(c.sub(b, a), new(big.Float).SetMantExp(c.i(1), -20))
				cc, co := chebInterpOct(o, f, c.sub(a, pad), c.add(b, pad), deg)
				if cc.Cmp(cm) != 0 {
					check(fmt.Errorf("centre"))
				}
				worst = worstRel(o, f, a, b, cm, co, worst)
				ps = append(ps, piece{e, j, cm, co})
			}
		}
		p("// %s: %s", job.name, job.doc)
		p("// Max interpolation error over all pieces: %s relative.", nstr3(worst))
		p("var %s = [...]octPiece{", job.name)
		for _, pc := range ps {
			emitPiece(pc, fmt.Sprintf("p in 2^%d * [%d/8, %d/8]", pc.e, 8+pc.j, 9+pc.j))
		}
		p("}")
		p("")
	}
}

// octaves (formerly fit_octaves.py) emits one octave table.
func octaves(name string) {
	jobs := map[string]struct {
		doc                  string
		op                   string
		emin, emax, nsub, dg int
	}{
		"besselI0eOct": {"I0(x) e^-x for x in [2^-3, 2^9).", "i0e", -3, 8, 8, 11},
		"besselK0eOct": {"K0(x) e^x for x in [1, 2^9).", "k0e", 0, 8, 8, 11},
	}
	j, ok := jobs[name]
	if !ok {
		check(fmt.Errorf("unknown table %q", name))
	}
	o := newOracle()
	c := c200
	f := evalFn(o, j.op, 200) // the oracle adds the 40 guard bits
	worst := new(big.Float).SetPrec(200)
	var ps []piece
	for e := j.emin; e <= j.emax; e++ {
		x0 := new(big.Float).SetPrec(200).SetMantExp(c.i(1), e)
		for k := 0; k < j.nsub; k++ {
			a := c.mul(x0, c.add(c.i(1), c.quo(c.i(int64(k)), c.i(int64(j.nsub)))))
			b := c.mul(x0, c.add(c.i(1), c.quo(c.i(int64(k+1)), c.i(int64(j.nsub)))))
			pad := c.mul(c.sub(b, a), new(big.Float).SetMantExp(c.i(1), -20))
			cm, co := chebInterpOct(o, f, c.sub(a, pad), c.add(b, pad), j.dg)
			if cm.Cmp(c.quo(c.add(a, b), c.i(2))) != 0 {
				check(fmt.Errorf("centre"))
			}
			worst = worstRel(o, f, a, b, cm, co, worst)
			ps = append(ps, piece{e, k, cm, co})
		}
	}
	if worst.Cmp(new(big.Float).SetMantExp(c.i(1), -60)) > 0 {
		check(fmt.Errorf("%s: interpolation error %s", name, nstr3(worst)))
	}
	p("// Code generated by %s %s; DO NOT EDIT.", generatedBy("octaves"), name)
	p("//")
	p("// %s: %s", name, j.doc)
	p("// Degree %d on %d pieces per octave of x in [2^%d, 2^%d), in s = x - c;", j.dg, j.nsub, j.emin, j.emax+1)
	p("// max interpolation error %s relative (mpmath %s, 200 bits).", nstr3(worst), mpmathVersion)
	p("")
	p("package special")
	p("")
	p("var %s = [...]octPiece{", name)
	for _, pc := range ps {
		emitPiece(pc, fmt.Sprintf("x in 2^%d * [%d/%d, %d/%d]", pc.e, j.nsub+pc.j, j.nsub, j.nsub+pc.j+1, j.nsub))
	}
	p("}")
}
