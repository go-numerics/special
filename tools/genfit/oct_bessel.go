package main

import (
	"fmt"
	"math/big"
	"strings"
)

// besselCells (formerly fit_bessel_cells.py): J0, J1, Y0, Y1 below x = 100 as one
// polynomial per cell (see the package comment).

const (
	cellDeg   = 11
	cellXMax  = 100
	lutScale  = 8
	cellExtra = 100 // guard bits of R's evaluation
)

type pt struct {
	z   *big.Float
	isz bool
}

type cellRes struct {
	a0, a1 *big.Float
	co     []*big.Float
	worst  *big.Float
}

type bessel struct {
	o    *oracle
	kind string // "J" or "Y"
	n    int
}

func (b bessel) op() string {
	if b.kind == "J" {
		return "besselj"
	}
	return "bessely"
}

// f returns f^(d)(x) at prec for each x.
func (b bessel) f(xs []*big.Float, d int, prec uint) []*big.Float {
	rs := make([]req, len(xs))
	for i, x := range xs {
		rs[i] = req{b.op(), prec, []any{b.n, x, d}}
	}
	return b.o.eval(rs)
}

func (b bessel) zeros(d int) []*big.Float {
	op := "besseljzero"
	if b.kind == "Y" {
		op = "besselyzero"
	}
	var zs []*big.Float
	lim := c200.i(cellXMax + 4)
	for k := 1; ; k++ {
		z := b.o.one(op, 200, b.n, k, d)
		if z.Cmp(lim) > 0 {
			return zs
		}
		zs = append(zs, z)
	}
}

// cell returns a0, a1, the coefficients of R in powers of h and the worst
// |h² ΔR|/|f| on the cell [l, r] about z.
func (b bessel) cell(z *big.Float, isz bool, l, r *big.Float) cellRes {
	c := c200
	hi := ctx{c.prec + cellExtra}
	var a0 *big.Float
	if isz {
		a0 = new(big.Float).SetPrec(200)
	} else {
		a0 = b.f([]*big.Float{z}, 0, 200)[0]
	}
	a1 := b.f([]*big.Float{z}, 1, 200)[0]
	// R(h) = (f(z+h) - a0 - a1 h)/h², with mp.workprec(prec + 100)
	R := func(hs []*big.Float) []*big.Float {
		xs := make([]*big.Float, len(hs))
		for i, h := range hs {
			if h.Sign() == 0 {
				check(fmt.Errorf("R(0) requested"))
			}
			xs[i] = hi.add(z, h)
		}
		fs := b.f(xs, 0, hi.prec)
		out := make([]*big.Float, len(hs))
		for i, h := range hs {
			v := hi.sub(hi.sub(fs[i], a0), hi.mul(a1, h))
			out[i] = hi.quo(v, hi.pow(h, 2))
		}
		return out
	}
	co := b.chebH(R, c.sub(l, z), c.sub(r, z))
	worst := new(big.Float).SetPrec(200)
	var xs, hs []*big.Float
	for k := 0; k <= 80; k++ {
		x := c.add(l, c.quo(c.mul(c.sub(r, l), c.i(int64(k))), c.i(80)))
		h := c.sub(x, z)
		if h.Sign() == 0 {
			continue
		}
		xs, hs = append(xs, x), append(hs, h)
	}
	fv := b.f(xs, 0, 200)
	rv := R(hs)
	for i, h := range hs {
		if fv[i].Sign() == 0 {
			continue
		}
		terms := make([]*big.Float, len(co))
		for k := range co {
			terms[k] = c.mul(co[k], c.pow(h, k))
		}
		pv := c.fsum(terms)
		e := c.abs(c.quo(c.mul(c.mul(h, h), c.sub(pv, rv[i])), fv[i]))
		if e.Cmp(worst) > 0 {
			worst = e
		}
	}
	return cellRes{a0, a1, co, worst}
}

// chebH is cheb_h: the Chebyshev interpolant of R on [hl, hr] as monomials
// in h.
func (b bessel) chebH(R func([]*big.Float) []*big.Float, hl, hr *big.Float) []*big.Float {
	c := c200
	cm := c.quo(c.add(hl, hr), c.i(2))
	w := c.quo(c.sub(hr, hl), c.i(2))
	N := cellDeg + 1
	pi := b.o.one("pi", c.prec)
	nodes := b.o.eval(chebCos(b.o, c, pi, N, 1))
	hs := make([]*big.Float, N)
	for k, t := range nodes {
		hs[k] = c.add(cm, c.mul(w, t))
	}
	mono := chebMono(b.o, c, pi, R(hs), cellDeg)
	out := make([]*big.Float, N)
	for i := range out {
		out[i] = new(big.Float).SetPrec(c.prec)
	}
	ncm := c.neg(cm)
	for i, v := range mono {
		for k := 0; k <= i; k++ {
			t := c.mul(v, binom(i, k))
			t = c.mul(t, c.pow(ncm, i-k))
			t = c.quo(t, c.pow(w, i))
			out[k] = c.add(out[k], t)
		}
	}
	return out
}

func binom(n, k int) *big.Float {
	return new(big.Float).SetInt(new(big.Int).Binomial(int64(n), int64(k)))
}

type cellOut struct {
	z    *big.Float
	isz  bool
	l, r *big.Float
	cellRes
}

func (b bessel) build(xmin float64) []cellOut {
	c := c200
	var sp []pt
	for d, isz := range []bool{true, false} {
		for _, z := range b.zeros(d) {
			if !isz {
				z = c.f(fl(z))
			}
			sp = append(sp, pt{z, isz})
		}
	}
	sortPts(sp)
	if b.kind == "J" {
		sp = append([]pt{{new(big.Float).SetPrec(200), b.n == 1}}, sp...)
	}
	var pts []pt
	for i := 0; i+1 < len(sp); i++ {
		pts = append(pts, sp[i])
		pts = append(pts, pt{c.f(fl(c.quo(c.add(sp[i].z, sp[i+1].z), c.i(2)))), false})
	}
	pts = append(pts, sp[len(sp)-1])
	cache := map[string]cellRes{}
	xminF, xmaxF := c.f(xmin), c.i(cellXMax)
	for {
		var out []cellOut
		split := -1
		var sl, sr *big.Float
		for i, q := range pts {
			z := q.z
			var l, r *big.Float
			if i == 0 {
				l = xminF
			} else {
				l = c.f(fl(c.quo(c.add(pts[i-1].z, z), c.i(2))))
			}
			if i+1 < len(pts) {
				r = c.f(fl(c.quo(c.add(z, pts[i+1].z), c.i(2))))
			} else {
				r = xmaxF
			}
			if l.Cmp(xminF) < 0 {
				l = xminF
			}
			if r.Cmp(xmaxF) > 0 {
				r = xmaxF
			}
			if r.Cmp(xminF) <= 0 || l.Cmp(xmaxF) >= 0 {
				continue
			}
			if z.Sign() != 0 && (l.Cmp(c.quo(z, c.i(2))) < 0 || r.Cmp(c.mul(z, c.i(2))) > 0) {
				check(fmt.Errorf("cell [%v, %v] about %v: h = x - z not exact", l, r, z))
			}
			k := key(z) + "|" + key(l) + "|" + key(r)
			res, ok := cache[k]
			if !ok {
				res = b.cell(z, q.isz, l, r)
				cache[k] = res
			}
			if res.worst.Cmp(new(big.Float).SetMantExp(c.i(1), -62)) > 0 {
				split, sl, sr = i, l, r
				break
			}
			out = append(out, cellOut{z, q.isz, l, r, res})
		}
		if split < 0 {
			return out
		}
		q := pts[split]
		pl := c.f(fl(c.quo(c.add(sl, q.z), c.i(2))))
		pr := c.f(fl(c.quo(c.add(q.z, sr), c.i(2))))
		var ins []pt
		if pl.Cmp(sl) > 0 {
			ins = append(ins, pt{pl, false})
		}
		ins = append(ins, q)
		if pr.Cmp(sr) < 0 {
			ins = append(ins, pt{pr, false})
		}
		pts = append(pts[:split], append(ins, pts[split+1:]...)...)
	}
}

func ddsplit(v *big.Float) (float64, float64) {
	h := fl(v)
	return h, fl(c200.sub(v, c200.f(h)))
}

func (b bessel) emit(xmin float64) {
	cells := b.build(xmin)
	name := fmt.Sprintf("bessel%s%dCells", b.kind, b.n)
	worst := new(big.Float).SetPrec(200)
	for _, c := range cells {
		if c.worst.Cmp(worst) > 0 {
			worst = c.worst
		}
	}
	p("// %s: %d cells on [%s, %d]; worst |h^2 dR|/|f| = %s.", name, len(cells), pyRepr(xmin), cellXMax, nstr3(worst))
	p("var %s = [...]besselCell{", name)
	for _, c := range cells {
		zh, zl := ddsplit(c.z)
		a0h, a0l := ddsplit(c.a0)
		a1h, a1l := ddsplit(c.a1)
		p("\t{lo: %s, z: %s, zlo: %s, a0: %s, a0lo: %s, a1: %s, a1lo: %s, r: [...]float64{",
			pyRepr(fl(c.l)), pyRepr(zh), pyRepr(zl), pyRepr(a0h), pyRepr(a0l), pyRepr(a1h), pyRepr(a1l))
		vs := make([]string, len(c.co))
		for i := range c.co {
			vs[i] = pyRepr(fl(c.co[len(c.co)-1-i]))
		}
		p("\t\t%s,", strings.Join(vs, ", "))
		p("\t}},")
	}
	p("}")
	p("")
	edges := make([]float64, 0, len(cells)+1)
	for _, c := range cells {
		edges = append(edges, fl(c.l))
	}
	edges = append(edges, cellXMax)
	var lut []int
	for k := 0; float64(k)/lutScale < cellXMax; k++ {
		x := float64(k) / lutScale
		j := 0
		for j+1 < len(cells) && edges[j+1] <= max(x, xmin) {
			j++
		}
		lut = append(lut, j)
	}
	p("var %sLUT = [...]uint16{", name)
	for i := 0; i < len(lut); i += 16 {
		vs := []string{}
		for _, v := range lut[i:min(i+16, len(lut))] {
			vs = append(vs, fmt.Sprint(v))
		}
		p("\t%s,", strings.Join(vs, ", "))
	}
	p("}")
	p("")
}

func besselCells() {
	o := newOracle()
	p("// Code generated by %s; DO NOT EDIT.", generatedBy("bessel-cells"))
	p("//")
	p("// Degree-%d remainders R of f(z+h) = a0 + a1 h + h^2 R(h) per cell, fitted", cellDeg)
	p("// to mpmath %s at 200 bits; lookup tables with %d entries per unit of x.", mpmathVersion, lutScale)
	p("")
	p("package special")
	p("")
	bessel{o, "J", 0}.emit(0.0)
	bessel{o, "J", 1}.emit(0.0)
	bessel{o, "Y", 0}.emit(0.6)
	bessel{o, "Y", 1}.emit(1.5)
}
