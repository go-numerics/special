package main

import (
	"math/big"
	"sort"
)

// Arithmetic on big.Float that follows mpmath's: every operation rounds
// its exact result to the working precision of the context (round to
// nearest), as mpf arithmetic does at mp.prec; fsum adds exactly and
// rounds once; integer powers are formed exactly and rounded once.
type ctx struct{ prec uint }

func (c ctx) f(x float64) *big.Float { return new(big.Float).SetPrec(53).SetFloat64(x) }
func (c ctx) i(x int64) *big.Float   { return new(big.Float).SetPrec(64).SetInt64(x) }

func (c ctx) add(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(c.prec).Add(x, y) }
func (c ctx) sub(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(c.prec).Sub(x, y) }
func (c ctx) mul(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(c.prec).Mul(x, y) }
func (c ctx) quo(x, y *big.Float) *big.Float { return new(big.Float).SetPrec(c.prec).Quo(x, y) }
func (c ctx) neg(x *big.Float) *big.Float    { return new(big.Float).SetPrec(c.prec).Neg(x) }
func (c ctx) abs(x *big.Float) *big.Float    { return new(big.Float).SetPrec(c.prec).Abs(x) }

// pow returns x^n (n >= 0) rounded once.
func (c ctx) pow(x *big.Float, n int) *big.Float {
	p := uint(n)*x.Prec() + 64
	r := new(big.Float).SetPrec(p).SetInt64(1)
	for k := 0; k < n; k++ {
		r.Mul(r, x)
	}
	return new(big.Float).SetPrec(c.prec).Set(r)
}

// fsum returns the exactly rounded sum.
func (c ctx) fsum(xs []*big.Float) *big.Float {
	if len(xs) == 0 {
		return new(big.Float).SetPrec(c.prec)
	}
	// exact: enough bits for the spread of exponents
	minE, maxE := 1<<30, -(1 << 30)
	maxP := uint(0)
	for _, x := range xs {
		if x.Sign() == 0 {
			continue
		}
		e := x.MantExp(nil)
		minE = min(minE, e-int(x.Prec()))
		maxE = max(maxE, e)
		maxP = max(maxP, x.Prec())
	}
	if maxE < minE {
		return new(big.Float).SetPrec(c.prec)
	}
	s := new(big.Float).SetPrec(uint(maxE-minE) + 64 + maxP)
	for _, x := range xs {
		s.Add(s, x)
	}
	return new(big.Float).SetPrec(c.prec).Set(s)
}

func fl(x *big.Float) float64 { f, _ := x.Float64(); return f }

// key is an exact text of x, for map keys.
func key(x *big.Float) string { return x.Text('p', 0) }

func sortPts(p []pt) {
	sort.SliceStable(p, func(i, j int) bool {
		if d := p[i].z.Cmp(p[j].z); d != 0 {
			return d < 0
		}
		return !p[i].isz && p[j].isz
	})
}

// chebNodes returns cos(π(k+1/2)/N), k < N, as mpmath computes them at
// prec: mp.pi rounded to prec, times (k+1/2), over N, cosine.
func chebCos(o *oracle, c ctx, pi *big.Float, n int, j int) []req {
	rs := make([]req, 0, n)
	for k := 0; k < n; k++ {
		t := c.mul(pi, c.i(int64(j)))
		if j == 1 {
			t = pi // mp.pi * 1 is mp.pi
		}
		t = c.mul(t, c.f(float64(k)+0.5))
		t = c.quo(t, c.i(int64(n)))
		rs = append(rs, req{"cos", c.prec, []any{t}})
	}
	return rs
}
