package special

import (
	"math"
	"math/big"
	"testing"
)

// A math/big oracle (320 bits) for the log-sum-exp family, independent of every
// code path of the package (after the independent review of the fast path:
// the double-double code it replaced shares log1pDD and dd.ExpMant with the
// fallback, so it cannot judge them).

const lsePrec = 320

func lseF(v float64) *big.Float { return new(big.Float).SetPrec(lsePrec).SetFloat64(v) }

var lseLn2 = func() *big.Float {
	s := new(big.Float).SetPrec(lsePrec)
	for k := 1; k < 400; k++ { // ln 2 = Σ 1/(k 2^k)
		t := new(big.Float).SetPrec(lsePrec).SetInt64(1)
		t.SetMantExp(t, -k)
		t.Quo(t, new(big.Float).SetPrec(lsePrec).SetInt64(int64(k)))
		s.Add(s, t)
	}
	return s
}()

// lseExp returns exp(x) for a big x of magnitude up to 1e4.
func lseExp(x *big.Float) *big.Float {
	if x.Sign() == 0 {
		return new(big.Float).SetPrec(lsePrec).SetInt64(1)
	}
	q := new(big.Float).SetPrec(lsePrec).Quo(x, lseLn2)
	qf, _ := q.Float64()
	k := math.Round(qf)
	r := new(big.Float).SetPrec(lsePrec).Mul(lseLn2, lseF(k))
	r.Sub(x, r)
	const sq = 30
	r.SetMantExp(r, -sq)
	sum := new(big.Float).SetPrec(lsePrec).SetInt64(1)
	term := new(big.Float).SetPrec(lsePrec).SetInt64(1)
	for i := 1; i < 40; i++ {
		term.Mul(term, r)
		term.Quo(term, new(big.Float).SetPrec(lsePrec).SetInt64(int64(i)))
		sum.Add(sum, term)
	}
	for i := 0; i < sq; i++ {
		sum.Mul(sum, sum)
	}
	return sum.SetMantExp(sum, int(k))
}

// lseLog1p returns log(1+s) for s >= 0.
func lseLog1p(s *big.Float) *big.Float {
	if s.Sign() == 0 {
		return new(big.Float).SetPrec(lsePrec)
	}
	if s.MantExp(nil) < -40 {
		sum := new(big.Float).SetPrec(lsePrec)
		p := new(big.Float).SetPrec(lsePrec).SetInt64(1)
		for k := 1; k < 12; k++ {
			p.Mul(p, s)
			t := new(big.Float).SetPrec(lsePrec).Quo(p, new(big.Float).SetPrec(lsePrec).SetInt64(int64(k)))
			if k%2 == 0 {
				sum.Sub(sum, t)
			} else {
				sum.Add(sum, t)
			}
		}
		return sum
	}
	a := new(big.Float).SetPrec(lsePrec).SetInt64(1)
	a.Add(a, s)
	af, _ := a.Float64()
	y := lseF(math.Log(af))
	for it := 0; it < 8; it++ { // Halley on exp(y) = a
		e := lseExp(y)
		num := new(big.Float).SetPrec(lsePrec).Sub(a, e)
		den := new(big.Float).SetPrec(lsePrec).Add(a, e)
		num.Quo(num, den)
		num.SetMantExp(num, 1)
		y.Add(y, num)
	}
	return y
}

type lseOut struct {
	sm, lsm []*big.Float // nil where x_i = -Inf (LogSoftmax)
	lse, l  *big.Float   // lse and L = log Σ exp(x_i - m)
}

// lseOracle evaluates the three functions at finite max x, exp memoised by value.
func lseOracle(x []float64) lseOut {
	m := floatMax(x)
	bm := lseF(m)
	memo := map[float64]*big.Float{}
	diff := func(v float64) *big.Float { return new(big.Float).SetPrec(lsePrec).Sub(lseF(v), bm) }
	ex := func(v float64) *big.Float {
		if e, ok := memo[v]; ok {
			return e
		}
		e := new(big.Float).SetPrec(lsePrec)
		if d := diff(v); !math.IsInf(v, -1) {
			if df, _ := d.Float64(); df >= -1e5 {
				e = lseExp(d)
			}
		}
		memo[v] = e
		return e
	}
	sp := new(big.Float).SetPrec(lsePrec) // Σ over x_i < m, plus ties - 1
	ties := 0
	for _, v := range x {
		if v == m {
			ties++
			continue
		}
		sp.Add(sp, ex(v))
	}
	sp.Add(sp, new(big.Float).SetPrec(lsePrec).SetInt64(int64(ties-1)))
	var o lseOut
	o.l = lseLog1p(sp)
	S := new(big.Float).SetPrec(lsePrec).Add(sp, new(big.Float).SetPrec(lsePrec).SetInt64(1))
	o.lse = new(big.Float).SetPrec(lsePrec).Add(bm, o.l)
	o.sm = make([]*big.Float, len(x))
	o.lsm = make([]*big.Float, len(x))
	for i, v := range x {
		o.sm[i] = new(big.Float).SetPrec(lsePrec).Quo(ex(v), S)
		if !math.IsInf(v, -1) {
			o.lsm[i] = new(big.Float).SetPrec(lsePrec).Sub(diff(v), o.l)
		}
	}
	return o
}

// lseErr returns |got - want| as a float64 (absolute) and in units of u
// relative to |want|.
func lseErr(got float64, want *big.Float) (abs, relU float64) {
	g := lseF(got)
	g.Sub(g, want)
	g.Abs(g)
	abs, _ = g.Float64()
	if want.Sign() == 0 {
		return abs, math.Inf(1)
	}
	g.Quo(g, new(big.Float).SetPrec(lsePrec).Abs(want))
	f, _ := g.Float64()
	return abs, math.Ldexp(f, 53)
}

// A maximum and a long constant tail of terms below 2^-53 (x_i - m < -36.7):
// the sum of the others is a long sum of tiny terms. The Fast2Sum of the first
// version, with the maximum left out, added them plain: 6.09u at n = 1000,
// 32.66u at n = 2e4 (the independent review), against 5u and 4u.
func TestTinyTailBig(t *testing.T) {
	for _, tc := range []struct {
		n int
		c float64
	}{{1000, -40}, {20000, -41.59815236796009}, {20000, -37}, {10000, -50}, {50000, -41.59815236796009}} {
		x := make([]float64, tc.n)
		for i := 1; i < tc.n; i++ {
			x[i] = tc.c
		}
		o := lseOracle(x)
		_, ul := lseErr(LogSumExp(x), o.lse)
		_, us := lseErr(LogSoftmax(nil, x)[0], o.lsm[0])
		t.Logf("n=%d c=%v: LogSumExp %.2f u, LogSoftmax[0] %.2f u", tc.n, tc.c, ul, us)
		if _, u := lseErr(LogSumExp(x), o.lse); u > inU(logSumExpErr) {
			t.Errorf("n=%d c=%v: LogSumExp %.2f u > %.0f u", tc.n, tc.c, u, inU(logSumExpErr))
		}
		if _, u := lseErr(LogSoftmax(nil, x)[0], o.lsm[0]); u > inU(logSoftmaxErr) {
			t.Errorf("n=%d c=%v: LogSoftmax[0] %.2f u > %.0f u", tc.n, tc.c, u, inU(logSoftmaxErr))
		}
		if _, u := lseErr(LogSoftmax(nil, x)[1], o.lsm[1]); u > inU(logSoftmaxErr) {
			t.Errorf("n=%d c=%v: LogSoftmax[1] %.2f u > %.0f u", tc.n, tc.c, u, inU(logSoftmaxErr))
		}
		if _, u := lseErr(Softmax(nil, x)[1], o.sm[1]); u > inU(softmaxErr) {
			t.Errorf("n=%d c=%v: Softmax[1] %.2f u > %.0f u", tc.n, tc.c, u, inU(softmaxErr))
		}
	}
}

// Where m + L cancels the double-double path answers; its error is stated in
// absolute terms: 2^-84 L + 2^-53 |lse| (log1pDD is within 2^-85.6 relative
// there; the relative error of LogSumExp is not bounded: 4.5e6 u at
// x = [-ln 2, -ln 2]).
func TestLogSumExpCancellationBig(t *testing.T) {
	for _, x := range [][]float64{
		{-math.Ln2, -math.Ln2}, {-1, -1}, {-0.5, -0.6}, {-1, -1, -1, -1}, {-0.7, -0.7, -50},
		{-1.3862943611198906, -1.3862943611198906, -1.3862943611198906, -1.3862943611198906},
	} {
		o := lseOracle(x)
		l, _ := o.l.Float64()
		lse, _ := o.lse.Float64()
		bound := math.Ldexp(l, -84) + math.Ldexp(math.Abs(lse), -53)
		for name, got := range map[string]float64{"LogSumExp": LogSumExp(x), "LogSumExpExact": LogSumExpExact(x)} {
			if abs, _ := lseErr(got, o.lse); abs > bound {
				t.Errorf("%s(%v) = %v: absolute error %g > %g", name, x, got, abs, bound)
			}
		}
	}
}
