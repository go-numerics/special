package special

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/go-numerics/special/dd"
)

// The pieces of kv_temme_fast.go: cosh and sinh(sigma)/sigma, the tail bounds,
// the truncation, the cancellation of the series of K_(u+1), the range edges,
// the overflow rule and the declines.

// bigCoshShc returns cosh(s) and sinh(s)/s for s = hi + lo at 320 bits.
func bigCoshShc(hi, lo float64) (ch, shc *big.Float) {
	ivOracleInit()
	s := bnew().Add(bf(hi), bf(lo))
	e, ei := bigExp(s), bigExp(bnew().Neg(s))
	ch = bnew().Quo(bnew().Add(e, ei), bf(2))
	if s.Sign() == 0 {
		return ch, bf(1)
	}
	shc = bnew().Quo(bnew().Sub(e, ei), bnew().Mul(bf(2), s))
	return ch, shc
}

// TestKvTemmeHyp: kvTemmeHyp against the oracle to 2^-66 relative, over sigma
// from 0 through the switch at 2^-6 (series below, quotients of E above) to
// +-347 (u = 1/2 at x = 2^-1000), and at the switch itself.
func TestKvTemmeHyp(t *testing.T) {
	var ss []float64
	for _, a := range []float64{0, 0x1p-60, 1e-10, 0x1p-20, 0.001, math.Nextafter(kvTemmeSmallSigma, 0), kvTemmeSmallSigma, math.Nextafter(kvTemmeSmallSigma, 1), 0.1, 0.5, 1, 7, 50, 200, 347} {
		ss = append(ss, a, -a)
	}
	r := rand.New(rand.NewSource(24))
	for i := 0; i < 300; i++ {
		ss = append(ss, float64(r.NormFloat64()*math.Exp(float64(r.Float64()*8-6))))
	}
	var wc, ws float64
	for _, s := range ss {
		sig := dd.DD{Hi: s, Lo: float64(s * 0x1p-55)}
		m, k := dd.ExpMant(sig)
		E := dd.Ldexp(m, k)
		ch, shc := kvTemmeHyp(sig, E, dd.Div(dd.F(1), E))
		wch, wshc := bigCoshShc(sig.Hi, sig.Lo)
		ec, es := relDD(ch.Hi, ch.Lo, wch), relDD(shc.Hi, shc.Lo, wshc)
		wc, ws = math.Max(wc, ec), math.Max(ws, es)
		if ec > 0x1p-66 || es > 0x1p-66 {
			t.Errorf("sigma=%g: cosh %.3g, sinh/sigma %.3g relative (limit 2^-66)", s, ec, es)
		}
	}
	t.Logf("%d points: cosh worst 2^%.1f, sinh(sigma)/sigma worst 2^%.1f relative", len(ss), math.Log2(wc), math.Log2(ws))
	// sigma = 0 exactly: 1 and 1
	z := dd.F(1)
	ch, shc := kvTemmeHyp(dd.DD{}, z, z)
	if ch != z || shc != z {
		t.Errorf("sigma = 0: cosh %v, sinh/sigma %v, want 1 and 1", ch, shc)
	}
	// at sigma = 2^-6 exactly the quotient branch answers (held bit for bit:
	// the series differs from it in the last bits)
	sig := dd.F(kvTemmeSmallSigma)
	m, k := dd.ExpMant(sig)
	E := dd.Ldexp(m, k)
	Ei := dd.Div(dd.F(1), E)
	ch, shc = kvTemmeHyp(sig, E, Ei)
	if want := dd.MulF(dd.Add(E, Ei), 0.5); ch != want {
		t.Errorf("sigma = 2^-6: cosh %v, quotient branch %v", ch, want)
	}
	if want := dd.Div(dd.Sub(E, Ei), dd.MulF(sig, 2)); shc != want {
		t.Errorf("sigma = 2^-6: sinh/sigma %v, quotient branch %v", shc, want)
	}
}

// TestKvTemmeTailBound: the bounds of kvTemmeTail are at least the tails
// they bound (computed by running the float64 recurrences to 60 terms), on a
// grid of u, x and k; and not absurdly loose (the ratio is logged).
func TestKvTemmeTailBound(t *testing.T) {
	var loose float64
	for _, u := range []float64{-0.5, -0.4999, -0.45, -0.3, -0.1, -1e-6, 1e-9, 0.0001, 0.1, 0.25, 0.4, 0.4999, 0.5} {
		for _, x := range []float64{1e-8, 1e-4, 0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.4999999999999998} {
			qh := float64(x * x / 4)
			var cs, fs, ps, qs, hs [61]float64
			kvTemmeSteps(u, x, 60, func(k int, c, f, p, q, h float64) {
				cs[k], fs[k], ps[k], qs[k], hs[k] = c, f, p, q, h
			})
			for k := 1; k <= 12; k++ {
				var tf, th float64
				for j := k + 1; j <= 60; j++ {
					tf += float64(cs[j] * fs[j])
					th += math.Abs(float64(cs[j] * hs[j]))
				}
				bf, bh := kvTemmeTail(k, qh, cs[k], fs[k], ps[k], qs[k])
				if tf == 0 {
					continue
				}
				if bf < tf*(1-1e-9) || bh < th*(1-1e-9) {
					t.Errorf("u=%g x=%g k=%d: bound %.4g (f) %.4g (h) below the tail %.4g, %.4g", u, x, k, bf, bh, tf, th)
				}
				loose = math.Max(loose, math.Max(bf/tf, bh/th))
			}
		}
	}
	t.Logf("loosest bound / tail ratio: %.3g", loose)
}

// TestKvTemmeTruncation: the pair at the production tolerance against the same
// loop run to 2^-90, within 2^-62 relative on kvSweepN points.
func TestKvTemmeTruncation(t *testing.T) {
	n := kvSweepN(30000)
	r := rand.New(rand.NewSource(25))
	var worst, wu, wx float64
	for i := 0; i < n; i++ {
		u, x := kvTemmePoint(r, i)
		a0, a1, ok := kvTemmeSeries(u, x, true)
		b0, b1, ok2 := kvTemmeSeriesTol(u, x, true, 0x1p-90)
		if !ok || !ok2 {
			t.Fatalf("u=%g x=%g: production %v, deep %v", u, x, ok, ok2)
		}
		d := math.Abs((a0.Hi-b0.Hi)+(a0.Lo-b0.Lo)) / b0.Hi
		if a1.Hi < kvTemmeBig {
			d = math.Max(d, math.Abs((a1.Hi-b1.Hi)+(a1.Lo-b1.Lo))/b1.Hi)
		}
		if d > worst {
			worst, wu, wx = d, u, x
		}
	}
	t.Logf("%d points: stopped vs 2^-90 worst 2^%.1f at u=%g x=%g", n, math.Log2(worst), wu, wx)
	if worst > 0x1p-62 {
		t.Errorf("truncation %.3g relative (limit 2^-62) at u=%g x=%g", worst, wu, wx)
	}
}

// TestKvTemmeCancellation: sum c_k |h_k| over |sum c_k h_k|, the loss of the
// series of (x/2) K_(u+1) (the header states at most 8), on a grid.
func TestKvTemmeCancellation(t *testing.T) {
	var worst, wu, wx float64
	for _, u := range []float64{-0.5, -0.4999, -0.45, -0.3, -0.1, -1e-6, 1e-9, 0.0001, 0.1, 0.25, 0.4, 0.4999, 0.5} {
		for _, x := range []float64{1e-8, 0.01, 0.1, 0.3, 0.5, 0.8, 1, 1.2, 1.3, 1.4, 1.4999999999999998} {
			var s, a float64
			kvTemmeSteps(u, x, 40, func(k int, c, f, p, q, h float64) {
				s += float64(c * h)
				a += math.Abs(float64(c * h))
			})
			if r := a / math.Abs(s); r > worst {
				worst, wu, wx = r, u, x
			}
		}
	}
	t.Logf("worst cancellation factor %.3g at u=%g x=%g", worst, wu, wx)
	if worst > 8.5 {
		t.Errorf("cancellation factor %.3g > 8.5 at u=%g x=%g", worst, wu, wx)
	}
}

// TestKvTemmeDone: the stopping test at its edges (the comparison of the bound
// with the tolerance, the positivity of the partial sums, the optional second
// series).
func TestKvTemmeDone(t *testing.T) {
	const s = kvTemmeStop
	for _, c := range []struct {
		bf, bh, sf, sh float64
		k1, want       bool
		name           string
	}{
		{s, s, 1, 1, true, true, "bounds exactly at the tolerance"},
		{math.Nextafter(s, 1), s, 1, 1, true, false, "K_u bound just above"},
		{s, math.Nextafter(s, 1), 1, 1, true, false, "K_(u+1) bound just above"},
		{s, math.Nextafter(s, 1), 1, 1, false, true, "K_(u+1) not needed"},
		{0, 0, 0, 1, true, false, "partial sum of K_u zero"},
		{0, 0, -1, 1, true, false, "partial sum of K_u negative"},
		{0, 0, 1, 0, true, false, "partial sum of K_(u+1) zero"},
		{0, 0, 1, -1, true, false, "partial sum of K_(u+1) negative"},
		{0, 0, 1, 0, false, true, "K_(u+1) not needed, its sum ignored"},
	} {
		if got := kvTemmeDone(c.bf, c.bh, c.sf, c.sh, c.k1, kvTemmeStop); got != c.want {
			t.Errorf("%s: kvTemmeDone = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestKvTemmeTerms: the split between the phases at its edges.
func TestKvTemmeTerms(t *testing.T) {
	for _, c := range []struct {
		x float64
		m int
	}{{kvTemmeMinX, 0}, {math.Nextafter(0x1p-10, 0), 0}, {0x1p-10, 1}, {math.Nextafter(0.125, 0), 1}, {0.125, 2},
		{math.Nextafter(0.5, 0), 2}, {0.5, 3}, {math.Nextafter(1, 0), 3}, {1, 4}, {math.Nextafter(kvCF2MinX, 0), 4}} {
		if got := kvTemmeTerms(c.x); got != c.m {
			t.Errorf("kvTemmeTerms(%g) = %d, want %d", c.x, got, c.m)
		}
	}
}

// TestKvTemmeCaps: the loop declines where it cannot converge (a negative
// tolerance is never met), and converges everywhere in the region.
func TestKvTemmeCaps(t *testing.T) {
	if _, _, ok := kvTemmeSeriesTol(0.3, 0.9, true, -1); ok {
		t.Error("kvTemmeSeriesTol(0.3, 0.9, tolerance -1) converged")
	}
	if _, _, ok := kvTemmeSeriesTol(0.3, 1e-300, false, -1); ok {
		t.Error("kvTemmeSeriesTol(0.3, 1e-300, tolerance -1) converged")
	}
	for _, x := range []float64{kvTemmeMinX, 1e-300, 1e-100, 1e-8, 0.001, 0.1, 0.5, 1, math.Nextafter(kvCF2MinX, 0)} {
		for _, u := range []float64{-0.5, -0.3, 0, 1e-300, 0.3, 0.5} {
			if _, _, ok := kvTemmeSeries(u, x, true); !ok {
				t.Errorf("kvTemmeSeries(%g, %g) did not converge", u, x)
			}
		}
	}
}

// TestKvTemmeRangeEdges: the guards of Path 4 at their exact edges, and the
// orders it takes.
func TestKvTemmeRangeEdges(t *testing.T) {
	below := math.Nextafter(kvCF2MinX, 0)
	for _, scaled := range []bool{false, true} {
		if r, ok := kvFast(0.3, below, scaled); !ok {
			t.Error("kvFast(0.3, 1.5-) declined")
		} else if r2, _ := kvTemmeChain(0.3, below, scaled); r != r2 {
			t.Errorf("kvFast(0.3, 1.5-) = %v, kvTemmeChain %v", r, r2)
		}
		if r, ok := kvFast(0.3, kvCF2MinX, scaled); !ok {
			t.Error("kvFast(0.3, 1.5) declined")
		} else if r2, _ := kvCF2Chain(0.3, kvCF2MinX, scaled); r != r2 {
			t.Errorf("kvFast(0.3, 1.5) = %v, not CF2's %v", r, r2)
		}
		// the lower edge: 2^-1000 answered, one float below declined
		if r, ok := kvFast(0.3, kvTemmeMinX, scaled); !ok {
			t.Error("kvFast(0.3, 2^-1000) declined")
		} else if r2, _ := kvTemmeChain(0.3, kvTemmeMinX, scaled); r != r2 {
			t.Errorf("kvFast(0.3, 2^-1000) = %v, kvTemmeChain %v", r, r2)
		}
		if _, ok := kvFast(0.3, math.Nextafter(kvTemmeMinX, 0), scaled); ok {
			t.Error("kvFast(0.3, 2^-1000-) answered")
		}
		// v = 32 is Debye, the order next to it this path
		if _, ok := kvFast(math.Nextafter(32, 0), 1, scaled); !ok {
			t.Error("kvFast(32-, 1) declined")
		}
		if got, ok := kvFast(32, 1, scaled); !ok {
			t.Error("kvFast(32, 1) declined")
		} else if d, _ := kvDebye(32, 1, scaled); got != d {
			t.Errorf("kvFast(32, 1) = %v, Debye %v", got, d)
		}
	}
	// orders below 1, half integers, integers (u = 0: K_0 and K_1 from the same
	// code), the order next to 1/2
	for _, v := range []float64{5e-324, 1e-300, 0.001, 0.3, math.Nextafter(0.5, 0), 0.5, 0.7, 0.999, 2, 3, 7, 16, 31, 1.5, 31.5, 31.999999} {
		if _, ok := kvFast(v, 0.3, false); !ok {
			t.Errorf("kvFast(%g, 0.3) declined", v)
		}
	}
	// kvChainUp: the answer of n = 0 is K_u itself, of n = 1 is K_(u+1)
	k0, _, _ := kvTemmeSeries(0.3, 0.9, false)
	if got, _ := kvTemmeChain(0.3, 0.9, false); got != k0.Hi+k0.Lo {
		t.Error("kvTemmeChain(0.3, 0.9) is not K_0.3")
	}
}

// TestKvTemmeSmallAndIntegerOrders: orders at the ends of u (tiny, an integer,
// next to 1/2) against the oracle, in ulps. The oracle takes |u| >= 2^-100: an
// integer order is held by the fixtures (kn rows) and by the recurrence from
// K_0 and K_1 of the same code against Kn at x >= 1 (kn_fast.go).
func TestKvTemmeSmallAndIntegerOrders(t *testing.T) {
	for _, v := range []float64{1e-30, 1e-12, 5e-324, 0.5, 0.5000000000000001, 0.49999999999999994, 1.0000000000000002, 1.5, 2.0000000000000004, 31.5, 31.999999999999996} {
		for _, x := range []float64{1e-300, 1e-100, 1e-9, 0.01, 0.3, 1.1, 1.4999999999999998} {
			if v < 1e-20 {
				continue // below the oracle's range (|u| >= 2^-100): K_v = K_0 to the last bit
			}
			ke := bigKve(v, x)
			if trueKvOverflows(ke, x) {
				continue
			}
			if e := ivOracleUlps(Kve(v, x), ke); e > kvFastBound {
				t.Errorf("Kve(%g, %g): %.3f ulp", v, x, e)
			}
			if e := ivOracleUlps(Kv(v, x), bnew().Mul(ke, bigExp(bf(-x)))); e > kvFastBound {
				t.Errorf("Kv(%g, %g): %.3f ulp", v, x, e)
			}
		}
	}
	// an order of 5e-324 and of 1e-300 is K_0 to the last bit, and equals Kv(0)
	for _, x := range []float64{1e-300, 0.01, 0.3, 1.4999999999999998} {
		if a, b := Kv(5e-324, x), K0(x); math.Abs(a-b) > 0.6*ulp(b) {
			t.Errorf("Kv(5e-324, %g) = %v, K0 %v", x, a, b)
		}
	}
	// integer orders, against the recurrence of Kn from K0 and K1 (x < 1, below
	// kn_fast.go's range: through K1 and K0 and the exact recurrence in big)
	for _, n := range []float64{2, 3, 7, 16, 31} {
		for _, x := range []float64{0.01, 0.3, 0.9} {
			k0, k1 := bigKint(0, x), bigKint(1, x)
			for j := 1; j < int(n); j++ {
				k0, k1 = k1, bnew().Add(k0, bnew().Mul(bnew().Quo(bf(float64(2*j)), bf(x)), k1))
			}
			if e := ivOracleUlps(Kv(n, x), k1); e > kvFastBound {
				t.Errorf("Kv(%g, %g): %.3f ulp", n, x, e)
			}
		}
	}
}

// bigKint returns K_0(x) or K_1(x) as the oracle's K_u and K_(u+1) at u =
// 2^-90 (the cancellation of 320 bits holds it): K_u differs from K_0 and
// K_(1-u) from K_1 by about 2^-90 ln(2/x), below 2^-80, enough for a 0.6 ulp
// check.
func bigKint(n int, x float64) *big.Float {
	u := 0x1p-90
	k0, k1 := bigKpair(u, x)
	if n == 0 {
		return k0
	}
	return k1
}

// TestKvTemmeOverflow: the path answers exactly where K_v(x) is below
// kvTemmeBig = 2^1020 and declines where it is not (the result, K_(u+1) and
// every intermediate of the chain increase to it), and where the true value
// overflows float64 Kv is +Inf (besselIK's answer). Points within 1e-9 of the
// edge are not judged.
func TestKvTemmeOverflow(t *testing.T) {
	n := kvSweepN(300)
	r := rand.New(rand.NewSource(26))
	lim := bnew().SetMantExp(bf(1), 1020)
	var below, above int
	for i := 0; i < n; i++ {
		v := 0.3 + float64(r.Float64()*31.5)
		if v == math.Trunc(v) {
			v += 0.25
		}
		// x so that K_v(x) is near 2^1000: ln K ~ v ln(2/x) + lnGamma(v) - ln 2
		lg, _ := math.Lgamma(v)
		lx := (707.0 + math.Log(2) - lg) / v
		x := 2 * math.Exp(-float64(lx*(0.5+r.Float64())))
		if !(x >= kvTemmeMinX && x < kvCF2MinX) {
			continue
		}
		k := bnew().Mul(bigKve(v, x), bigExp(bf(-x)))
		c := k.Cmp(lim)
		ratio, _ := bnew().Quo(k, lim).Float64()
		if math.Abs(ratio-1) < 1e-9 {
			continue
		}
		_, ok := kvFast(v, x, false)
		if ok != (c < 0) {
			t.Errorf("Kv(%g, %g): K %s 2^1000 but the path answered %v", v, x, map[bool]string{true: "<", false: ">"}[c < 0], ok)
		}
		if c < 0 {
			below++
		} else {
			above++
		}
		if trueKvOverflows(bigKve(v, x), x) && !math.IsInf(Kv(v, x), 1) {
			t.Errorf("Kv(%g, %g) = %v, the true value overflows", v, x, Kv(v, x))
		}
	}
	t.Logf("%d points below 2^1000 (answered), %d above (declined)", below, above)
	if below < n/10 || above < n/10 {
		t.Errorf("sweep did not cross the edge: %d below, %d above", below, above)
	}
	// where K_v does not overflow float64 but is above 2^1000, besselIK answers
	// and Kv is its bit for bit
	if _, ok := kvFast(1.4, 1e-300, false); ok {
		t.Error("kvFast(1.4, 1e-300) answered")
	}
	if got := Kv(1.4, 1e-300); !math.IsInf(got, 1) {
		t.Errorf("Kv(1.4, 1e-300) = %v, want +Inf", got)
	}
}

// TestKvDeclinedBitIdentity: where no fast path answers, Kv and Kve are
// besselIK, bit for bit, on kvSweepN random points in four classes of declines:
// x below 2^-1000, K_v of 2^1000 or more, v above the Debye range, and the single
// series of besselAsymIK at x >= 24. (kv_bitcheck_test.go compares the whole
// function with a baseline build on 2e5 points.)
func TestKvDeclinedBitIdentity(t *testing.T) {
	n := kvSweepN(20000)
	r := rand.New(rand.NewSource(14))
	var checked int
	for i := 0; i < n; i++ {
		var v, x float64
		switch i % 4 {
		case 0:
			v, x = float64(r.Float64()*31.9), math.Exp(float64(r.Float64()*(math.Log(1e-302)-math.Log(1e-320)))+math.Log(1e-320))
		case 1:
			v, x = 1.5+float64(r.Float64()*30), math.Exp(math.Log(1e-300)+float64(r.Float64()*(math.Log(1e-30)-math.Log(1e-300))))
		case 2:
			v, x = 1100.1+float64(r.Float64()*3000), math.Exp(float64(r.Float64()*8)-4)
		case 3:
			v = 2 + float64(r.Float64()*4)
			x = 24 + float64(r.Float64()*40)
		}
		if v == 0 || v == 1 || v == math.Trunc(v) {
			continue
		}
		if _, ok := kvFast(v, x, false); ok {
			continue
		}
		checked++
		_, k := besselIK(v, x, false, true, false)
		if got := Kv(v, x); got != k && !(got != got && k != k) {
			t.Fatalf("Kv(%g, %g) = %v, besselIK %v", v, x, got, k)
		}
		if _, ok := kvFast(v, x, true); ok {
			continue
		}
		_, ke := besselIK(v, x, false, true, true)
		if got := Kve(v, x); got != ke && !(got != got && ke != ke) {
			t.Fatalf("Kve(%g, %g) = %v, besselIK %v", v, x, got, ke)
		}
	}
	if checked < n/4 {
		t.Errorf("only %d of %d points declined", checked, n)
	}
}
