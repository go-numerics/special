package special

import (
	"math"
	"math/big"
	"math/rand"
	"testing"

	"github.com/go-numerics/special/dd"
)

// The tests of kv_temme_fast.go (Path 4 of kv_fast.go): the start pair against
// the math/big oracle of kv_oracle_test.go, the whole function against it, the
// pieces (the Gamma series, cosh and sinh(sigma)/sigma), the signs and the
// tail bounds the truncation rests on, and the edges of every range guard.

// kvTemmePoint draws (u, x) for the sweeps: u uniform in [-1/2, 1/2] with
// shares very near 0 and +-1/2, x log-uniform in [1e-300, 1.5) with a share in
// [1, 1.5) (where the series is longest and cancels most) and a share of tiny x.
func kvTemmePoint(r *rand.Rand, i int) (u, x float64) {
	u = r.Float64() - 0.5
	switch i % 8 {
	case 1:
		u = float64((r.Float64() - 0.5) * 1e-6)
	case 2:
		u = math.Copysign(0.5-float64(r.Float64()*1e-6), r.Float64()-0.5)
	case 3:
		u = float64((r.Float64() - 0.5) * 1e-12)
	}
	x = math.Exp(math.Log(1e-3) + float64(r.Float64()*(math.Log(kvCF2MinX)-math.Log(1e-3))))
	switch i % 7 {
	case 0:
		x = 1 + float64(r.Float64()*(kvCF2MinX-1))
	case 1:
		x = math.Exp(math.Log(1e-300) + float64(r.Float64()*(math.Log(1e-3)-math.Log(1e-300))))
	}
	if x >= kvCF2MinX {
		x = math.Nextafter(kvCF2MinX, 0)
	}
	if u == 0 {
		u = 1e-20
	}
	return
}

// TestKvTemmePairOracle: the pair K_u, K_(u+1) (unscaled) within 2^-60 relative
// of the exact value, the budget of the path (kv_temme_fast.go), and the worst
// of each logged. A K_(u+1) that overflows float64 is declined by the chain, so
// only finite ones are compared.
func TestKvTemmePairOracle(t *testing.T) {
	n := kvSweepN(400)
	r := rand.New(rand.NewSource(21))
	us, xs := make([]float64, n), make([]float64, n)
	for i := range us {
		us[i], xs[i] = kvTemmePoint(r, i)
	}
	e0, e1 := make([]float64, n), make([]float64, n)
	kvParallel(n, func(i int) {
		k0, k1, ok := kvTemmeSeries(us[i], xs[i], true)
		if !ok {
			e0[i], e1[i] = math.Inf(1), math.Inf(1)
			return
		}
		b0, b1 := bigKpair(us[i], xs[i])
		e0[i] = relDD(k0.Hi, k0.Lo, b0)
		if k1.Hi < kvTemmeBig {
			e1[i] = relDD(k1.Hi, k1.Lo, b1)
		}
	})
	var w0, w1, wx0, wx1 float64
	for i := range e0 {
		if e0[i] > w0 {
			w0, wx0 = e0[i], xs[i]
		}
		if e1[i] > w1 {
			w1, wx1 = e1[i], xs[i]
		}
		if e0[i] > 0x1p-60 || e1[i] > 0x1p-60 {
			t.Errorf("pair at u=%g x=%g: K_u %.3g, K_u+1 %.3g relative (limit 2^-60 = %.3g)", us[i], xs[i], e0[i], e1[i], 0x1p-60)
		}
	}
	t.Logf("%d points: K_u worst 2^%.2f (x=%g), K_{u+1} worst 2^%.2f (x=%g) relative", n, math.Log2(w0), wx0, math.Log2(w1), wx1)
}

// kvTemmeOrders draws orders for the whole-function sweeps: uniform in (0, 32),
// a third within 1e-3 of an integer, a tenth half-integers, a twentieth below
// 1e-3, none an integer.
func kvTemmeOrders(r *rand.Rand, i int) float64 {
	v := float64(r.Float64() * 32)
	switch {
	case i%3 == 0:
		v = math.Floor(v) + float64((r.Float64()-0.5)*2e-3)
	case i%10 == 1:
		v = math.Floor(v) + 0.5
	case i%20 == 2:
		v = float64(r.Float64() * 1e-3)
	}
	if v <= 0 || v == math.Trunc(v) || v >= 32 {
		v = 0.37 + float64(i%30)
	}
	return v
}

// trueKvClass compares the exact K_v(x) (unscaled, big) with MaxFloat64: it
// returns true when K_v(x) rounds to +Inf, that is when it is at least
// MaxFloat64 (1 + 2^-54).
func trueKvOverflows(kve *big.Float, x float64) bool {
	k := bnew().Mul(kve, bigExp(bf(-x)))
	lim := bnew().Mul(bf(math.MaxFloat64), bnew().Add(bf(1), bnew().SetMantExp(bf(1), -54)))
	return k.Cmp(lim) >= 0
}

// TestKvTemmeOracle: Kv and Kve of the region against the oracle, in ulps: the
// bound kvFastBound; where the true Kv overflows float64 the public function
// returns +Inf (this path declines there, besselIK answers).
func TestKvTemmeOracle(t *testing.T) {
	n := kvSweepN(300)
	r := rand.New(rand.NewSource(22))
	vs, xs := make([]float64, n), make([]float64, n)
	for i := range vs {
		vs[i] = kvTemmeOrders(r, i)
		_, xs[i] = kvTemmePoint(r, i)
	}
	eu, es := make([]float64, n), make([]float64, n)
	ov, fast := make([]int, n), make([]bool, n)
	kvParallel(n, func(i int) {
		ke := bigKve(vs[i], xs[i])
		_, fast[i] = kvFast(vs[i], xs[i], false)
		if trueKvOverflows(ke, xs[i]) {
			ov[i] = 1
			if !math.IsInf(Kv(vs[i], xs[i]), 1) || fast[i] {
				ov[i] = 2
			}
			return
		}
		if !fast[i] {
			return // declined: besselIK answers, not this path's to hold (the band 2^1020 up to MaxFloat64)
		}
		es[i] = ivOracleUlps(Kve(vs[i], xs[i]), ke)
		eu[i] = ivOracleUlps(Kv(vs[i], xs[i]), bnew().Mul(ke, bigExp(bf(-xs[i]))))
	})
	var wu, ws float64
	nov, nfast := 0, 0
	for i := range vs {
		if fast[i] {
			nfast++
		}
		if ov[i] > 0 {
			nov++
			if ov[i] == 2 {
				t.Errorf("Kv(%g, %g) overflows: got %v, fast path answered %v", vs[i], xs[i], Kv(vs[i], xs[i]), fast[i])
			}
			continue
		}
		wu, ws = math.Max(wu, eu[i]), math.Max(ws, es[i])
		if eu[i] > kvFastBound || es[i] > kvFastBound {
			t.Errorf("Kv(%g, %g): %.3f ulp, Kve %.3f ulp (limit %.2f)", vs[i], xs[i], eu[i], es[i], kvFastBound)
		}
	}
	t.Logf("%d points (%d by the fast path, %d overflow): Kv worst %.3f ulp, Kve worst %.3f ulp", n, nfast, nov, wu, ws)
}

// kvTemmeSteps replays the recurrences of kvTemmeSeriesTol in float64 for
// 0 <= k <= steps, calling visit with k, c_k, f_k, p_k, q_k and h_k (the
// starting values f_0, p_0 and q_0 from the float64 formulas with the library
// exponential and hyperbolic functions: the tests that use it measure sizes and
// signs, not digits).
func kvTemmeSteps(u, x float64, steps int, visit func(k int, c, f, p, q, h float64)) {
	g1, g2 := besselRGammaPair(u)
	rp := g2 - float64(u*g1)
	rm := g2 + float64(u*g1)
	L := -math.Log(x / 2)
	sg := float64(u * L)
	ch, shc := math.Cosh(sg), 1.0
	if sg != 0 {
		shc = math.Sinh(sg) / sg
	}
	f := (float64(ch*g1) + float64(float64(shc*L)*g2)) / float64(rp*rm)
	E := math.Exp(sg)
	p := E / (2 * rp)
	q := 1 / (2 * E * rm)
	c := 1.0
	visit(0, c, f, p, q, p)
	qh := float64(x * x / 4)
	for k := 1; k <= steps; k++ {
		fk := float64(k)
		f = (float64(fk*f) + p + q) / (float64(fk*fk) - float64(u*u))
		p /= fk - u
		q /= fk + u
		c = float64(c * qh / fk)
		visit(k, c, f, p, q, p-float64(fk*f))
	}
}

// TestKvTemmeSigns: p_k, q_k > 0 for all k, f_k > 0 for k >= 1, on a grid of
// the region (the signs the truncation bound of kv_temme_fast.go needs).
func TestKvTemmeSigns(t *testing.T) {
	for _, u := range []float64{-0.5, -0.4999, -0.3, -0.1, -1e-9, 1e-12, 0.0001, 0.2, 0.4, 0.4999, 0.5} {
		for _, x := range []float64{1e-300, 1e-100, 1e-8, 1e-3, 0.05, 0.3, 0.7, 1.0, 1.123, 1.3, 1.4999999999999998} {
			kvTemmeSteps(u, x, 30, func(k int, c, f, p, q, h float64) {
				if !(p > 0 && q > 0) || (k >= 1 && !(f > 0)) {
					t.Errorf("u=%g x=%g k=%d: f=%g p=%g q=%g", u, x, k, f, p, q)
				}
			})
		}
	}
}

// TestKvTemmeTermWeights prints, for each x of a grid and each count m of
// double-double terms, the worst weight over a grid of u of the terms past
// the m-th: sum_(k>m) c_k f_k / K_u and sum_(k>m) c_k |h_k| / |H|.
func TestKvTemmeTermWeights(t *testing.T) {
	us := []float64{-0.5, -0.4999, -0.45, -0.3, -0.1, -1e-6, 1e-9, 0.0001, 0.1, 0.25, 0.4, 0.4999, 0.5}
	xs := []float64{math.Nextafter(0x1p-10, 0), math.Nextafter(0.125, 0), math.Nextafter(0.5, 0), math.Nextafter(1, 0), 1e-300, 1e-100, 1e-20, 1e-10, 1e-8, 1e-6, 1e-4, 1e-3, 0.01, 0.02, 0.03, 0.0312, 0.0313, 0.05, 0.1, 0.125, 0.2, 0.3, 0.35, 0.5, 0.7, 0.8, 1, 1.2, 1.4999999999999998}
	for _, x := range xs {
		var wf, wh [6]float64
		for _, u := range us {
			var K, H float64
			var cf, ch [40]float64
			kvTemmeSteps(u, x, 39, func(k int, c, f, p, q, h float64) {
				cf[k], ch[k] = float64(c*f), float64(c*h)
				K += cf[k]
				H += ch[k]
			})
			for m := 0; m < 6; m++ {
				var tf, th float64
				for k := m + 1; k < 40; k++ {
					tf += math.Abs(cf[k])
					th += math.Abs(ch[k])
				}
				wf[m] = math.Max(wf[m], tf/K)
				wh[m] = math.Max(wh[m], th/math.Abs(H))
			}
		}
		m := kvTemmeTerms(x)
		t.Logf("x=%-8.4g m=%d weights K: %.1e %.1e %.1e %.1e %.1e  H: %.1e %.1e %.1e %.1e %.1e", x, m, wf[0], wf[1], wf[2], wf[3], wf[4], wh[0], wh[1], wh[2], wh[3], wh[4])
		if wf[m] > kvTemmeTailWeight || wh[m] > kvTemmeTailWeight {
			t.Errorf("x=%g: terms past the %d in double-double weigh %.3g (K_u), %.3g (K_u+1); limit %g", x, m, wf[m], wh[m], kvTemmeTailWeight)
		}
	}
}

// kvTemmeTailWeight bounds the weight of the float64 terms: four roundings of
// 2^-53 on 1e-3 of a sum are 2^-61.

const kvTemmeTailWeight = 1e-3

// bigRGamma returns 1/Gamma(1+s u) for s = +-1 (both arguments in [1/2, 3/2]).
func bigRGamma(u float64, s float64) *big.Float {
	ivOracleInit()
	a := bnew().Add(bf(1), bf(float64(s*u)))
	return bigExp(bnew().Neg(bigLogGamma(a)))
}

// TestKvTemmeGammaSeries: g1 and g2 (the Taylor series of zz_kvTemme_coeffs.go
// in w = u^2) against (1/Gamma(1-u) -+ 1/Gamma(1+u)) from the math/big ln
// Gamma at 320 bits, to 2^-66 relative, on a grid of u that includes the
// limit u -> 0: g1 stays -gamma without ever dividing by u.
func TestKvTemmeGammaSeries(t *testing.T) {
	ivOracleInit()
	us := []float64{1e-300, 1e-30, 1e-12, 1e-6, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.45, 0.49, 0.4999999, 0.5, -1e-30, -0.001, -0.25, -0.4999999, -0.5}
	r := rand.New(rand.NewSource(23))
	for i := 0; i < 300; i++ {
		us = append(us, r.Float64()-0.5)
	}
	var w1, w2 float64
	for _, u := range us {
		w := dd.MulFF(u, u)
		g1, g2 := kvTemmeGamma(w)
		if math.Abs(u) < 1e-20 {
			continue // 1 +- u is 1 at 320 bits: held at u = 0 below
		}
		rp, rm := bigRGamma(u, 1), bigRGamma(u, -1)
		want2 := bnew().Quo(bnew().Add(rp, rm), bf(2))
		want1 := bnew().Quo(bnew().Sub(rm, rp), bnew().Mul(bf(2), bf(u)))
		e1, e2 := relDD(g1.Hi, g1.Lo, want1), relDD(g2.Hi, g2.Lo, want2)
		w1, w2 = math.Max(w1, e1), math.Max(w2, e2)
		if e1 > 0x1p-66 || e2 > 0x1p-66 {
			t.Errorf("u=%g: g1 %.3g, g2 %.3g relative (limit 2^-66)", u, e1, e2)
		}
	}
	t.Logf("%d points: g1 worst 2^%.1f, g2 worst 2^%.1f relative", len(us), math.Log2(w1), math.Log2(w2))
	// u = 0 exactly: g1 = -gamma, g2 = 1
	g1, g2 := kvTemmeGamma(dd.MulFF(0, 0))
	gamma, _, _ := big.ParseFloat("0.57721566490153286060651209008240243104215933593992359880576723488486772677766467", 10, ivOraclePrec, big.ToNearestEven)
	if e := relDD(g1.Hi, g1.Lo, bnew().Neg(gamma)); e > 0x1p-100 {
		t.Errorf("g1(0) = %v %v, not -gamma (%.3g relative)", g1.Hi, g1.Lo, e)
	}
	if g2.Hi != 1 || g2.Lo != 0 {
		t.Errorf("g2(0) = %v %v, not 1", g2.Hi, g2.Lo)
	}
}
