package special

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"strconv"
	"sync"
	"testing"
)

// The tests of kv_cf2_fast.go (Path 3 of kv_fast.go): the start pair against
// the math/big oracle of kv_oracle_test.go, the whole function against it, the
// signs and the bracket the certificate rests on, the truncation, and the
// edges of every range guard.

// kvSweepN is the size of the random sweeps: def, or KV_SWEEP from the
// environment (the 100k+ run quoted in the pull request is KV_SWEEP=120000).
func kvSweepN(def int) int {
	if s := os.Getenv("KV_SWEEP"); s != "" {
		if n, err := strconv.Atoi(s); err == nil && n > 0 {
			return n
		}
	}
	if testing.Short() {
		return def / 5
	}
	return def
}

// kvParallel runs f(i) for i in [0, n) on all CPUs.
func kvParallel(n int, f func(i int)) {
	var wg sync.WaitGroup
	next := make(chan int)
	for w := 0; w < runtime.GOMAXPROCS(0); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range next {
				f(i)
			}
		}()
	}
	for i := 0; i < n; i++ {
		next <- i
	}
	close(next)
	wg.Wait()
}

// kvCF2Point draws (u, x) for the sweeps: u uniform in [-1/2, 1/2] with a
// share very near 0 and +-1/2, x log-uniform in [kvCF2MinX, 24) with a share just
// above the lower edge.
func kvCF2Point(r *rand.Rand, i int) (u, x float64) {
	u = r.Float64() - 0.5
	switch i % 8 {
	case 1:
		u = float64((r.Float64() - 0.5) * 1e-6)
	case 2:
		u = math.Copysign(0.5-float64(r.Float64()*1e-6), r.Float64()-0.5)
	}
	x = math.Exp(math.Log(kvCF2MinX) + float64(r.Float64()*(math.Log(kvAsymMinX)-math.Log(kvCF2MinX))))
	if i%5 == 0 {
		x = kvCF2MinX + float64(r.Float64()*0.5)
	}
	if u == 0 {
		u = 1e-9
	}
	return
}

// TestKvCF2PairOracle: the pair K_u e^x, K_{u+1} e^x within 2^-57 relative of the
// exact value (the budget of the path, kv_cf2_fast.go), and the worst of each
// logged.
func TestKvCF2PairOracle(t *testing.T) {
	n := kvSweepN(1500)
	r := rand.New(rand.NewSource(11))
	us, xs := make([]float64, n), make([]float64, n)
	for i := range us {
		us[i], xs[i] = kvCF2Point(r, i)
	}
	e0, e1 := make([]float64, n), make([]float64, n)
	kvParallel(n, func(i int) {
		k0, k1, ok := kvCF2Pair(us[i], xs[i])
		if !ok {
			e0[i], e1[i] = math.Inf(1), math.Inf(1)
			return
		}
		b0, b1 := bigKpair(us[i], xs[i])
		ex := bigExp(bf(xs[i]))
		e0[i] = relDD(k0.Hi, k0.Lo, bnew().Mul(b0, ex))
		e1[i] = relDD(k1.Hi, k1.Lo, bnew().Mul(b1, ex))
	})
	var w0, w1 float64
	for i := range e0 {
		if e0[i] > w0 {
			w0 = e0[i]
		}
		if e1[i] > w1 {
			w1 = e1[i]
		}
		if e0[i] > 0x1p-57 || e1[i] > 0x1p-57 {
			t.Errorf("pair at u=%g x=%g: K_u %.3g, K_u+1 %.3g relative (limit 2^-57 = %.3g)", us[i], xs[i], e0[i], e1[i], 0x1p-57)
		}
	}
	t.Logf("%d points: K_u worst 2^%.2f, K_{u+1} worst 2^%.2f relative", n, math.Log2(w0), math.Log2(w1))
}

// kvCF2Orders draws orders for the whole-function sweeps: uniform in (0, 32),
// a third within 1e-3 of an integer, a tenth half-integers, none an integer.
func kvCF2Orders(r *rand.Rand, i int) float64 {
	v := float64(r.Float64() * 32)
	switch {
	case i%3 == 0:
		v = math.Floor(v) + float64((r.Float64()-0.5)*2e-3)
	case i%10 == 1:
		v = math.Floor(v) + 0.5
	}
	if v <= 0 || v == math.Trunc(v) || v >= 32 {
		v = 0.37 + float64(i%30)
	}
	return v
}

// TestKvCF2Oracle: Kv and Kve of the region against the oracle, in ulps: the
// bound kvFastBound.
func TestKvCF2Oracle(t *testing.T) {
	n := kvSweepN(1200)
	r := rand.New(rand.NewSource(12))
	vs, xs := make([]float64, n), make([]float64, n)
	for i := range vs {
		vs[i] = kvCF2Orders(r, i)
		_, xs[i] = kvCF2Point(r, i)
	}
	eu, es := make([]float64, n), make([]float64, n)
	kvParallel(n, func(i int) {
		ke := bigKve(vs[i], xs[i])
		es[i] = ivOracleUlps(Kve(vs[i], xs[i]), ke)
		eu[i] = ivOracleUlps(Kv(vs[i], xs[i]), bnew().Mul(ke, bigExp(bf(-xs[i]))))
	})
	var wu, ws float64
	for i := range vs {
		wu, ws = math.Max(wu, eu[i]), math.Max(ws, es[i])
		if eu[i] > kvFastBound || es[i] > kvFastBound {
			t.Errorf("Kv(%g, %g): %.3f ulp, Kve %.3f ulp (limit %.2f)", vs[i], xs[i], eu[i], es[i], kvFastBound)
		}
	}
	t.Logf("%d points: Kv worst %.3f ulp, Kve worst %.3f ulp", n, wu, ws)
}

// kvCF2Steps replays the recurrences of kvCF2PairTol in float64, calling visit
// with k, a_k, D_k, delta_k, t_k and Q_k after each step.
func kvCF2Steps(u, x float64, steps int, visit func(k int, a, D, delta, tk, Q float64)) {
	a := float64(u*u) - 0.25
	b := float64(2 * (x + 1))
	D := 1 / b
	delta := D
	tm2, tm1 := 0.0, -a
	Q := tm1
	for k := 2; k <= steps; k++ {
		ak1, bprev := a, b
		a -= float64(2 * (k - 1))
		b += 2
		Dp := D
		D = 1 / (b + float64(a*D))
		delta = float64(float64(delta * (-a * Dp * D)))
		tk := (ak1*tm2/float64(k-1) + float64(bprev*tm1)) / float64(k)
		tm2, tm1 = tm1, tk
		Q += tk
		visit(k, a, D, delta, tk, Q)
	}
}

// TestKvCF2Signs: for k >= 2, a_k < 0, D_k > 0, delta_k > 0, t_k > 0 and Q_k
// increasing, on a grid of the region: every term of f and of S is positive, as
// the argument of kv_cf2_fast.go needs (no cancellation).
func TestKvCF2Signs(t *testing.T) {
	for _, u := range []float64{-0.4999999, -0.4, -0.25, -0.01, 1e-9, 0.01, 0.25, 0.4, 0.4999999} {
		for _, x := range []float64{kvCF2MinX, 1.7, 2, 3, 5, 8, 12, 18, 23.9999} {
			prevQ := math.Inf(-1)
			kvCF2Steps(u, x, 150, func(k int, a, D, delta, tk, Q float64) {
				if !(a < 0 && D > 0 && delta > 0 && tk > 0 && Q > prevQ) {
					t.Fatalf("u=%g x=%g k=%d: a=%g D=%g delta=%g t=%g Q=%g (previous %g)", u, x, k, a, D, delta, tk, Q, prevQ)
				}
				prevQ = Q
			})
		}
	}
}

// TestKvCF2FBracket: the rigorous bound on the tail of f (kvCF2FTail), against
// f computed by the backward evaluation of the fraction at 320 bits from level
// 800: f - f_k lies in [0, kvCF2FTail], and the bound is within a factor of
// 100 of it where the loop stops (a bound that is true and loose by orders of
// magnitude would cost iterations).
func TestKvCF2FBracket(t *testing.T) {
	for _, u := range []float64{-0.45, 0.01, 0.3, 0.5} {
		for _, x := range []float64{kvCF2MinX, 2.5, 5, 10, 23.5} {
			ftrue := bigCF2f(u, x, 800)
			var fk float64
			a := float64(u*u) - 0.25
			b := float64(2 * (x + 1))
			D := 1 / b
			fk = D
			delta := D
			for k := 2; k <= 60; k++ {
				a -= float64(2 * (k - 1))
				b += 2
				Dp := D
				D = 1 / (b + float64(a*D))
				delta = float64(float64(delta * (-a * Dp * D)))
				fk += delta
				bound := kvCF2FTail(k, delta, D)
				d := ftrue - fk
				if d < -1e-15*ftrue || d > float64(bound*(1+1e-9))+float64(1e-15*ftrue) {
					t.Fatalf("u=%g x=%g k=%d: f - f_k = %.3g, bound %.3g", u, x, k, d, bound)
				}
				if k == 40 && d > 1e-12 && bound > 100*d {
					t.Errorf("u=%g x=%g k=%d: bound %.3g is more than 100 times the tail %.3g", u, x, k, bound, d)
				}
			}
		}
	}
	if !math.IsInf(kvCF2FTail(3, 0.1, 0.5), 1) { // w D = 3.5 * 0.5 >= 1: no bracket
		t.Error("kvCF2FTail did not refuse w D >= 1")
	}
}

// bigCF2f is f = 1/(b_1 + a_2/(b_2 + a_3/(... a_N/b_N))) by the backward
// evaluation at 320 bits, a_k = u^2 - (k - 1/2)^2, b_k = 2 (x + k): the tail
// after level N is dropped (it is below 2^-300 for the x and N used).
func bigCF2f(u, x float64, N int) float64 {
	t := bnew()
	h := bf(0.5)
	for k := N; k >= 2; k-- {
		km := bnew().Sub(bf(float64(k)), h)
		a := bnew().Sub(bnew().Mul(bf(u), bf(u)), bnew().Mul(km, km))
		b := bnew().Mul(bf(2), bnew().Add(bf(x), bf(float64(k))))
		t = bnew().Quo(a, bnew().Add(b, t))
	}
	b1 := bnew().Mul(bf(2), bnew().Add(bf(x), bf(1)))
	f, _ := bnew().Quo(bf(1), bnew().Add(b1, t)).Float64()
	return f
}

// TestKvCF2TermWeights: the claim of the comment on kvCF2PairTol: at the
// first float64 step (k = kvCF2Terms(x) + 1) the terms still to come are
// below 4e-3 of S (x < 3), 2e-3 (x < 8), 1e-3 (beyond), so that their
// float64 rounding is below 2^-58 of the sum.
func TestKvCF2TermWeights(t *testing.T) {
	limit := func(x float64) float64 {
		switch {
		case x < 3:
			return 4e-3
		case x < 8:
			return 2e-3
		}
		return 1e-3
	}
	worst := map[float64]float64{}
	defer func() { t.Logf("worst weight of the float64 terms by limit class: %v", worst) }()
	for _, u := range []float64{-0.4999, -0.3, -0.1, 0.0001, 0.2, 0.4, 0.4999} {
		for _, x := range []float64{kvCF2MinX, 1.9, 2, 2.9, 3, 5, 7.9, 8, 12, 23.9} {
			m := kvCF2Terms(x)
			var S, tail float64 = 1, 0
			kvCF2Steps(u, x, 400, func(k int, a, D, delta, tk, Q float64) {
				if k == 2 {
					S += float64((0.25 - float64(u*u)) * (1 / float64(2*(x+1))))
				}
				S += float64(Q * delta)
				if k > m {
					tail += float64(Q * delta)
				}
			})
			worst[limit(x)] = math.Max(worst[limit(x)], tail/S)
			if tail/S > limit(x) {
				t.Errorf("u=%g x=%g: terms past the %d in double-double weigh %.3g of S (limit %.3g)", u, x, m, tail/S, limit(x))
			}
		}
	}
}

// TestKvCF2Truncation: the pair at the production tolerances against the same
// loop run to 2^-84 (above that the loop's ratio test, term < 0.9 term_prev,
// is never met at x near kvCF2MinX before the cap of kvCF2MaxIters): the
// stopped pair is within 2^-60 of the deep one, on kvSweepN points (2^-61.5 at
// worst on 1e5, at x just above kvCF2MinX, where the ratio of consecutive terms
// is 0.75 at the stop and the tail about four terms). This measures the
// truncation of S, for which kv_cf2_fast.go has no proof.
func TestKvCF2Truncation(t *testing.T) {
	n := kvSweepN(100000)
	r := rand.New(rand.NewSource(13))
	var worst float64
	var wu, wx float64
	for i := 0; i < n; i++ {
		u, x := kvCF2Point(r, i)
		a0, a1, ok := kvCF2Pair(u, x)
		b0, b1, ok2 := kvCF2PairTol(u, x, 0x1p-84, 0x1p-76)
		if !ok || !ok2 {
			t.Fatalf("u=%g x=%g: production %v, deep %v", u, x, ok, ok2)
		}
		d := math.Max(math.Abs((a0.Hi-b0.Hi)+(a0.Lo-b0.Lo))/b0.Hi, math.Abs((a1.Hi-b1.Hi)+(a1.Lo-b1.Lo))/b1.Hi)
		if d > worst {
			worst, wu, wx = d, u, x
		}
	}
	t.Logf("%d points: stopped vs 2^-84 worst 2^%.1f at u=%g x=%g", n, math.Log2(worst), wu, wx)
	if worst > 0x1p-60 {
		t.Errorf("truncation %.3g relative (limit 2^-60) at u=%g x=%g", worst, wu, wx)
	}
}

// TestKvCF2RangeEdges: the guards of Path 3 at their exact edges.
func TestKvCF2RangeEdges(t *testing.T) {
	// the lower edge x = kvCF2MinX, answered; one float below, declined
	if _, ok := kvFast(7.3, kvCF2MinX, false); !ok {
		t.Error("kvFast(7.3, kvCF2MinX) declined")
	}
	if _, ok := kvFast(7.3, math.Nextafter(kvCF2MinX, 0), false); ok {
		t.Error("kvFast(7.3, kvCF2MinX-) answered")
	}
	// the upper edge: CF2 answers just below 24, the chain of Path 2 at 24
	if r, ok := kvFast(7.3, math.Nextafter(kvAsymMinX, 0), false); !ok {
		t.Error("kvFast(7.3, 24-) declined")
	} else if r2, _ := kvCF2Chain(7.3, math.Nextafter(kvAsymMinX, 0), false); r != r2 {
		t.Errorf("kvFast(7.3, 24-) = %v, kvCF2Chain %v", r, r2)
	}
	if r, ok := kvFast(7.3, kvAsymMinX, false); !ok {
		t.Error("kvFast(7.3, 24) declined")
	} else if r2, _ := kvAsymChain(7.3, kvAsymMinX, false); r != r2 {
		t.Errorf("kvFast(7.3, 24) = %v, not the large-argument chain %v", r, r2)
	}
	// at x = 24 with a series that converges alone (4 v^2 - 1 <= 4 x) neither
	// path answers: besselAsymIK does, bit for bit
	if _, ok := kvFast(2.5, kvAsymMinX, false); ok {
		t.Error("kvFast(2.5, 24) answered")
	}
	if _, ok := kvCF2Chain(0.3, 0.01, false); ok {
		t.Error("kvCF2Chain(0.3, 0.01) answered past the cap")
	}
	// orders below 1 and the nearest integers: v = 1/2 reduces to u = -1/2
	for _, v := range []float64{0.001, 0.3, 0.5, 0.7, 0.999, 1.5, 31.5, 31.999999} {
		if _, ok := kvFast(v, 5, false); !ok {
			t.Errorf("kvFast(%g, 5) declined", v)
		}
	}
	// v = 32 is Debye, not this path (it would have answered at x = 10 as well)
	d, _ := kvDebye(32, 10, false)
	if got, ok := kvFast(32, 10, false); !ok || got != d {
		t.Errorf("kvFast(32, 10) = %v, %v; Debye %v", got, ok, d)
	}
	// the number of terms in double-double at its two edges
	for _, c := range []struct {
		x float64
		m int
	}{{kvCF2MinX, 4}, {math.Nextafter(3, 0), 4}, {3, 3}, {math.Nextafter(8, 0), 3}, {8, 2}, {23.9, 2}} {
		if got := kvCF2Terms(c.x); got != c.m {
			t.Errorf("kvCF2Terms(%g) = %d, want %d", c.x, got, c.m)
		}
	}
}

// TestKvCF2Caps: the iteration cap is reached below the region only (the CF2
// of x = 0.01 needs more than kvCF2MaxIters steps), which declines, and no
// point of the region reaches it.
func TestKvCF2Caps(t *testing.T) {
	if _, _, ok := kvCF2Pair(0.3, 0.01); ok {
		t.Error("kvCF2Pair(0.3, 0.01) converged within the cap")
	}
	if _, _, ok := kvCF2Pair(0.3, kvCF2MinX); !ok {
		t.Error("kvCF2Pair(0.3, kvCF2MinX) did not converge")
	}
}

// TestKvCF2HalfIntegers: v = n + 1/2 has u = +-1/2, a_1 = 0 and S = 1: the
// closed forms K_{1/2} = sqrt(pi/(2x)) e^-x, K_{3/2} = K_{1/2} (1 + 1/x), to
// the oracle (TestKvOracleClosedForms holds it) within the bound.
func TestKvCF2HalfIntegers(t *testing.T) {
	for _, v := range []float64{0.5, 1.5, 2.5, 7.5, 31.5} {
		for _, x := range []float64{kvCF2MinX, 2.7, 9.1, 23.99} {
			want := bigKve(v, x)
			if e := ivOracleUlps(Kve(v, x), want); e > kvFastBound {
				t.Errorf("Kve(%g, %g): %.3f ulp", v, x, e)
			}
		}
	}
}

// TestKvDeclinedBitIdentity: where Path 3 declines (below kvCF2MinX, and the
// orders it does not take), Kv and Kve are besselIK, bit for bit, on
// kvSweepN random points.
func TestKvDeclinedBitIdentity(t *testing.T) {
	n := kvSweepN(20000)
	r := rand.New(rand.NewSource(14))
	var checked int
	for i := 0; i < n; i++ {
		v := float64(r.Float64() * 40)
		x := math.Exp(float64(r.Float64()*math.Log(30)) - 3) // 0.05 .. 30 * 0.05
		if i%2 == 0 {
			x = float64(r.Float64() * kvCF2MinX)
		}
		if v == 0 || v == 1 || v == math.Trunc(v) {
			continue
		}
		if _, ok := kvFast(v, x, false); ok {
			continue
		}
		checked++
		_, k := besselIK(v, x, false, true, false)
		_, ke := besselIK(v, x, false, true, true)
		if got := Kv(v, x); got != k && !(got != got && k != k) {
			t.Fatalf("Kv(%g, %g) = %v, besselIK %v", v, x, got, k)
		}
		if got := Kve(v, x); got != ke && !(got != got && ke != ke) {
			t.Fatalf("Kve(%g, %g) = %v, besselIK %v", v, x, got, ke)
		}
	}
	if checked < n/4 {
		t.Errorf("only %d of %d points declined", checked, n)
	}
}
