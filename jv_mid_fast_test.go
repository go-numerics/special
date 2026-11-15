package special

import (
	"math"
	"math/big"
	"math/rand"
	"os"
	"testing"
)

// The tests of jv_mid_fast.go. Sizes: the defaults keep the suite short (the
// oracle costs about a millisecond a point); JV_SWEEP=1 runs the sweeps of the
// pull request (TestJvMidOracle: 60000 draws, J and Y, 120000 answers),
// JVMID_BITS_DUMP / JVMID_BITS_COMPARE the bit-identity check (TestJvMidBits).

// jvMidBound is the stated bound of the path in ulps: 0.5 for the rounding and
// 2^-57 |F| <= 2^-4 ulp for the certified error of the pair.
const jvMidBound = 0.5625

// jvMidSample draws a non-integer order and x in [1, 20]: orders uniform in
// |v| <= 40.5, a third of them in |v| <= 5, and x uniform with an eighth of the
// draws within 0.5 of an edge.
func jvMidSample(r *rand.Rand) (v, x float64) {
	for {
		v = float64(float64(r.Float64()*81) - 40.5)
		if r.Intn(3) == 0 {
			v = float64(float64(r.Float64()*10) - 5)
		}
		if v != math.Trunc(v) {
			break
		}
	}
	x = 1 + float64(r.Float64()*19)
	switch r.Intn(8) {
	case 0:
		x = 1 + float64(r.Float64()*0.5)
	case 1:
		x = 20 - float64(r.Float64()*0.5)
	}
	return v, x
}

// TestJvMidOracle: on a sweep of J and Y against the oracle of jv_oracle_test.go
// every answer is within jvMidBound ulp, and within the certified bound of the
// pair it rounds; the share it answers is reported.
func TestJvMidOracle(t *testing.T) {
	n := 300
	if os.Getenv("JV_SWEEP") != "" {
		n = 60000
	} else if testing.Short() {
		n = 100
	}
	r := rand.New(rand.NewSource(57))
	var worst [2]float64
	var asked, answered [2]int
	for i := 0; i < n; i++ {
		v, x := jvMidSample(r)
		for w := 0; w < 2; w++ {
			asked[w]++
			got, e, ok := jvMidFast(v, x, w == 1)
			if !ok {
				continue
			}
			answered[w]++
			want := bigJv(v, x)
			if w == 1 {
				want = bigYv(v, x)
			}
			u := jvUlps(got, want)
			if u > worst[w] {
				worst[w] = u
			}
			if u > jvMidBound {
				t.Fatalf("jvMidFast(%v, %v, Y=%v) = %v: %.4f ulp from the oracle", v, x, w == 1, got, u)
			}
			d, _ := bnew().Sub(bf(got), want).Float64()
			if math.Abs(d) > e+float64(0.5*ulp(got)) {
				t.Fatalf("jvMidFast(%v, %v, Y=%v): error %g exceeds its certificate %g + half ulp", v, x, w == 1, d, e)
			}
		}
	}
	t.Logf("J: %d of %d answered, worst %.4f ulp; Y: %d of %d, worst %.4f ulp",
		answered[0], asked[0], worst[0], answered[1], asked[1], worst[1])
	if answered[0] < asked[0]*98/100 || answered[1] < asked[1]*95/100 {
		t.Errorf("answered only %d of %d (J) and %d of %d (Y)", answered[0], asked[0], answered[1], asked[1])
	}
}

// jvMidBigSeries returns S(w) at x by the series at 700 bits.
func jvMidBigSeries(w, x float64) *big.Float {
	ivOracleInit()
	q := bjn().Quo(bjn().Mul(bj(x), bj(x)), bj(4))
	q.Neg(q)
	term := bj(1)
	sum := bj(1)
	for k := 1; k < 5000; k++ {
		den := bjn().Mul(bj(float64(k)), bjn().Add(bj(w), bj(float64(k))))
		term.Mul(term, q)
		term.Quo(term, den)
		sum.Add(sum, term)
		if float64(k)+w > 0 && float64(k)*(float64(k)+w) > x*x && term.Cmp(bjn().SetMantExp(bj(1), -400)) < 0 &&
			term.Cmp(bjn().SetMantExp(bj(-1), -400)) > 0 {
			break
		}
	}
	return sum
}

// TestJvMidSeries: S(w) against math/big within jvMidSeriesErr, on random
// orders of either sign and x in [1, 20]; the terms and the cancellation sabs
// / |S| are reported.
func TestJvMidSeries(t *testing.T) {
	n := 400
	if os.Getenv("JV_SWEEP") != "" {
		n = 20000
	} else if testing.Short() {
		n = 100
	}
	r := rand.New(rand.NewSource(58))
	worst, cancel, maxN := 0.0, 0.0, 0
	for i := 0; i < n; i++ {
		w, x := jvMidSample(r)
		qh, ql := dd2(x)
		s, sabs, tail, k := jvMidSeries(w, qh, ql, jvMidMaxTerms)
		if math.IsNaN(sabs) {
			t.Fatalf("jvMidSeries(%v, x=%v) reached the cap", w, x)
		}
		want := jvMidBigSeries(w, x)
		d, _ := bnew().Sub(bnew().Add(bf(s.Hi), bf(s.Lo)), want).Float64()
		bound := jvMidSeriesErr(k, sabs, tail)
		if math.Abs(d) > bound {
			t.Fatalf("jvMidSeries(%v, x=%v): error %g above its bound %g", w, x, d, bound)
		}
		if rel := math.Abs(d) / bound; rel > worst {
			worst = rel
		}
		wf, _ := want.Float64()
		if c := sabs / math.Abs(wf); c > cancel {
			cancel = c
		}
		maxN = max(maxN, k)
	}
	t.Logf("worst error / bound %.3g; cancellation up to 2^%.1f; at most %d terms", worst, math.Log2(cancel), maxN)
}

// dd2 returns x^2/4 as an exact pair.
func dd2(x float64) (qh, ql float64) {
	h, l := float64(x*x), math.FMA(x, x, -x*x)
	return float64(h * 0.25), float64(l * 0.25)
}

// TestJvMidSeriesStop holds the stopping rule that no refusal guards: the loop
// does not stop while k + w <= 0 (an order below zero, where the first ratios
// are not the bound of the later ones), and stops at once when the terms are
// negligible.
func TestJvMidSeriesStop(t *testing.T) {
	q := 1e-40
	if _, _, _, n := jvMidSeries(-3.5, q, 0, jvMidMaxTerms); n != 4 {
		t.Errorf("w = -3.5: stopped at term %d, want 4 (the first with k + w > 0)", n)
	}
	if _, _, _, n := jvMidSeries(3.5, q, 0, jvMidMaxTerms); n != 1 {
		t.Errorf("w = 3.5: stopped at term %d, want 1", n)
	}
	// large x^2/4: no stop while the ratio is above 1/2 (k = 12 gives
	// 100/(12 14.5) = 0.57, k = 13 gives 0.50)
	if _, _, _, n := jvMidSeries(2.5, 100, 0, jvMidMaxTerms); n < 13 {
		t.Errorf("w = 2.5, q = 100: stopped at term %d, before the ratio is below 1/2", n)
	}
}

// TestJvMidCap holds the cap of the series from both sides: the path answers
// with exactly the terms it needs and declines with one fewer, on a point of
// each branch (J of either sign, Y of either sign).
func TestJvMidCap(t *testing.T) {
	for _, c := range []struct {
		v, x float64
		y    bool
	}{{3.3, 12.3, false}, {-3.3, 12.3, false}, {3.3, 12.3, true}, {-3.3, 12.3, true}, {0.4, 1.7, true}, {38.5, 19.5, true}} {
		r0, _, n, ok := jvMidFastN(c.v, c.x, c.y, jvMidMaxTerms)
		if !ok {
			t.Fatalf("(%v, %v, Y=%v) declines", c.v, c.x, c.y)
		}
		r1, _, _, ok := jvMidFastN(c.v, c.x, c.y, n)
		if !ok || r1 != r0 {
			t.Errorf("(%v, %v, Y=%v): cap %d (the terms it needs) gives %v, %v; want %v", c.v, c.x, c.y, n, r1, ok, r0)
		}
		if _, _, _, ok := jvMidFastN(c.v, c.x, c.y, n-1); ok {
			t.Errorf("(%v, %v, Y=%v): cap %d answers; the series needs %d terms", c.v, c.x, c.y, n-1, n)
		}
	}
}

// TestJvMidGuards holds the range of the path by assertion: both sides of each
// edge, integers, and the non-finite arguments.
func TestJvMidGuards(t *testing.T) {
	above := func(f float64) float64 { return math.Nextafter(f, math.Inf(1)) }
	below := func(f float64) float64 { return math.Nextafter(f, 0) }
	for _, c := range []struct {
		v, x float64
		want bool
	}{
		{0.3, 1, true}, {0.3, below(1), false}, {0.3, 20, true}, {0.3, above(20), false},
		{40.5, 3.3, true}, {above(40.5), 3.3, false}, {-40.5, 3.3, true}, {-above(40.5), 3.3, false},
		{below(40.5), 3.3, true}, {-below(40.5), 3.3, true},
		{3, 5.5, false}, {-3, 5.5, false}, {0, 5.5, false}, {40, 5.5, false},
		{math.NaN(), 5.5, false}, {0.3, math.NaN(), false}, {math.Inf(1), 5.5, false},
		{math.Inf(-1), 5.5, false}, {0.3, math.Inf(1), false}, {0.3, 0, false}, {0.3, -5, false},
	} {
		for _, y := range []bool{false, true} {
			if _, _, ok := jvMidFast(c.v, c.x, y); ok != c.want {
				t.Errorf("jvMidFast(%v, %v, Y=%v): ok = %v, want %v", c.v, c.x, y, ok, c.want)
			}
		}
	}
}

// TestJvMidCertified holds the acceptance test at its boundary, for either sign
// of the value.
func TestJvMidCertified(t *testing.T) {
	for _, f := range []float64{1.25, -1.25, 0.001, -3e5} {
		e := float64(jvMidAccept * math.Abs(f))
		if !jvMidCertified(e, f) {
			t.Errorf("e = 2^-57 |f| refused for f = %v", f)
		}
		if jvMidCertified(math.Nextafter(e, 1), f) {
			t.Errorf("e just above 2^-57 |f| accepted for f = %v", f)
		}
	}
	if jvMidCertified(math.NaN(), 1) || jvMidCertified(1, math.NaN()) {
		t.Error("a NaN is accepted")
	}
	if jvMidCertified(1e-30, 0) {
		t.Error("a zero value is accepted with a nonzero bound")
	}
}

// TestJvMidTailDone holds the stopping comparison at its boundary.
func TestJvMidTailDone(t *testing.T) {
	for _, s := range []float64{1, 7.5, 1e9} {
		tb := float64(jvMidTailTol * s)
		if !jvMidTailDone(tb, s) {
			t.Errorf("tb = tol sabs refused for sabs = %v", s)
		}
		if jvMidTailDone(math.Nextafter(tb, 1), s) {
			t.Errorf("tb just above tol sabs accepted for sabs = %v", s)
		}
	}
}

// TestJvMidPrefix: the prefix (x/2)^nu / Gamma(nu+1) against math/big within
// the relative bound jvMidPrefix returns.
func TestJvMidPrefix(t *testing.T) {
	n := 400
	if os.Getenv("JV_SWEEP") != "" {
		n = 30000
	} else if testing.Short() {
		n = 100
	}
	r := rand.New(rand.NewSource(59))
	worst, worstBound := 0.0, 0.0
	for i := 0; i < n; i++ {
		nu := float64(r.Float64() * 40.5)
		switch i % 3 {
		case 0:
			nu = float64(r.Float64() * 2)
		case 1:
			nu = 8.5 + float64(r.Float64()*2) // the shift m changes at 9
		}
		x := 1 + float64(r.Float64()*19)
		p, rel := jvMidPrefix(nu, x)
		want := bigPrefix(nu, x)
		d, _ := bnew().Quo(bnew().Sub(bnew().Add(bf(p.Hi), bf(p.Lo)), want), want).Float64()
		if math.Abs(d) > rel {
			t.Fatalf("jvMidPrefix(%v, %v): relative error %g above its bound %g", nu, x, d, rel)
		}
		if math.Abs(d) > worst {
			worst, worstBound = math.Abs(d), rel
		}
	}
	t.Logf("worst relative error 2^%.1f (its bound 2^%.1f)", math.Log2(worst), math.Log2(worstBound))
}

// bigPrefix returns (x/2)^nu / Gamma(nu+1) at 700 bits.
func bigPrefix(nu, x float64) *big.Float {
	ivOracleInit()
	hx := bnew().Quo(bf(x), bf(2))
	lg, _ := bigGammaSigned(bnew().Add(bf(nu), bf(1)))
	lt := bnew().Mul(bf(nu), bigLog(hx))
	lt.Sub(lt, lg)
	return bjn().Set(bigExp(lt))
}

// TestJvMidLogGamma: log Gamma(y), y >= 10, against math/big within 2^-70.
func TestJvMidLogGamma(t *testing.T) {
	ivOracleInit()
	r := rand.New(rand.NewSource(60))
	worst := 0.0
	for i := 0; i < 300; i++ {
		y := 10 + float64(r.Float64()*32)
		if i%5 == 0 {
			y = 10 + float64(r.Float64()*0.01)
		}
		ls := jvMidLogGamma(y)
		d, _ := bnew().Sub(bnew().Add(bf(ls.Hi), bf(ls.Lo)), bigLogGamma(bf(y))).Float64()
		worst = math.Max(worst, math.Abs(d))
	}
	t.Logf("worst absolute error 2^%.1f", math.Log2(worst))
	if worst > 0x1p-68 {
		t.Errorf("log Gamma error 2^%.1f above 2^-68", math.Log2(worst))
	}
}

// TestJvMidSinCosPi: sin(pi nu) and cos(pi nu) against math/big within
// jvMidEpsT, relatively, including next to the integers and half-integers.
func TestJvMidSinCosPi(t *testing.T) {
	ivOracleInit()
	n := 1500
	if os.Getenv("JV_SWEEP") != "" {
		n = 100000
	} else if testing.Short() {
		n = 300
	}
	r := rand.New(rand.NewSource(61))
	worst := 0.0
	for i := 0; i < n; i++ {
		nu := float64(r.Float64() * 41)
		switch i % 4 {
		case 1:
			nu = float64(r.Intn(41)) + float64((r.Float64()-0.5)*math.Pow(2, -float64(1+r.Intn(50))))
		case 2:
			nu = float64(r.Intn(41)) + 0.5 + float64((r.Float64()-0.5)*math.Pow(2, -float64(1+r.Intn(50))))
		}
		if nu <= 0 {
			continue
		}
		s, c := jvMidSinCosPi(nu)
		// the oracle reduces by whole quarter turns in float64 (exact: a = nu -
		// m/2) and rotates; bigPi is good to about 2^-110 only, so pi nu itself
		// would swamp the relative error next to an integer
		m := math.Floor(float64(2*nu) + 0.5)
		a := nu - float64(m/2)
		za := bnew().Mul(bigPi, bf(a))
		sa := bigCos(bnew().Sub(bnew().Quo(bigPi, bf(2)), za))
		ca := bigCos(za)
		if a == 0 { // exact: bigCos(pi/2) is only as good as bigPi
			sa, ca = bnew(), bf(1)
		}
		var ws, wc *big.Float
		switch int(math.Mod(m, 4)) {
		case 0:
			ws, wc = sa, ca
		case 1:
			ws, wc = ca, bnew().Neg(sa)
		case 2:
			ws, wc = bnew().Neg(sa), bnew().Neg(ca)
		default:
			ws, wc = bnew().Neg(ca), sa
		}
		for _, p := range []struct {
			got  [2]float64
			want *big.Float
			name string
		}{{[2]float64{s.Hi, s.Lo}, ws, "sin"}, {[2]float64{c.Hi, c.Lo}, wc, "cos"}} {
			if p.want.Sign() == 0 { // cos at a half-integer: exactly zero
				if p.got[0] != 0 || p.got[1] != 0 {
					t.Fatalf("%s(pi %v) = %v, want 0", p.name, nu, p.got)
				}
				continue
			}
			d, _ := bnew().Quo(bnew().Sub(bnew().Add(bf(p.got[0]), bf(p.got[1])), p.want), p.want).Float64()
			worst = math.Max(worst, math.Abs(d))
			if math.Abs(d) > jvMidEpsT {
				t.Fatalf("%s(pi %v): relative error %g above %g", p.name, nu, d, jvMidEpsT)
			}
		}
	}
	t.Logf("worst relative error 2^%.1f", math.Log2(worst))
}

// TestJvMidWired: Jv and Yv answer with the fast path where it answers, and as
// the code that was there where it declines, and outside its range.
func TestJvMidWired(t *testing.T) {
	r := rand.New(rand.NewSource(62))
	var fast, declined int
	for i := 0; i < 400; i++ {
		v, x := jvMidSample(r)
		if i%4 == 0 {
			x = float64(r.Float64() * 1.2) // below the range, partly
		}
		for _, y := range []bool{false, true} {
			var got, old float64
			if y {
				got = Yv(v, x)
				_, old = oldJY(v, x, false, true)
			} else {
				got = Jv(v, x)
				old, _ = oldJY(v, x, true, false)
			}
			if m, _, ok := jvMidFast(v, x, y); ok {
				fast++
				if got != m {
					t.Errorf("Jv/Yv(%v, %v) Y=%v: %v, the fast path says %v", v, x, y, got, m)
				}
			} else {
				declined++
				if !sameBits(got, old) {
					t.Errorf("Jv/Yv(%v, %v) Y=%v: %v, the old code %v", v, x, y, got, old)
				}
			}
		}
	}
	if fast == 0 || declined == 0 {
		t.Errorf("fast %d, declined %d: one side of the test is empty", fast, declined)
	}
}

// oldJY is what Jv and Yv computed below the fast path: besselJYPos or
// besselJYRefl.
func oldJY(v, x float64, needJ, needY bool) (J, Y float64) {
	if v > 0 {
		return besselJYPos(v, x, needJ, needY)
	}
	return besselJYRefl(v, x, needJ, needY)
}

func init() {
	jvMidProbe = func(v, x float64, wantY bool) bool {
		_, _, ok := jvMidFast(v, x, wantY)
		return ok
	}
}
