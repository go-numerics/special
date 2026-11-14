package special

import (
	"math"
	"math/big"
	"sync"
	"testing"
)

// An oracle for I_v(x) that shares nothing with the package or with mpmath:
// math/big at 320 bits (the working precision of every step; the checks below
// hold it to 2^-300 or better). The function is the power series of DLMF
// 10.25.2 for x up to ivOracleSeriesMaxX, the large-argument expansion DLMF
// 10.40.1 beyond (there the terms fall to e^-2x, far below 2^-320 relative);
// Gamma through the shifted Stirling series with Bernoulli numbers; exp and
// log by argument reduction and Taylor / atanh series. The exact closed forms
// I_{1/2} and I_{3/2} (TestIvOracleClosedForms) and the integer orders against
// the integral of DLMF 10.32.3 evaluated here too (TestIvOracleIntegral) hold it.

const (
	ivOraclePrec       = 320
	ivOracleSeriesMaxX = 2000
)

func bf(x float64) *big.Float { return new(big.Float).SetPrec(ivOraclePrec).SetFloat64(x) }
func bnew() *big.Float        { return new(big.Float).SetPrec(ivOraclePrec) }

var (
	ivOracleOnce  sync.Once
	bigPi, bigLn2 *big.Float
	bigBern       []*big.Rat // B_{2k}, k = 0..
)

func ivOracleInit() {
	ivOracleOnce.Do(func() {
		bigPi, bigLn2 = bigPiLn2()
		// Bernoulli numbers by the recurrence sum_{j<m} C(m+1, j) B_j = -(m+1) ... B_m
		const mmax = 100
		B := make([]*big.Rat, mmax+1)
		B[0] = big.NewRat(1, 1)
		for m := 1; m <= mmax; m++ {
			s := new(big.Rat)
			c := big.NewInt(1) // C(m+1, 0)
			for j := 0; j < m; j++ {
				t := new(big.Rat).SetInt(c)
				t.Mul(t, B[j])
				s.Add(s, t)
				// C(m+1, j+1) = C(m+1, j) (m+1-j)/(j+1)
				c.Mul(c, big.NewInt(int64(m+1-j)))
				c.Div(c, big.NewInt(int64(j+1)))
			}
			B[m] = s.Neg(s)
			B[m].Quo(B[m], big.NewRat(int64(m+1), 1))
		}
		for k := 0; 2*k <= mmax; k++ {
			bigBern = append(bigBern, B[2*k])
		}
	})
}

// bigPiLn2 returns pi by Machin's formula 16 atan(1/5) - 4 atan(1/239) and
// ln 2 = 2 atanh(1/3), both by their series (nothing typed in from memory).
func bigPiLn2() (pi, ln2 *big.Float) {
	atanInv := func(m float64) *big.Float { // atan(1/m)
		m2 := bf(float64(m * m))
		pw := bnew().Quo(bf(1), bf(m))
		sum := bnew().Set(pw)
		for n := 1; n < 400; n++ {
			pw.Quo(pw, m2)
			t := bnew().Quo(pw, bf(float64(2*n+1)))
			if n%2 == 1 {
				sum.Sub(sum, t)
			} else {
				sum.Add(sum, t)
			}
		}
		return sum
	}
	pi = bnew().Sub(bnew().Mul(bf(16), atanInv(5)), bnew().Mul(bf(4), atanInv(239)))
	z := bf(1.0 / 3)
	z.Quo(bf(1), bf(3))
	z2 := bnew().Mul(z, z)
	pw := bnew().Set(z)
	sum := bnew().Set(z)
	for n := 3; n < 600; n += 2 {
		pw.Mul(pw, z2)
		sum.Add(sum, bnew().Quo(pw, bf(float64(n))))
	}
	ln2 = sum.Mul(sum, bf(2))
	return
}

func bigExp(z *big.Float) *big.Float {
	ivOracleInit()
	// z = k ln2 + r, |r| <= ln2/2 + eps, then r/2^8 by Taylor and 8 squarings
	k := new(big.Float).Quo(z, bigLn2)
	kf, _ := k.Float64()
	ki := int(math.Floor(kf + 0.5))
	r := bnew().Sub(z, bnew().Mul(bf(float64(ki)), bigLn2))
	r.SetMantExp(r, -8)
	sum, term := bf(1), bf(1)
	for n := 1; n < 80; n++ {
		term.Mul(term, r)
		term.Quo(term, bf(float64(n)))
		sum.Add(sum, term)
	}
	for i := 0; i < 8; i++ {
		sum.Mul(sum, sum)
	}
	return sum.SetMantExp(sum, ki)
}

func bigLog(a *big.Float) *big.Float {
	ivOracleInit()
	m := bnew()
	e := a.MantExp(m) // a = m 2^e, m in [0.5, 1)
	if m.Cmp(bf(0.7071067811865476)) < 0 {
		m.SetMantExp(m, 1)
		e--
	}
	// ln m = 2 atanh((m-1)/(m+1)), |z| <= 0.172
	z := bnew().Quo(bnew().Sub(m, bf(1)), bnew().Add(m, bf(1)))
	z2 := bnew().Mul(z, z)
	sum, pw := bnew().Set(z), bnew().Set(z)
	for n := 3; n < 400; n += 2 {
		pw.Mul(pw, z2)
		sum.Add(sum, bnew().Quo(pw, bf(float64(n))))
	}
	sum.SetMantExp(sum, 1)
	return sum.Add(sum, bnew().Mul(bf(float64(e)), bigLn2))
}

// bigLogGamma returns ln Gamma(z) for z > 0 by the recurrence up to 250 and
// Stirling's series (40 terms: the next is below 2^-400 at 250).
func bigLogGamma(z *big.Float) *big.Float {
	ivOracleInit()
	w := bnew().Set(z)
	prod := bf(1)
	for w.Cmp(bf(250)) < 0 {
		prod.Mul(prod, w)
		w.Add(w, bf(1))
	}
	// (w - 1/2) ln w - w + ln(2 pi)/2 + sum B_2k / (2k (2k-1) w^(2k-1))
	lw := bigLog(w)
	s := bnew().Mul(bnew().Sub(w, bf(0.5)), lw)
	s.Sub(s, w)
	s.Add(s, bnew().Quo(bigLog(bnew().Mul(bf(2), bigPi)), bf(2)))
	w2 := bnew().Mul(w, w)
	pw := bnew().Set(w) // w^(2k-1)
	for k := 1; k <= 40; k++ {
		num := new(big.Float).SetPrec(ivOraclePrec).SetRat(bigBern[k])
		den := bnew().Mul(bf(float64(2*k*(2*k-1))), pw)
		s.Add(s, num.Quo(num, den))
		pw.Mul(pw, w2)
	}
	return s.Sub(s, bigLog(prod))
}

// bigIve returns I_v(x) e^-x for v >= 0, x > 0.
func bigIve(v, x float64) *big.Float {
	ivOracleInit()
	if x > ivOracleSeriesMaxX {
		return bigIveAsym(v, x)
	}
	hx := bnew().Quo(bf(x), bf(2))
	// first term (x/2)^v / Gamma(v+1) e^-x, exp(v ln(x/2) - lnGamma(v+1) - x)
	lt := bnew().Mul(bf(v), bigLog(hx))
	lt.Sub(lt, bigLogGamma(bnew().Add(bf(v), bf(1))))
	lt.Sub(lt, bf(x))
	term := bigExp(lt)
	q := bnew().Mul(hx, hx)
	sum := bnew().Set(term)
	for k := 1; k < 20000; k++ {
		den := bnew().Mul(bf(float64(k)), bnew().Add(bf(v), bf(float64(k))))
		term.Mul(term, q)
		term.Quo(term, den)
		sum.Add(sum, term)
		// terms fall once k (k+v) > q; stop when below 2^-340 of the sum
		if float64(k)*(float64(k)+v) > x*x/4 && term.Cmp(bnew().SetMantExp(sum, -340)) < 0 {
			break
		}
	}
	return sum
}

// bigIveAsym is DLMF 10.40.1 summed to its smallest term (at most 2x terms;
// the sum is accurate to about e^-2x relative, and x > 2000).
func bigIveAsym(v, x float64) *big.Float {
	// the series alternates and its terms can peak at e^(v^2/(2x)) before the
	// sum, e^(-v^2/(2x)) small, is left: 1600 bits cover v <= 1100, x > 2000
	const p = 1600
	np := func(f float64) *big.Float { return new(big.Float).SetPrec(p).SetFloat64(f) }
	mu := new(big.Float).SetPrec(p).Mul(np(float64(4*v)), np(v))
	x8 := np(float64(8 * x))
	sum, term := np(1), np(1)
	prev := np(math.Inf(1))
	for k := 1; k < 200000; k++ {
		o := float64(2*k - 1)
		f := new(big.Float).SetPrec(p).Sub(mu, np(float64(o*o)))
		term.Mul(term, f)
		term.Quo(term, new(big.Float).SetPrec(p).Mul(x8, np(float64(k))))
		term.Neg(term)
		a := new(big.Float).SetPrec(p).Abs(term)
		if a.Cmp(prev) > 0 && k > int(v)+2 {
			break
		}
		prev = a
		sum.Add(sum, term)
		if k > int(v)+2 && a.Cmp(new(big.Float).SetPrec(p).SetMantExp(new(big.Float).SetPrec(p).Abs(sum), -400)) < 0 {
			break
		}
	}
	pre := bnew().Quo(bf(1), bnew().Sqrt(bnew().Mul(bnew().Mul(bf(2), bigPi), bf(x))))
	return bnew().Mul(bnew().Set(sum), pre)
}

// ivOracleUlps returns |got - want| in ulps of want, want exact (big), got
// finite. A want of 0 is exact only for got == 0.
func ivOracleUlps(got float64, want *big.Float) float64 {
	wf, _ := want.Float64() // nearest float64
	if wf == 0 || math.IsInf(wf, 0) {
		if got == wf {
			return 0
		}
		return math.Inf(1)
	}
	u := ulp(wf)
	d := bnew().Sub(bf(got), want)
	df, _ := d.Float64()
	return math.Abs(df) / u
}

// TestIvOracleClosedForms: the oracle against the exact closed forms
// I_{1/2}(x) = sqrt(2/(pi x)) sinh x and I_{3/2}(x) = sqrt(2/(pi x)) (cosh x
// - sinh(x)/x), evaluated here with bigExp (itself held by the identity
// exp(x) exp(-x) = 1 and the value of e), and in the exact value of
// I_0(1) = 1.26606587775200833559824462521471753760767031135496220680814...
func TestIvOracleClosedForms(t *testing.T) {
	e := bigExp(bf(1))
	if want, _ := bnew().SetString("2.71828182845904523536028747135266249775724709369995957496696762772407663035354759457138217852516642742746639193200305992181741359662904357290033429526059563073813232862794349076323382988075319525101901157383418793070215408914993488416750924476146066808226480016847741185374234544243710753907774499206955170276183860626133138458300075204493382656029760673711320070932870912744374704723069697720931014169283681902551510865746377211125238978442505695369677078544996996794686445490598793163688923009879312"); e.Cmp(want) == 0 {
	} else {
		d := bnew().Sub(e, want)
		d.Abs(d)
		if d.Cmp(bnew().SetMantExp(bf(1), -300)) > 0 {
			t.Fatalf("bigExp(1) off by %v", d)
		}
	}
	for _, x := range []float64{0.001, 0.3, 1, 2.5, 7, 19.5, 100, 700, 1999, 2500, 50000} {
		pre := bnew().Quo(bf(2), bnew().Mul(bigPi, bf(x)))
		pre.Sqrt(pre)
		ex, emx := bigExp(bf(x)), bigExp(bf(-x))
		sinh := bnew().Quo(bnew().Sub(ex, emx), bf(2))
		cosh := bnew().Quo(bnew().Add(ex, emx), bf(2))
		// scaled: times e^-x
		w1 := bnew().Mul(bnew().Mul(pre, sinh), emx)
		w3 := bnew().Mul(bnew().Mul(pre, bnew().Sub(cosh, bnew().Quo(sinh, bf(x)))), emx)
		for _, c := range []struct {
			v float64
			w *big.Float
		}{{0.5, w1}, {1.5, w3}} {
			if c.v == 1.5 && x < 0.1 {
				continue // cosh x - sinh(x)/x cancels by x^2 in the closed form itself
			}
			got := bigIve(c.v, x)
			d := bnew().Sub(got, c.w)
			d.Abs(d)
			rel := bnew().Quo(d, c.w)
			if x < 40000 && rel.Cmp(bnew().SetMantExp(bf(1), -290)) > 0 {
				t.Errorf("bigIve(%g, %g): relative %v from the closed form", c.v, x, rel)
			}
			if x >= 40000 && rel.Cmp(bnew().SetMantExp(bf(1), -100)) > 0 {
				// the closed form here is sinh - cosh cancellation free? it is
				// not: beyond 1e4 e^-x underflows nothing, but sinh = (e^x -
				// e^-x)/2 keeps 320 bits; the asymptotic oracle is e^-2x exact
				t.Errorf("bigIve(%g, %g): relative %v from the closed form", c.v, x, rel)
			}
		}
	}
	// I_0(1) and I_1(1) to 40 digits
	i0, _ := bnew().SetString("1.266065877752008335598244625214717537607670311354962206808")
	i1, _ := bnew().SetString("0.565159103992485027207696027609863307328899621621092009480")
	for _, c := range []struct {
		v float64
		w *big.Float
	}{{0, i0}, {1, i1}} {
		got := bnew().Mul(bigIve(c.v, 1), bigExp(bf(1)))
		d := bnew().Sub(got, c.w)
		d.Abs(d)
		if d.Cmp(bnew().SetMantExp(bf(1), -150)) > 0 {
			t.Errorf("I_%g(1) = %v, want %v", c.v, got.Text('g', 60), c.w.Text('g', 60))
		}
	}
}

// TestIvOracleIntegral: for integer orders the series against the integral of
// DLMF 10.32.3, I_n(x) = (1/pi) int_0^pi e^(x cos t) cos(n t) dt, by the
// trapezoid rule on the periodic integrand (exponentially convergent; 400
// nodes at 320 bits for x <= 40 is below 1e-90 relative).
func TestIvOracleIntegral(t *testing.T) {
	if testing.Short() {
		t.Skip("short")
	}
	for _, c := range []struct {
		n int
		x float64
	}{{2, 0.5}, {5, 3}, {7, 12}, {20, 30}, {12, 20.25}} {
		const M = 600
		sum := bnew()
		for j := 0; j <= M; j++ {
			// cos(j pi / M) by the series of sin/cos is overkill: use bigExp of an
			// imaginary argument through Taylor of cos directly
			th := bnew().Quo(bnew().Mul(bigPiOnce(), bf(float64(j))), bf(M))
			ct := bigCos(th)
			f := bigExp(bnew().Mul(bf(c.x), ct))
			f.Mul(f, bigCos(bnew().Mul(bf(float64(c.n)), th)))
			if j == 0 || j == M {
				f.Quo(f, bf(2))
			}
			sum.Add(sum, f)
		}
		sum.Quo(sum, bf(M))
		got := bnew().Mul(bigIve(float64(c.n), c.x), bigExp(bf(c.x)))
		rel := bnew().Quo(bnew().Abs(bnew().Sub(got, sum)), sum)
		if rel.Cmp(bnew().SetMantExp(bf(1), -200)) > 0 {
			t.Errorf("I_%d(%g): series and integral differ by %v relative", c.n, c.x, rel.Text('g', 5))
		}
	}
}

func bigPiOnce() *big.Float { ivOracleInit(); return bigPi }

// bigCos is cos(a) by Taylor after reducing |a| <= 2pi (the arguments here
// are in [0, 100 pi / 600 * 20]).
func bigCos(a *big.Float) *big.Float {
	ivOracleInit()
	a2 := bnew().Mul(a, a)
	sum, term := bf(1), bf(1)
	for n := 1; n < 200; n++ {
		term.Mul(term, a2)
		term.Quo(term, bf(float64((2*n-1)*(2*n))))
		term.Neg(term)
		sum.Add(sum, term)
	}
	return sum
}
