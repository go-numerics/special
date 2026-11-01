package dd

import (
	"math"
	"math/big"
	"math/rand"
	"strings"
	"testing"
)

const u = 0x1p-53

// mk builds a normalised double-double from a float64 and a signed fraction
// of its half-ulp, keeping exponents far from overflow and underflow.
func mk(hi float64, frac float64) DD {
	if hi != hi || math.IsInf(hi, 0) {
		hi = 1
	}
	m, e := math.Frexp(hi)
	e %= 200
	hi = math.Ldexp(m, e)
	if hi == 0 {
		hi = 1
	}
	if frac != frac || math.IsInf(frac, 0) {
		frac = 0
	}
	frac = math.Mod(frac, 1)
	lo := hi * u * frac
	s, t := FastTwoSum(hi, lo)
	return DD{s, t}
}

func rel(got DD, want *big.Float) float64 {
	if want.Sign() == 0 {
		if got.Hi == 0 {
			return 0
		}
		return math.Inf(1)
	}
	d := new(big.Float).SetPrec(2400).Sub(got.Big(), want)
	r, _ := new(big.Float).SetPrec(2400).Quo(d, want).Float64()
	return math.Abs(r)
}

func bigOf(a DD) *big.Float { return new(big.Float).SetPrec(2400).Set(a.Big()) }

func checkOps(t *testing.T, a, b DD) {
	t.Helper()
	pa, pb := bigOf(a), bigOf(b)
	sum := new(big.Float).SetPrec(2400).Add(pa, pb)
	if r := rel(Add(a, b), sum); r > 3*u*u {
		t.Fatalf("Add(%v,%v) rel %g > 3u^2", a, b, r)
	}
	diff := new(big.Float).SetPrec(2400).Sub(pa, pb)
	if r := rel(Sub(a, b), diff); r > 3*u*u {
		t.Fatalf("Sub(%v,%v) rel %g > 3u^2", a, b, r)
	}
	prod := new(big.Float).SetPrec(2400).Mul(pa, pb)
	if r := rel(Mul(a, b), prod); r > 5*u*u {
		t.Fatalf("Mul(%v,%v) rel %g > 5u^2", a, b, r)
	}
	if r := rel(MulF(a, b.Hi), new(big.Float).SetPrec(2400).Mul(pa, new(big.Float).SetFloat64(b.Hi))); r > 2*u*u {
		t.Fatalf("MulF rel %g > 2u^2", r)
	}
	if r := rel(AddF(a, b.Hi), new(big.Float).SetPrec(2400).Add(pa, new(big.Float).SetFloat64(b.Hi))); r > 2*u*u {
		t.Fatalf("AddF rel %g > 2u^2", r)
	}
	if r := rel(MulFF(a.Hi, b.Hi), new(big.Float).SetPrec(2400).Mul(new(big.Float).SetFloat64(a.Hi), new(big.Float).SetFloat64(b.Hi))); r != 0 {
		t.Fatalf("MulFF not exact: %g", r)
	}
	quo := new(big.Float).SetPrec(2400).Quo(pa, pb)
	if r := rel(Div(a, b), quo); r > 15*u*u+56*u*u*u {
		t.Fatalf("Div(%v,%v) rel %g > 15u^2+56u^3", a, b, r)
	}
	if r := rel(DivF(a, b.Hi), new(big.Float).SetPrec(2400).Quo(pa, new(big.Float).SetFloat64(b.Hi))); r > 15*u*u+56*u*u*u {
		t.Fatalf("DivF rel %g", r)
	}
	if a.Hi > 0 {
		sq := new(big.Float).SetPrec(2400).Sqrt(pa)
		// measured bound, not from the reference: one Newton step.
		if r := rel(Sqrt(a), sq); r > 8*u*u {
			t.Fatalf("Sqrt(%v) rel %g > 8u^2", a, r)
		}
	}
	if r := rel(Sqr(a), new(big.Float).SetPrec(2400).Mul(pa, pa)); r > 5*u*u {
		t.Fatalf("Sqr rel %g", r)
	}
}

func FuzzArith(f *testing.F) {
	f.Add(1.0, 0.5, 3.0, -0.25)
	f.Add(-7.5, 0.9, 7.5, 0.1)
	f.Add(1e30, -0.3, 1e-30, 0.7)
	f.Add(1.0, 0.0, 1.0, 0.0)
	f.Fuzz(func(t *testing.T, ah, af, bh, bf float64) {
		checkOps(t, mk(ah, af), mk(bh, bf))
	})
}

func TestArithRandom(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	for i := 0; i < 20000; i++ {
		a := mk(rng.NormFloat64()*math.Exp(rng.NormFloat64()*20), rng.Float64()*2-1)
		b := mk(rng.NormFloat64()*math.Exp(rng.NormFloat64()*20), rng.Float64()*2-1)
		checkOps(t, a, b)
	}
	// near-cancellation
	for i := 0; i < 2000; i++ {
		a := mk(rng.NormFloat64(), rng.Float64()*2-1)
		b := Neg(Add(a, DD{a.Hi * 0x1p-60 * rng.Float64(), 0}))
		checkOps(t, a, b)
	}
}

func TestFuzzHelperDegenerate(t *testing.T) {
	// NaN, Inf and zero inputs to the generator itself.
	for _, h := range []float64{math.NaN(), math.Inf(1), 0} {
		d := mk(h, math.NaN())
		if d.Hi != 1 || d.Lo != 0 {
			t.Fatalf("mk(%v) = %v", h, d)
		}
	}
	if d := mk(2, math.Inf(1)); d.Lo != 0 {
		t.Fatal(d)
	}
	// relative error against an exact zero.
	if rel(DD{}, new(big.Float)) != 0 || !math.IsInf(rel(DD{1, 0}, new(big.Float)), 1) {
		t.Fatal("rel zero")
	}
}

func TestSqrtNonPositive(t *testing.T) {
	if r := Sqrt(DD{0, 0}); r.Hi != 0 {
		t.Fatal(r)
	}
	if r := Sqrt(DD{-1, 0}); !r.IsNaN() {
		t.Fatal(r)
	}
}

func TestFloatLdexp(t *testing.T) {
	a := DD{1, 0x1p-60}
	if a.Float() != 1 {
		t.Fatal(a.Float())
	}
	l := Ldexp(a, 10)
	if l.Hi != 1024 || l.Lo != 0x1p-50 {
		t.Fatal(l)
	}
}

func TestDotSum(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	for _, n := range []int{0, 1, 2, 7, 100, 1000} {
		x := make([]float64, n)
		y := make([]float64, n)
		xd := make([]DD, n)
		yd := make([]DD, n)
		exact := new(big.Float).SetPrec(4000)
		exactD := new(big.Float).SetPrec(4000)
		exactS := new(big.Float).SetPrec(4000)
		abs := new(big.Float).SetPrec(4000)
		absD := new(big.Float).SetPrec(4000)
		absS := new(big.Float).SetPrec(4000)
		for i := range x {
			x[i] = rng.NormFloat64() * math.Exp(rng.NormFloat64()*5)
			y[i] = rng.NormFloat64() * math.Exp(rng.NormFloat64()*5)
			xd[i] = mk(x[i], rng.Float64()*2-1)
			yd[i] = mk(y[i], rng.Float64()*2-1)
			p := new(big.Float).SetPrec(4000).Mul(new(big.Float).SetFloat64(x[i]), new(big.Float).SetFloat64(y[i]))
			exact.Add(exact, p)
			abs.Add(abs, new(big.Float).Abs(p))
			pd := new(big.Float).SetPrec(4000).Mul(xd[i].Big(), yd[i].Big())
			exactD.Add(exactD, pd)
			absD.Add(absD, new(big.Float).Abs(pd))
			exactS.Add(exactS, new(big.Float).SetFloat64(x[i]))
			absS.Add(absS, new(big.Float).Abs(new(big.Float).SetFloat64(x[i])))
		}
		// Bound: n terms, each Add costs 3u^2 of the running magnitude, each
		// Mul 5u^2 of its term (test bound; sum of |terms| is the scale).
		tol := float64(n+1) * 8 * u * u
		for _, c := range []struct {
			name        string
			got         DD
			want, scale *big.Float
		}{
			{"Dot", Dot(x, y), exact, abs},
			{"DotDD", DotDD(xd, yd), exactD, absD},
			{"Sum", Sum(x), exactS, absS},
		} {
			d := new(big.Float).SetPrec(4000).Sub(c.got.Big(), c.want)
			if c.scale.Sign() == 0 {
				if d.Sign() != 0 {
					t.Fatalf("%s n=%d nonzero", c.name, n)
				}
				continue
			}
			r, _ := new(big.Float).Quo(d, c.scale).Float64()
			if math.Abs(r) > tol {
				t.Fatalf("%s n=%d err/scale %g > %g", c.name, n, r, tol)
			}
		}
	}
	// Beats float64: an ill-conditioned dot with exact answer 1.
	x := []float64{1e16, 1, -1e16}
	y := []float64{1, 1, 1}
	if r := Dot(x, y).Float(); r != 1 {
		t.Fatalf("Dot ill-conditioned = %v", r)
	}
	if r := AddProd(F(1), 3, 5); r.Float() != 16 {
		t.Fatal(r)
	}
	if r := AddProdDD(F(1), F(3), F(5)); r.Float() != 16 {
		t.Fatal(r)
	}
}

func TestDotPanics(t *testing.T) {
	for _, f := range []func(){
		func() { Dot([]float64{1}, nil) },
		func() { DotDD([]DD{{}}, nil) },
	} {
		func() {
			defer func() {
				if recover() == nil {
					t.Fatal("no panic")
				}
			}()
			f()
		}()
	}
}

func TestPredicates(t *testing.T) {
	if Abs(DD{-2, -1e-20}) != (DD{2, 1e-20}) || Abs(DD{2, 1e-20}) != (DD{2, 1e-20}) {
		t.Fatal("Abs")
	}
	if z := Abs(DD{math.Copysign(0, -1), 0}); math.Signbit(z.Hi) {
		t.Fatal("Abs -0")
	}
	if (DD{1, 0}).Sign() != 1 || (DD{-1, 0}).Sign() != -1 || (DD{}).Sign() != 0 || (DD{math.NaN(), 0}).Sign() != 0 {
		t.Fatal("Sign")
	}
	if !(DD{math.NaN(), 0}).IsNaN() || !(DD{0, math.NaN()}).IsNaN() || (DD{}).IsNaN() {
		t.Fatal("IsNaN")
	}
	if !(DD{math.Inf(-1), 0}).IsInf() || (DD{}).IsInf() {
		t.Fatal("IsInf")
	}
	cases := []struct {
		a, b DD
		w    int
	}{
		{DD{1, 0}, DD{2, 0}, -1}, {DD{2, 0}, DD{1, 0}, 1},
		{DD{1, 1e-20}, DD{1, 2e-20}, -1}, {DD{1, 2e-20}, DD{1, 1e-20}, 1},
		{DD{1, 1e-20}, DD{1, 1e-20}, 0}, {DD{math.NaN(), 0}, DD{1, 0}, 0},
	}
	for _, c := range cases {
		if g := Cmp(c.a, c.b); g != c.w {
			t.Fatalf("Cmp(%v,%v)=%d want %d", c.a, c.b, g, c.w)
		}
	}
}

func TestConversions(t *testing.T) {
	for _, v := range []int64{0, 1, -1, 1<<53 + 1, -(1<<53 + 1), math.MaxInt64, math.MinInt64, math.MinInt64 + 1, 123456789012345678} {
		d := FromInt64(v)
		want := new(big.Float).SetPrec(100).SetInt64(v)
		if d.Big().Cmp(want) != 0 {
			t.Fatalf("FromInt64(%d) = %v", v, d)
		}
	}
	// FromBig/Big round trip on representable values, and nearest rounding.
	rng := rand.New(rand.NewSource(3))
	for i := 0; i < 2000; i++ {
		a := mk(rng.NormFloat64()*10, rng.Float64()*2-1)
		if g := FromBig(a.Big()); g != a {
			t.Fatalf("round trip %v -> %v", a, g)
		}
		x := new(big.Float).SetPrec(300).Quo(big.NewFloat(1), big.NewFloat(3+float64(i)))
		g := FromBig(x)
		if r := rel(g, x); r > 0x1p-105 {
			t.Fatalf("FromBig rel %g", r)
		}
	}
	if g := FromBig(nil); !g.IsNaN() {
		t.Fatal(g)
	}
	if g := FromBig(new(big.Float).SetInf(true)); !math.IsInf(g.Hi, -1) {
		t.Fatal(g)
	}
	huge := new(big.Float).SetMantExp(big.NewFloat(1), 5000)
	if g := FromBig(huge); !math.IsInf(g.Hi, 1) || g.Lo != 0 {
		t.Fatal(g)
	}
	if g := FromBig(new(big.Float)); g != (DD{}) {
		t.Fatal(g)
	}
	// Big of a pair with a huge exponent gap is exact.
	b := (DD{1, 0x1p-1000}).Big()
	if b.Cmp(new(big.Float).Add(big.NewFloat(1), big.NewFloat(0x1p-1000))) != 0 {
		// SetPrec of NewFloat is 53: compare via exact construction instead.
		e := new(big.Float).SetPrec(2200).SetFloat64(1)
		e.Add(e, new(big.Float).SetFloat64(0x1p-1000))
		if b.Cmp(e) != 0 {
			t.Fatal("Big not exact")
		}
	}
}

func TestParseString(t *testing.T) {
	d, err := Parse("0.1")
	if err != nil {
		t.Fatal(err)
	}
	tenth, _, _ := big.ParseFloat("0.1", 10, 300, big.ToNearestEven)
	if r := rel(d, tenth); r > 0x1p-105 {
		t.Fatalf("Parse rel %g", r)
	}
	if _, err := Parse("zz"); err == nil || !strings.Contains(err.Error(), "dd: parse") {
		t.Fatal(err)
	}
	if s := d.String(); s != "0.1" {
		t.Fatal(s)
	}
	if (DD{math.NaN(), 0}).String() != "NaN" || (DD{math.Inf(1), 0}).String() != "+Inf" {
		t.Fatal("String specials")
	}
}

func TestElementaryEdges(t *testing.T) {
	// ExpMant with the reduction beyond 2^21 multiples of ln2/64.
	for _, x := range []float64{2e5, -2e5, 5e6, -9e6} {
		m, k := ExpMant(F(x))
		want := new(big.Float).SetPrec(300)
		// exp(x) = m 2^k: compare logs, ln m + k ln2 = x.
		lm := Log(m.Hi + m.Lo)
		got := Add(lm, Add(MulFF(float64(k), Ln2Hi), MulFF(float64(k), Ln2Lo)))
		_ = want
		if math.Abs(got.Float()-x) > 1e-9*math.Abs(x)*1e-3 {
			t.Fatalf("ExpMant(%g): log back %g", x, got.Float())
		}
	}
	// Log of a subnormal and of arguments above sqrt2 within an octave.
	for _, x := range []float64{5e-324, 1e-310, 1.5, 1.9, 0.75} {
		got := Log(x)
		if math.Abs(got.Float()-math.Log(x)) > 1e-15*math.Abs(math.Log(x)) {
			t.Fatalf("Log(%g)", x)
		}
	}
}
