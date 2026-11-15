package special

import (
	"math"
	"math/big"
	"math/rand"
	"os"
	"testing"

	"github.com/go-numerics/special/dd"
)

// An oracle for J_v(x) and Y_v(x), v real not an integer, 0 < x <= 120, that
// shares nothing with the package or with mpmath: math/big through the helpers
// of i_oracle_test.go. J_v is its power series (DLMF 10.2.2) summed at 700
// bits (it cancels like e^x: 2^173 at x = 120), with the prefix
// (x/2)^v/Gamma(v+1) from bigLogGamma at 320 bits (Gamma at a negative
// argument by the recurrence up to a positive one); Y_v = (J_v cos(v pi) -
// J_-v)/sin(v pi) (DLMF 10.2.3). TestJvOracleClosedForms holds it against
// J_{1/2}, J_{-1/2}, Y_{1/2}.

const jvOraclePrec = 700

func bj(x float64) *big.Float { return new(big.Float).SetPrec(jvOraclePrec).SetFloat64(x) }
func bjn() *big.Float         { return new(big.Float).SetPrec(jvOraclePrec) }

// bigGammaSigned returns ln|Gamma(w)| and its sign, w not a non-positive integer.
func bigGammaSigned(w *big.Float) (*big.Float, float64) {
	z := bnew().Set(w)
	prod := bf(1)
	for z.Cmp(bf(1)) < 0 {
		prod.Mul(prod, z)
		z.Add(z, bf(1))
	}
	lg := bigLogGamma(z)
	sg := 1.0
	if prod.Sign() < 0 {
		sg = -1
		prod.Neg(prod)
	}
	return lg.Sub(lg, bigLog(prod)), sg
}

// bigJv returns J_v(x), v not an integer.
func bigJv(v, x float64) *big.Float {
	ivOracleInit()
	hx := bnew().Quo(bf(x), bf(2))
	lg, sg := bigGammaSigned(bnew().Add(bf(v), bf(1))) // v + 1 is not a float64
	lt := bnew().Mul(bf(v), bigLog(hx))
	lt.Sub(lt, lg)
	term := bjn().Set(bigExp(lt))
	if sg < 0 {
		term.Neg(term)
	}
	q := bjn().Mul(bjn().Set(hx), bjn().Set(hx))
	q.Neg(q)
	sum := bjn().Set(term)
	for k := 1; k < 20000; k++ {
		den := bjn().Mul(bj(float64(k)), bjn().Add(bj(v), bj(float64(k))))
		term.Mul(term, q)
		term.Quo(term, den)
		sum.Add(sum, term)
		at := bjn().Abs(term)
		if float64(k)*(float64(k)+v) > x*x/4 && at.Cmp(bjn().SetMantExp(bjn().Abs(sum), -300)) < 0 &&
			at.Cmp(bjn().SetMantExp(bj(1), -300)) < 0 {
			break
		}
	}
	return sum
}

// bigYv returns Y_v(x), v not an integer.
func bigYv(v, x float64) *big.Float {
	ivOracleInit()
	jp, jm := bigJv(v, x), bigJv(-v, x)
	vp := bnew().Mul(bigPi, bf(v))
	c := bigCos(vp)
	s := bigCos(bnew().Sub(bnew().Quo(bigPi, bf(2)), vp))
	y := bjn().Sub(bjn().Mul(jp, c), jm)
	return y.Quo(y, s)
}

func TestJvOracleClosedForms(t *testing.T) {
	ivOracleInit()
	for _, x := range []float64{0.7, 3, 21.5, 60, 119} {
		pre := bnew().Quo(bf(2), bnew().Mul(bigPi, bf(x)))
		pre.Sqrt(pre)
		sinx := bigCos(bnew().Sub(bnew().Quo(bigPi, bf(2)), bf(x)))
		cosx := bigCos(bf(x))
		for _, c := range []struct {
			name string
			got  *big.Float
			want *big.Float
		}{
			{"J1/2", bigJv(0.5, x), bnew().Mul(pre, sinx)},
			{"J-1/2", bigJv(-0.5, x), bnew().Mul(pre, cosx)},
			{"Y1/2", bigYv(0.5, x), bnew().Neg(bnew().Mul(pre, cosx))},
		} {
			d := bnew().Sub(c.got, c.want)
			d.Abs(d)
			if d.Cmp(bnew().SetMantExp(bf(1), -120)) > 0 {
				t.Errorf("%s(%g) differs from the closed form by %v", c.name, x, d.Text('g', 4))
			}
		}
	}
}

// jvUlps is |got - want| in ulps of want.
func jvUlps(got float64, want *big.Float) float64 { return ivOracleUlps(got, want) }

// hankelSample draws a non-integer order (either sign, |v| <= 40) and x in
// (20, 120].
func hankelSample(r *rand.Rand) (v, x float64) {
	for {
		v = float64(hankelU(r) * hankelMaxOrder)
		if r.Intn(3) == 0 {
			v = float64(hankelU(r) * 5)
		}
		if v != math.Trunc(v) {
			break
		}
	}
	return v, 20 + float64(r.Float64()*100)
}

// TestHankelOracle: on a sweep of J and Y at orders |v| <= 40 and 20 < x <=
// 120 against the oracle, every answer is within 0.63 ulp, and within the
// certified bound of the pair it rounds; the share it answers is reported.
func TestHankelOracle(t *testing.T) {
	n := 250
	if os.Getenv("JV_SWEEP") != "" {
		n = 60000 // 2 x ~53000 answers: the sweep of the PR (about 3 minutes)
	} else if testing.Short() {
		n = 600
	}
	r := rand.New(rand.NewSource(56))
	var worst [2]float64
	var asked, answered, askedS, answeredS [2]int // S: |v| <= 5
	for i := 0; i < n; i++ {
		v, x := hankelSample(r)
		for w := 0; w < 2; w++ {
			asked[w]++
			if math.Abs(v) <= 5 {
				askedS[w]++
			}
			got, e, ok := hankelFast(v, x, w == 1)
			if !ok {
				continue
			}
			answered[w]++
			if math.Abs(v) <= 5 {
				answeredS[w]++
			}
			var want *big.Float
			if w == 0 {
				want = bigJv(v, x)
			} else {
				want = bigYv(v, x)
			}
			u := jvUlps(got, want)
			if u > worst[w] {
				worst[w] = u
			}
			if u > 0.63 {
				t.Fatalf("hankelFast(%v, %v, Y=%v) = %v: %.3f ulp from the oracle", v, x, w == 1, got, u)
			}
			d := bnew().Sub(bf(got), want)
			df, _ := d.Float64()
			if math.Abs(df) > e+float64(0.5*ulp(got)) {
				t.Fatalf("hankelFast(%v, %v, Y=%v): error %g exceeds its certificate %g + half ulp", v, x, w == 1, df, e)
			}
		}
	}
	t.Logf("J: %d of %d answered, worst %.3f ulp; Y: %d of %d, worst %.3f ulp",
		answered[0], asked[0], worst[0], answered[1], asked[1], worst[1])
	for w := 0; w < 2; w++ {
		if answeredS[w] < askedS[w]*8/10 {
			t.Errorf("kind %d: only %d of %d answered at |v| <= 5", w, answeredS[w], askedS[w])
		}
	}
}

// TestHankelSinCosOracle holds hankelSinCos within hankelTrigErr of the true
// sine and cosine on random double-double angles |z| <= pi/4.
func TestHankelSinCosOracle(t *testing.T) {
	ivOracleInit()
	n := 1500
	if os.Getenv("JV_SWEEP") != "" {
		n = 1000000
	} else if testing.Short() {
		n = 2000
	}
	r := rand.New(rand.NewSource(8))
	worst := 0.0
	for i := 0; i < n; i++ {
		zh := float64(hankelU(r) * (math.Pi / 4))
		if i%7 == 0 {
			zh = float64(zh * math.Pow(2, -float64(r.Intn(60))))
		}
		zl := float64(float64(hankelU(r)*0x1p-53) * zh)
		sh, sl, ch, cl := hankelSinCos(dd.DD{Hi: zh, Lo: zl})
		z := bnew().Add(bf(zh), bf(zl))
		wc := bigCos(z)
		ws := bigCos(bnew().Sub(bnew().Quo(bigPi, bf(2)), z))
		es := math.Abs(relDDabs(sh, sl, ws))
		ec := math.Abs(relDDabs(ch, cl, wc))
		if es > worst {
			worst = es
		}
		if ec > worst {
			worst = ec
		}
		if es > hankelTrigErr || ec > hankelTrigErr {
			t.Fatalf("hankelSinCos(%v + %v): errors %g, %g above %g", zh, zl, es, ec, hankelTrigErr)
		}
	}
	t.Logf("worst absolute error %g = 2^%.1f", worst, math.Log2(worst))
}

// relDDabs returns (h + l) - want, absolute.
func relDDabs(h, l float64, want *big.Float) float64 {
	d := bnew().Sub(bnew().Add(bf(h), bf(l)), want)
	f, _ := d.Float64()
	return f
}
