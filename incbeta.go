package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// The regularised incomplete beta function
//
//	I_x(a, b) = (1/B(a, b)) ∫₀ˣ t^(a-1) (1-t)^(b-1) dt   (DLMF 8.17.2)
//
// and its complement 1 - I_x(a, b) = I_(1-x)(b, a).
//
// Algorithm: DiDonato & Morris, "Algorithm 708: Significant digit
// computation of the incomplete beta function ratios", ACM TOMS 18(3),
// 1992 (BRATIO and its parts BPSER, BUP, BFRAC, BGRAT, BASYM, APSER, FPSER),
// ported from SciPy's C translation scipy/special/cdflib.c (BSD-3-Clause).
// It computes whichever of I and 1-I is the smaller directly, by a power
// series, a continued fraction, a sum shifting a or b to where those
// converge, or the asymptotic expansions for a >> b (BGRAT) and for large
// a and b near the mean (BASYM, a Temme-type uniform expansion in erfc).
//
// What is done differently, for accuracy (the cancellations and roundings
// that limit the float64 original):
//
//   - every power prefix x^a y^b / B(a, b) (BRCOMP, BRCMP1 and the prefixes
//     of BPSER, BGRAT and FPSER) is formed as a double-double logarithm,
//     with log y = log1p(-x) when y = 1 - x is not exact, and exponentiated
//     once; log B(a, b) comes from the cancellation-free logBetaPos;
//   - λ = (a+b)y - b, from which BASYM builds the exponent of its Gaussian
//     factor, is a double-double formed from whichever of x, y is exact
//     (the float64 λ carries an error of (a+b)·ulp(y), which for a, b ~ 1e15
//     is most of the result's digits), and the exponent -a·log1pmx(-λ/a)
//     - b·log1pmx(λ/b) and the argument of erfcx follow in double-double;
//   - the series and fractions are run to 2^-56 instead of 1e-15.

// ibetaEps is the tolerance of the series and continued fractions (the
// original uses max(machine epsilon, 1e-15)).
const ibetaEps = 0x1p-56

// BetaInc returns the regularised incomplete beta function
// I_x(a, b) = (1/B(a, b)) ∫₀ˣ t^(a-1) (1-t)^(b-1) dt (DLMF 8.17.2).
//
// Domain: a >= 0, b >= 0, 0 <= x <= 1, not both a and b zero or infinite.
// Special values: I_0 = 0 and I_1 = 1; for a = 0 (or b = +Inf), I_x = 1
// for x > 0; for b = 0 (or a = +Inf), I_x = 0 for x < 1; NaN for an
// argument outside the domain or NaN.
//
// Algorithm: DiDonato & Morris' TOMS 708 (BRATIO) with double-double
// prefixes; see the comment at the top of incbeta.go.
func BetaInc(a, b, x float64) float64 {
	if w, _, done := ibetaEdges(a, b, x); done {
		return w
	}
	w, _ := ibetaRatio(a, b, ibetaNewXY(x, a+b))
	return w
}

// BetaIncC returns the complement of the regularised incomplete beta
// function, 1 - I_x(a, b) = I_(1-x)(b, a), computed directly where it is
// the smaller of the two so that it keeps its relative accuracy for x near
// 1.
//
// Domain and special values: as BetaInc, with the complementary values.
//
// Algorithm: as BetaInc.
func BetaIncC(a, b, x float64) float64 {
	if _, w1, done := ibetaEdges(a, b, x); done {
		return w1
	}
	_, w1 := ibetaRatio(a, b, ibetaNewXY(x, a+b))
	return w1
}

// ibetaEdges handles the special values of I and 1-I (SciPy's
// conventions).
func ibetaEdges(a, b, x float64) (w, w1 float64, done bool) {
	nan := math.NaN()
	switch {
	case math.IsNaN(a) || math.IsNaN(b) || math.IsNaN(x) || a < 0 || b < 0 || x < 0 || x > 1:
		return nan, nan, true
	case (a == 0 && b == 0) || (math.IsInf(a, 1) && math.IsInf(b, 1)):
		return nan, nan, true
	case a == 0 || math.IsInf(b, 1):
		if x == 0 {
			return 0, 1, true
		}
		return 1, 0, true
	case b == 0 || math.IsInf(a, 1):
		if x == 1 {
			return 1, 0, true
		}
		return 0, 1, true
	case x == 0:
		return 0, 1, true
	case x == 1:
		return 1, 0, true
	}
	return 0, 0, false
}

// ibetaXY carries x and y = 1 - x, at least one of them exact, with their
// logarithms in double-double computed from the exact one.
type ibetaXY struct {
	x, y   float64
	xd, yd dd.DD // x and y exactly (one of them is a float64 already)
	lx, ly dd.DD
	xExact bool // x is exact (y may be rounded); else y is exact
	// lb = log B(a, b) when hasLB (set once by ibetaRatio: B is symmetric,
	// so it survives a swap of the parameters with x and y).
	lb    dd.DD
	hasLB bool
}

// ibetaNewXY returns the pair for an exact x in (0, 1). The logarithms
// are multiplied by at most scale (a + b: by a, b or ν = a + (b-1)/2) and
// carry the precision that keeps the products' absolute error below about
// 2^-64 (igamLogScaled, igamLog1pDDScaled).
func ibetaNewXY(x, scale float64) ibetaXY {
	y := 1 - x
	if x <= 0.5 {
		return ibetaXY{x: x, y: y, xd: dd.F(x), yd: sumDD(1, -x), lx: igamLogScaled(x, scale), ly: igamLog1pDDScaled(dd.F(-x), scale), xExact: true}
	}
	// y = 1 - x is exact for x >= 1/2 (Sterbenz)
	return ibetaXY{x: x, y: y, xd: dd.F(x), yd: dd.F(y), lx: igamLog1pDDScaled(dd.F(-y), scale), ly: igamLogScaled(y, scale), xExact: false}
}

func (v ibetaXY) swap() ibetaXY {
	return ibetaXY{x: v.y, y: v.x, xd: v.yd, yd: v.xd, lx: v.ly, ly: v.lx, xExact: !v.xExact, lb: v.lb, hasLB: v.hasLB}
}

// ibetaLogBeta returns log B(a, b) for a, b > 0 finite, as a double-double.
//
// For b >= 10 (a <= b) it is logBetaPos's cancellation-free form,
//
//	log Γ(a) - a log b - b log1pmx(a/b) - (a-1/2) log1p(a/b) + S(b) - S(a+b),
//
// with log1p(a/b) computed once and log1pmx taken from it where |a/b| >=
// 2^-10 (the cancellation then costs at most 11 of its 72 bits), and a/b
// from one division and its FMA remainder.
func ibetaLogBeta(a, b float64) dd.DD {
	if a > b {
		a, b = b, a
	}
	if b < 10 {
		return ibetaLogBetaSmall(a, b)
	}
	xh := a / b
	x := dd.DD{Hi: xh, Lo: math.FMA(-xh, b, a) / b}
	L := log1pDD(x)
	lpm := dd.Sub(L, x)
	if xh < 0x1p-10 {
		lpm = log1pmxDD(x)
	}
	t := dd.Neg(dd.MulF(dd.Log(b), a))
	t = dd.Sub(t, dd.MulF(lpm, b))
	t = dd.Sub(t, dd.Mul(sumDD(a, -0.5), L))
	t = dd.AddF(t, stirlingSeries(b)-stirlingSeries(a+b))
	return dd.Add(logGammaPos(a), t)
}

// ibetaLogBetaSmall returns log B(a, b) for 0 < a <= b < 10 as
// log(Γ(a) Γ(b) / Γ(s)) from the double-double Γ of gammaPos, with one
// logarithm where logBetaPos takes three (s = a + b rounded; s.Lo enters
// as -ψ(s)·s.Lo, where ψ is needed to 2^-9 only). Its absolute error, about
// 2^-62, is what the power prefix x^a y^b/B needs; LogBeta itself, which
// must keep its relative accuracy where log B ~ 0, stays on logBetaPos.
// Arguments below 2^-56 (Γ ~ 1/a, its 1/a in double-double) go to
// logBetaPos as well.
func ibetaLogBetaSmall(a, b float64) dd.DD {
	if a < 0x1p-56 {
		return logBetaPos(a, b)
	}
	s := sumDD(a, b)
	g := dd.Mul(gammaPos(a), gammaPos(b))
	var l dd.DD
	if s.Hi < 10 {
		l = logDD(dd.Div(g, gammaPos(s.Hi)))
	} else {
		l = dd.Sub(logDD(g), logGammaStirling(s.Hi))
	}
	return dd.AddF(l, -ibetaPsiApprox(s.Hi)*s.Lo)
}

// ibetaPsiApprox returns ψ(x) for x > 0 to about 2^-12 absolutely (enough
// for the first-order term ψ(s)·s.Lo): -1/x - γ + (π²/6)x below 0.1, else
// the recurrence to x + 6 and log t - 1/(2t) - 1/(12t²).
func ibetaPsiApprox(x float64) float64 {
	if x < 0.1 {
		return -1/x - 0.5772156649015329 + 1.6449340668482264*x
	}
	r := 0.0
	for ; x < 6; x++ {
		r -= 1 / x
	}
	return r + math.Log(x) - 0.5/x - 1/(12*x*x)
}

// ibetaLogABeta returns log(a·B(a, b)) as a double-double for a, b > 0
// finite. For a < 2^-60, log a and log B(a, b) ~ -log a would cancel to
// the last 2^-104·|log a| absolutely; when also a < 2^-60·b it is formed
// as log Γ(1+a) + [log Γ(b) - log Γ(a+b)] with the bracket -a·ψ(b) (its
// next term, a²ψ'(b)/2, is below 2^-60 of it).
func ibetaLogABeta(a, b float64) dd.DD {
	if a < 0x1p-60 && a < 0x1p-60*b {
		return dd.Sub(igamLog1pDD(gamma1pm1(a)), dd.MulF(digammaPos(b), a))
	}
	return dd.Add(dd.LogAccurate(a), ibetaLogBeta(a, b))
}

// ibetaPowMant returns m, k with x^a y^b / B(a, b) = m·2^k (TOMS 708's
// BRCOMP), from its double-double logarithm.
func ibetaPowMant(a, b float64, v ibetaXY) (dd.DD, int) {
	l := dd.Add(dd.MulF(v.lx, a), dd.MulF(v.ly, b))
	lb := v.lb
	if !v.hasLB {
		lb = ibetaLogBeta(a, b)
	}
	return dd.ExpMant(dd.Sub(l, lb))
}

// ibetaLdexpDD returns m·2^k as a double-double; where the value nears the
// subnormal range it is rounded once instead (its complement is then 1).
func ibetaLdexpDD(m dd.DD, k int) dd.DD {
	if k < -960 {
		return dd.F(math.Ldexp(m.Hi+m.Lo, k))
	}
	return dd.Ldexp(m, k)
}

// ibetaScale returns m·2^k·f rounded to float64.
func ibetaScale(m dd.DD, k int, f float64) float64 {
	w := dd.MulF(m, f)
	return math.Ldexp(w.Hi+w.Lo, k)
}

// ibetaRatio returns I_x(a, b) and 1 - I_x(a, b) for finite a, b > 0 and
// x in (0, 1): TOMS 708's BRATIO.
func ibetaRatio(a, b float64, v ibetaXY) (w, w1 float64) {
	const eps = ibetaEps
	// res returns (w, w1) from the directly computed one, undoing a swap.
	// The complement is formed from the double-double (one rounding of
	// the direct value would cost up to w/(1-w) ulp of the complement).
	res := func(swapped bool, direct dd.DD, directIsW bool) (float64, float64) {
		d := direct.Hi + direct.Lo
		c := (1 - direct.Hi) - direct.Lo
		w, w1 := d, c
		if !directIsW {
			w, w1 = c, d
		}
		if swapped {
			return w1, w
		}
		return w, w1
	}
	a0, b0, v0 := a, b, v
	swapped := false
	if math.Min(a0, b0) <= 1 {
		if v.x > 0.5 {
			swapped = true
			a0, b0, v0 = b, a, v.swap()
		}
		x0 := v0.x
		if b0 < math.Min(eps, eps*a0) {
			return res(swapped, dd.F(ibetaFpser(a0, b0, v0, eps)), true)
		}
		if a0 < math.Min(eps, eps*b0) && b0*x0 <= 1 {
			return res(swapped, dd.F(ibetaApser(a0, b0, x0, eps)), false)
		}
		if math.Max(a0, b0) <= 1 {
			if math.Pow(x0, a0) <= 0.9 || a0 >= math.Min(0.2, b0) {
				return res(swapped, ibetaBpserDD(a0, b0, v0, eps), true)
			}
			if x0 >= 0.3 {
				return res(swapped, ibetaBpserDD(b0, a0, v0.swap(), eps), false)
			}
			w1 := ibetaBup(b0, a0, v0.swap(), 20, eps)
			w1 = ibetaBgrat(b0+20, a0, v0.swap(), w1, 15*eps)
			return res(swapped, dd.F(w1), false)
		}
		if b0 <= 1 {
			return res(swapped, ibetaBpserDD(a0, b0, v0, eps), true)
		}
		if x0 >= 0.3 {
			return res(swapped, ibetaBpserDD(b0, a0, v0.swap(), eps), false)
		}
		if x0 < 0.1 && math.Pow(x0*b0, a0) <= 0.7 {
			return res(swapped, ibetaBpserDD(a0, b0, v0, eps), true)
		}
		if b0 > 15 {
			w1 := ibetaBgrat(b0, a0, v0.swap(), 0, 15*eps)
			return res(swapped, dd.F(w1), false)
		}
		w1 := ibetaBup(b0, a0, v0.swap(), 20, eps)
		w1 = ibetaBgrat(b0+20, a0, v0.swap(), w1, 15*eps)
		return res(swapped, dd.F(w1), false)
	}
	// a, b > 1. λ = (a+b)y - b = a - (a+b)x in double-double.
	lambda := ibetaLambda(a, b, v)
	if lambda.Hi < 0 {
		swapped = true
		a0, b0, v0 = b, a, v.swap()
		lambda = dd.Neg(lambda)
	}
	x0 := v0.x
	if b0 < 40 && b0*x0 <= 0.7 {
		return res(swapped, ibetaBpserDD(a0, b0, v0, eps), true)
	}
	if b0 < 40 {
		n := int(b0)
		b0 -= float64(n)
		if b0 == 0 {
			n--
			b0 = 1
		}
		// every prefix below is for the pair (a0, b0)
		v0.lb, v0.hasLB = ibetaLogBeta(a0, b0), true
		wd := ibetaBupDD(b0, a0, v0.swap(), n, eps)
		if x0 <= 0.7 {
			return res(swapped, dd.Add(wd, ibetaBpserDD(a0, b0, v0, eps)), true)
		}
		if a0 <= 15 {
			wd = dd.Add(wd, ibetaBupDD(a0, b0, v0, 20, eps))
			a0 += 20
		}
		w := ibetaBgrat(a0, b0, v0, wd.Hi+wd.Lo, 15*eps)
		return res(swapped, dd.F(w), true)
	}
	if (a0 > b0 && (b0 <= 100 || lambda.Hi > 0.03*b0)) || (a0 <= b0 && (a0 <= 100 || lambda.Hi > 0.03*a0)) {
		return res(swapped, dd.F(ibetaBfrac(a0, b0, v0, lambda, 15*eps)), true)
	}
	return res(swapped, dd.F(ibetaBasym(a0, b0, lambda, 100*eps)), true)
}

// ibetaLambda returns λ = (a+b)y - b = a - (a+b)x as a double-double,
// formed from whichever of x and y is exact.
func ibetaLambda(a, b float64, v ibetaXY) dd.DD {
	s := sumDD(a, b)
	if v.xExact {
		return dd.Sub(dd.F(a), dd.MulF(s, v.x))
	}
	return dd.Sub(dd.MulF(s, v.y), dd.F(b))
}

// ibetaBpserDD returns I_x(a, b), as a double-double, where TOMS 708 uses its power series BPSER
// (b <= 1 or b·x <= 0.7). The series is taken in Euler's form,
//
//	I_x(a,b) = x^a y^b/(a B(a,b)) Σ_{i>=0} (a+b)_i/(a+1)_i x^i
//
// (DLMF 8.17.8 / 15.8.1 applied to BPSER's 2F1(a, 1-b; a+1; x)), whose terms
// are all positive: BPSER's own terms alternate for b > 1 and a·Σ cancels
// against the leading 1. It is BUP run to convergence (ibetaPosSum).
func ibetaBpserDD(a, b float64, v ibetaXY, eps float64) dd.DD {
	m, k := ibetaPowMant(a, b, v)
	m = igamDivF(m, a)
	if !(m.Hi == 0 || a <= 0.1*eps) {
		m = dd.Mul(m, ibetaPosSum(a, b, v.xd, 0, 1<<30, eps))
	}
	return ibetaLdexpDD(m, k)
}

// ibetaBup returns I_x(a, b) - I_x(a+n, b) for a positive integer n
// (TOMS 708's BUP):
//
//	x^a y^b / (a B(a,b)) Σ_{i=0}^{n-1} d_i,  d_0 = 1,
//	d_{i+1} = d_i (a+b+i)/(a+1+i) x.
func ibetaBup(a, b float64, v ibetaXY, n int, eps float64) float64 {
	return ibetaBupDD(a, b, v, n, eps).Float()
}

// ibetaBupDD is ibetaBup as a double-double.
func ibetaBupDD(a, b float64, v ibetaXY, n int, eps float64) dd.DD {
	m, k := ibetaPowMant(a, b, v)
	m = igamDivF(m, a)
	if n == 1 || m.Hi == 0 {
		return dd.Ldexp(m, k)
	}
	// k0: index of the largest term; the terms before it are all added
	k0 := 0
	if b > 1 {
		// (y > 1e-4 here: BUP is only called with b > 1 on the side where
		// x <= 0.7 or with b < 1)
		r := (b-1)*v.x/v.y - a
		if r >= 1 {
			k0 = int(math.Min(r, float64(n-1)))
		}
	}
	return ibetaLdexpDD(dd.Mul(m, ibetaPosSum(a, b, v.xd, k0, n, eps)), k)
}

// ibetaPosSum returns Σ_{i=0}^{n-1} d_i, d_0 = 1, d_{i+1} = d_i (a+b+i)/(a+1+i)
// x, stopping once i >= k0 and d_i <= 2^-8 eps of the sum.
//
// Every quantity is a pair hi + lo carried without renormalisation: the
// factor (a+b+i) x/(a+1+i) to about 2^-104 (its numerator and denominator
// from TwoSum, the quotient th·(1/s2) corrected by its exact FMA remainder,
// so one division per term), the product
// d·f with TwoProd's error, the sum with TwoSum's. That is double-double
// accuracy (the terms are positive: no cancellation can amplify the low
// parts' own roundings) for about a third of the operations of the dd
// package, which renormalises after each one. Once the terms fall below
// 2^-16 of the sum the rest is summed in float64.
func ibetaPosSum(a, b float64, x dd.DD, k0, n int, eps float64) dd.DD {
	abh, abl := dd.TwoSum(a, b)
	a1h, a1l := dd.TwoSum(a, 1)
	xh, xl := x.Hi, x.Lo
	dh, dl := 1.0, 0.0
	sh, sl := 1.0, 0.0
	tol := eps * 0x1p-8
	i := 0
	for ; i < n-1; i++ {
		fi := float64(i)
		s1h, e1 := dd.TwoSum(abh, fi)
		s1l := e1 + abl
		s2h, e2 := dd.TwoSum(a1h, fi)
		s2l := e2 + a1l
		th := s1h * xh
		tl := math.FMA(s1h, xh, -th) + (s1h*xl + s1l*xh)
		r := 1 / s2h
		fh := th * r
		fl := (math.FMA(-fh, s2h, th) + tl - fh*s2l) * r
		ph := float64(dh * fh) // rounded: it feeds the TwoSum below
		dl = math.FMA(dh, fh, -ph) + (dh*fl + dl*fh)
		dh = ph
		var e float64
		sh, e = dd.TwoSum(sh, dh)
		sl += e + dl
		if i >= k0 && dh <= 0x1p-16*sh {
			i++
			break
		}
	}
	// The terms below 2^-16 of the sum in plain float64: their relative
	// errors grow by about 3 units of 2^-53 per term, and their weight
	// (below 2^-16 ρ^k) keeps the total under 2^-60 of the sum for a ratio
	// of decrease ρ up to 0.9, 2^-56 up to 0.98.
	tail := 0.0
	for ; i < n-1 && dh > tol*sh; i++ {
		fi := float64(i)
		dh *= xh * (abh + fi) / (a1h + fi)
		tail += dh
	}
	sl += tail
	h, l := dd.FastTwoSum(sh, sl)
	return dd.DD{Hi: h, Lo: l}
}

// ibetaBfrac returns I_x(a, b) by the continued fraction of TOMS 708's
// BFRAC (a, b > 1, λ = (a+b)y - b >= 0).
//
// BFRAC's fraction is r = 1/(c/c1 + α_1/(β_1 + α_2/(β_2 + ...))), c = 1 + λ,
// c1 = 1 + 1/a. The original runs its convergents forward, renormalised at
// every step, and its rounding errors accumulate to several ulps (it was
// the largest error of I for a, b in [1, 100)). The forward pass is kept
// only to find the depth; the terms are stored and the fraction is then
// summed backwards, t_n = α_n/(β_n + t_(n+1)), which is stable, with the
// leading term c/c1 = a(1+λ)/(a+1) and the top level (α_1 and β_1 from
// their closed forms) in double-double: t_1 reaches the result undamped,
// and BFRAC's float64 α_1 carries several roundings.
// The terms are stored on the stack up to depth ibetaBfracMax (deeper
// fractions, not met for a, b up to 1e15 near the mean, grow the store on
// the heap).
func ibetaBfrac(a, b float64, v ibetaXY, lambda dd.DD, eps float64) float64 {
	m, k := ibetaPowMant(a, b, v)
	if m.Hi == 0 {
		return 0
	}
	x, y := v.x, v.y
	c := 1 + lambda.Hi
	c0 := b / a
	c1 := 1 + 1/a
	yp1 := y + 1
	n := 0.0
	p := 1.0
	s := a + 1
	an := 0.0
	bn := 1.0
	anp1 := 1.0
	bnp1 := c / c1
	r := c1 / c
	var abuf, bbuf [ibetaBfracMax]float64
	al, be := abuf[:0], bbuf[:0]
	for i := 0; i < 100000; i++ {
		n++
		t := n / a
		w := n * (b - n) * x
		e := a / s
		alpha := (p * (p + c0) * e * e) * (w * x)
		e = (1 + t) / (c1 + t + t)
		beta := n + w/s + e*(c+n*yp1)
		al, be = append(al, alpha), append(be, beta)
		p = 1 + t
		s += 2
		t = alpha*an + beta*anp1
		an = anp1
		anp1 = t
		t = alpha*bn + beta*bnp1
		bn = bnp1
		bnp1 = t
		r0 := r
		r = anp1 / bnp1
		if !(math.Abs(r-r0) > eps*r) {
			break
		}
		an /= bnp1
		bn /= bnp1
		anp1 = r
		bnp1 = 1
	}
	t := 0.0
	for i := len(al) - 1; i >= ibetaBfracDD; i-- {
		t = al[i] / (be[i] + t)
	}
	// The top levels, which the result feels nearly undamped, from the
	// closed forms α_n = (a+n-1)(a+b+n-1) n(b-n) x²/(a+2n-1)² and β_n = n +
	// n(b-n)x/(a+2n-1) + (a+n)(1 + λ + n(1+y))/(a+2n+1) in double-double.
	td := dd.F(t)
	for i := min(ibetaBfracDD, len(al)) - 1; i >= 0; i-- {
		n := float64(i + 1)
		s := sumDD(a, 2*n-1)
		bn := sumDD(b, -n)
		w := dd.Mul(dd.MulF(bn, n), v.xd) // n(b-n)x
		al := dd.Div(dd.Mul(dd.Mul(sumDD(a, n-1), dd.AddF(sumDD(a, b), n-1)), dd.Mul(w, v.xd)), dd.Mul(s, s))
		e := dd.Div(sumDD(a, n), sumDD(a, 2*n+1))
		be := dd.Add(dd.AddF(dd.Div(w, s), n), dd.Mul(e, dd.Add(dd.AddF(lambda, 1), dd.MulF(dd.AddF(v.yd, 1), n))))
		td = dd.Div(al, dd.Add(be, td))
	}
	// c/c1 = a(1+λ)/(a+1) in double-double, then 1/(c/c1 + t)
	num := dd.MulF(dd.AddF(lambda, 1), a)
	b0 := dd.Div(num, sumDD(a, 1))
	wd := dd.Add(b0, td)
	rh := 1 / wd.Hi
	rl := (math.FMA(-rh, wd.Hi, 1) - rh*wd.Lo) * rh
	rr := dd.Mul(m, dd.DD{Hi: rh, Lo: rl})
	return math.Ldexp(rr.Hi+rr.Lo, k)
}

// ibetaBfracDD is the number of top levels of BFRAC's fraction summed in
// double-double: the first suffices (dense check: a, b in [1, 100) 2.84 ->
// 1.05 ulp; more levels change nothing measurable).
const ibetaBfracDD = 1

// ibetaBfracMax is the depth of BFRAC's fraction that ibetaBfrac keeps on
// the stack.
const ibetaBfracMax = 256

// ibetaBgrat returns w + I_x(a, b) by the asymptotic expansion of TOMS
// 708's BGRAT for a >= 15, b <= 1 (in powers of 1/ν, ν = a + (b-1)/2,
// with the incomplete gamma ratio Q(b, z), z = -ν log x). The prefix
// u = z^b e^-z Γ(a+b) / (Γ(b) Γ(a) ν^b) is a double-double logarithm:
// log u = b log(-log x) - z - log Γ(b) - [log Γ(a) - log Γ(a+b)] (the
// powers of z and ν combine exactly into (-log x)^b). On failure (a zero
// prefix) w is returned unchanged, as in the original.
func ibetaBgrat(a, b float64, v ibetaXY, w, eps float64) float64 {
	bm1 := sumDD(b, -1)
	nu := dd.Add(dd.F(a), dd.MulF(bm1, 0.5))
	lnx := v.lx
	z := dd.Neg(dd.Mul(nu, lnx))
	if b*z.Hi == 0 {
		return w
	}
	lu := dd.Sub(dd.MulF(igamLogDD(dd.Neg(lnx)), b), z)
	lu = dd.Sub(lu, logGammaPos(b))
	lu = dd.Sub(lu, logGammaDiff(b, a))
	um, uk := dd.ExpMant(lu)
	if um.Hi == 0 {
		return w
	}
	u := math.Ldexp(um.Hi+um.Lo, uk)
	// j = Q(b, z) / r, r = z^b e^-z / Γ(b), corrected to first order for
	// the low part of z: dj/dz = -1/z - j (b/z - 1).
	zh := z.Hi
	j := igamQOverPrefix(b, zh)
	j += z.Lo * (-1/zh - j*(b/zh-1))
	v2 := 0.25 / (nu.Hi * nu.Hi)
	t2 := 0.25 * lnx.Hi * lnx.Hi
	l := w / u
	sum := j
	t := 1.0
	cn := 1.0
	n2 := 0.0
	var c, d [30]float64
	bm := bm1.Hi + bm1.Lo
	for n := 1; n <= 30; n++ {
		bp2n := b + n2
		j = (bp2n*(bp2n+1)*j + (zh+bp2n+1)*t) * v2
		n2 += 2
		t *= t2
		cn /= n2 * (n2 + 1)
		c[n-1] = cn
		s := 0.0
		coef := b - float64(n)
		for i := 1; i < n; i++ {
			s += coef * c[i-1] * d[n-i-1]
			coef += b
		}
		d[n-1] = bm*cn + s/float64(n)
		dj := d[n-1] * j
		sum += dj
		if !(math.Abs(dj) > eps*(sum+l)) {
			break
		}
	}
	return w + ibetaScale(um, uk, sum)
}

// igamQOverPrefix returns Q(a, x) / (x^a e^-x / Γ(a)) for a > 0, x > 0
// (finite even where both underflow): from the continued fraction where
// it applies, else as the quotient.
func igamQOverPrefix(a, x float64) float64 {
	if x >= 1.1 && x-1/(3*x) >= a {
		return 1 / igamFractionDen(a, x)
	}
	_, q := igamPQ(a, x)
	return q / dd.Exp(igamLogPrefix(a, x))
}

// ibetaBasym returns I_x(a, b) for large a, b (>= 15) by TOMS 708's
// asymptotic expansion BASYM (a uniform expansion in erfc, after Temme),
// for λ = (a+b)y - b >= 0 given in double-double.
func ibetaBasym(a, b float64, lambda dd.DD, eps float64) float64 {
	const (
		e0 = 1.12837916709551257389615890312154517  // 2/sqrt(pi)
		e1 = 0.353553390593273762200422181052424519 // 2^(-3/2)
	)
	const num = 20
	var a0, b0, c, d [num + 1]float64
	var h, r0, r1, w0 float64
	if a < b {
		h = a / b
		r0 = 1 / (1 + h)
		r1 = (b - a) / b
		w0 = 1 / math.Sqrt(a*(1+h))
	} else {
		h = b / a
		r0 = 1 / (1 + h)
		r1 = (b - a) / a
		w0 = 1 / math.Sqrt(b*(1+h))
	}
	// f = a·rlog1(-λ/a) + b·rlog1(λ/b), rlog1(t) = t - log(1+t) = -log1pmx(t)
	f := dd.Neg(dd.Add(dd.MulF(igamLog1pmxDD(dd.Neg(igamDivF(lambda, a))), a),
		dd.MulF(igamLog1pmxDD(igamDivF(lambda, b)), b)))
	tm, tk := dd.ExpMant(dd.Neg(f))
	if tm.Hi == 0 {
		return 0
	}
	z0d := dd.Sqrt(f)
	z0 := z0d.Hi
	z := 0.5 * (z0 / e1)
	z2 := f.Hi + f.Hi
	a0[0] = (2.0 / 3.0) * r1
	c[0] = -0.5 * a0[0]
	d[0] = -c[0]
	j0 := (0.5 / e0) * ibetaErfcx(z0d, f)
	j1 := e1
	sum := j0 + d[0]*w0*j1
	s := 1.0
	h2 := h * h
	hn := 1.0
	w := w0
	znm1 := z
	zn := z2
	for n := 2; n <= num; n += 2 {
		hn *= h2
		a0[n-1] = 2 * r0 * (1 + h*hn) / (float64(n) + 2)
		s += hn
		a0[n] = 2 * r1 * s / (float64(n) + 3)
		for i := n; i <= n+1; i++ {
			r := -0.5 * (float64(i) + 1)
			b0[0] = r * a0[0]
			for m := 2; m <= i; m++ {
				bsum := 0.0
				for j := 1; j < m; j++ {
					mmj := m - j
					bsum += (float64(j)*r - float64(mmj)) * a0[j-1] * b0[mmj-1]
				}
				b0[m-1] = r*a0[m-1] + bsum/float64(m)
			}
			c[i-1] = b0[i-1] / (float64(i) + 1)
			dsum := 0.0
			for j := 1; j < i; j++ {
				dsum += d[i-j-1] * c[j-1]
			}
			d[i-1] = -(dsum + c[i-1])
		}
		j0 = e1*znm1 + (float64(n)-1)*j0
		j1 = e1*zn + float64(n)*j1
		znm1 *= z2
		zn *= z2
		w *= w0
		t0 := d[n-1] * w * j0
		w *= w0
		t1 := d[n] * w * j1
		sum += t0 + t1
		if math.Abs(t0)+math.Abs(t1) <= eps*sum {
			break
		}
	}
	// u = exp(-bcorr(a, b)), bcorr = del(a) + del(b) - del(a+b) with del
	// the Stirling series remainder.
	u := math.Exp(-(stirlingSeries(a) + stirlingSeries(b) - stirlingSeries(a+b)))
	return ibetaScale(tm, tk, e0*u*sum)
}

// ibetaErfcx returns erfcx(z) = exp(z²) erfc(z) for a double-double
// z >= 0 with z² = f.
func ibetaErfcx(z, f dd.DD) float64 {
	if z.Hi >= 0.5 {
		return erfcxCorrected(z).Float()
	}
	return math.Exp(f.Hi) * erfcDD(z).Float()
}

// ibetaApser returns I_(1-x)(b, a) for a <= min(eps, eps·b), b·x <= 1,
// x <= 1/2 (TOMS 708's APSER, a very small).
func ibetaApser(a, b, x, eps float64) float64 {
	// c + s carries a cancellation (log x against ψ(b), by a factor of
	// about 10 at b ~ 10): every part is a double-double.
	g := dd.DD{Hi: 0.5772156649015329, Lo: -4.942915152430645e-18}
	bx := dd.MulFF(b, x)
	t := dd.Sub(dd.F(x), bx)
	var c dd.DD
	if b*eps > 0.02 {
		c = dd.Add(igamLogDD(bx), g)
	} else {
		c = dd.Add(dd.Add(dd.LogAccurate(x), digammaPos(b)), g)
	}
	c = dd.Add(c, t)
	tol := 5 * eps * math.Abs(c.Hi) * 0x1p-8
	s := dd.F(0)
	for j := 2.0; j < 100000; j++ {
		t = dd.Mul(t, dd.Sub(dd.F(x), igamDivF(bx, j)))
		aj := igamDivF(t, j)
		s = dd.Add(s, aj)
		if math.Abs(aj.Hi) <= tol {
			break
		}
	}
	return dd.MulF(dd.Add(c, s), -a).Float()
}

// ibetaFpser returns I_x(a, b) for b < min(eps, eps·a), x <= 1/2 (TOMS
// 708's FPSER, b very small): x^a (b/a) [1 + a Σ_{n>=1} x^n/(a+n)]
// (1/B(a, b) ~ b there).
func ibetaFpser(a, b float64, v ibetaXY, eps float64) float64 {
	x := v.x
	m, k := dd.F(1), 0
	if !(a <= 1e-3*eps) {
		m, k = dd.ExpMant(dd.MulF(v.lx, a))
	}
	tol := eps / a
	an := a + 1
	t := x
	s := t / an
	for i := 0; i < 100000; i++ {
		an++
		t *= x
		c := t / an
		s += c
		if !(math.Abs(c) > tol) {
			break
		}
	}
	return ibetaScale(dd.MulF(m, b/a), k, 1+a*s)
}
