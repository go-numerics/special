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
	w, _ := ibetaRatio(a, b, ibetaNewXY(x))
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
	_, w1 := ibetaRatio(a, b, ibetaNewXY(x))
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
}

// ibetaNewXY returns the pair for an exact x in (0, 1).
func ibetaNewXY(x float64) ibetaXY {
	y := 1 - x
	if x <= 0.5 {
		return ibetaXY{x: x, y: y, xd: dd.F(x), yd: sumDD(1, -x), lx: dd.LogAccurate(x), ly: igamLog1pDD(dd.F(-x)), xExact: true}
	}
	// y = 1 - x is exact for x >= 1/2 (Sterbenz)
	return ibetaXY{x: x, y: y, xd: dd.F(x), yd: dd.F(y), lx: igamLog1pDD(dd.F(-y)), ly: dd.LogAccurate(y), xExact: false}
}

func (v ibetaXY) swap() ibetaXY {
	return ibetaXY{x: v.y, y: v.x, xd: v.yd, yd: v.xd, lx: v.ly, ly: v.lx, xExact: !v.xExact}
}

// ibetaLogBeta returns log B(a, b) for a, b > 0 finite, as a double-double.
func ibetaLogBeta(a, b float64) dd.DD {
	if a > b {
		a, b = b, a
	}
	return logBetaPos(a, b)
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
	return dd.ExpMant(dd.Sub(l, ibetaLogBeta(a, b)))
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
	res := func(swapped bool, direct float64, directIsW bool) (float64, float64) {
		var w, w1 float64
		if directIsW {
			w, w1 = direct, 1-direct
		} else {
			w, w1 = 1-direct, direct
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
			return res(swapped, ibetaFpser(a0, b0, v0, eps), true)
		}
		if a0 < math.Min(eps, eps*b0) && b0*x0 <= 1 {
			return res(swapped, ibetaApser(a0, b0, x0, eps), false)
		}
		if math.Max(a0, b0) <= 1 {
			if math.Pow(x0, a0) <= 0.9 || a0 >= math.Min(0.2, b0) {
				return res(swapped, ibetaBpser(a0, b0, v0, eps), true)
			}
			if x0 >= 0.3 {
				return res(swapped, ibetaBpser(b0, a0, v0.swap(), eps), false)
			}
			w1 := ibetaBup(b0, a0, v0.swap(), 20, eps)
			w1 = ibetaBgrat(b0+20, a0, v0.swap(), w1, 15*eps)
			return res(swapped, w1, false)
		}
		if b0 <= 1 {
			return res(swapped, ibetaBpser(a0, b0, v0, eps), true)
		}
		if x0 >= 0.3 {
			return res(swapped, ibetaBpser(b0, a0, v0.swap(), eps), false)
		}
		if x0 < 0.1 && math.Pow(x0*b0, a0) <= 0.7 {
			return res(swapped, ibetaBpser(a0, b0, v0, eps), true)
		}
		if b0 > 15 {
			w1 := ibetaBgrat(b0, a0, v0.swap(), 0, 15*eps)
			return res(swapped, w1, false)
		}
		w1 := ibetaBup(b0, a0, v0.swap(), 20, eps)
		w1 = ibetaBgrat(b0+20, a0, v0.swap(), w1, 15*eps)
		return res(swapped, w1, false)
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
		return res(swapped, ibetaBpser(a0, b0, v0, eps), true)
	}
	if b0 < 40 {
		n := int(b0)
		b0 -= float64(n)
		if b0 == 0 {
			n--
			b0 = 1
		}
		w := ibetaBup(b0, a0, v0.swap(), n, eps)
		if x0 <= 0.7 {
			w += ibetaBpser(a0, b0, v0, eps)
			return res(swapped, w, true)
		}
		if a0 <= 15 {
			w += ibetaBup(a0, b0, v0, 20, eps)
			a0 += 20
		}
		w = ibetaBgrat(a0, b0, v0, w, 15*eps)
		return res(swapped, w, true)
	}
	if (a0 > b0 && (b0 <= 100 || lambda.Hi > 0.03*b0)) || (a0 <= b0 && (a0 <= 100 || lambda.Hi > 0.03*a0)) {
		return res(swapped, ibetaBfrac(a0, b0, v0, lambda.Hi, 15*eps), true)
	}
	return res(swapped, ibetaBasym(a0, b0, lambda, 100*eps), true)
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

// ibetaBpser returns I_x(a, b) by the power series (TOMS 708's BPSER),
// used when b <= 1 or b·x <= 0.7:
//
//	I_x(a,b) = x^a/(a B(a,b)) [1 + a Σ_{n>=1} (1-b)(2-b)...(n-b) x^n / (n! (a+n))].
func ibetaBpser(a, b float64, v ibetaXY, eps float64) float64 {
	// prefix x^a / (a B(a, b))
	l := dd.Sub(dd.MulF(v.lx, a), ibetaLogABeta(a, b))
	m, k := dd.ExpMant(l)
	if m.Hi == 0 || a <= 0.1*eps {
		return ibetaScale(m, k, 1)
	}
	// The terms are carried in double-double: with b > 1 they alternate
	// and a·Σ can cancel against the leading 1, and their product form
	// accumulates one rounding per term otherwise.
	sum := dd.F(0)
	c := dd.F(1)
	tol := eps / a
	for n := 1.0; n < 100000; n++ {
		c = dd.Mul(c, v.xd)
		c = dd.Sub(c, dd.DivF(dd.MulF(c, b), n)) // c·(1 - b/n)
		w := dd.Div(c, sumDD(a, n))
		sum = dd.Add(sum, w)
		if !(math.Abs(w.Hi) > tol*0x1p-8) {
			break
		}
	}
	r := dd.Mul(m, dd.AddF(dd.MulF(sum, a), 1))
	return math.Ldexp(r.Hi+r.Lo, k)
}

// ibetaBup returns I_x(a, b) - I_x(a+n, b) for a positive integer n
// (TOMS 708's BUP):
//
//	x^a y^b / (a B(a,b)) Σ_{i=0}^{n-1} d_i,  d_0 = 1,
//	d_{i+1} = d_i (a+b+i)/(a+1+i) x.
func ibetaBup(a, b float64, v ibetaXY, n int, eps float64) float64 {
	m, k := ibetaPowMant(a, b, v)
	m = dd.DivF(m, a)
	if n == 1 || m.Hi == 0 {
		return ibetaScale(m, k, 1)
	}
	x, y := v.x, v.y
	nm1 := n - 1
	// d_{i+1} = d_i (a+b+i)/(a+1+i) x in double-double (a+b and a+1 exact):
	// the n-1 products otherwise accumulate one rounding each.
	apb := sumDD(a, b)
	ap1 := sumDD(a, 1)
	d := dd.F(1)
	w := d
	next := func(i int) {
		fi := float64(i)
		d = dd.Div(dd.Mul(dd.Mul(d, dd.AddF(apb, fi)), v.xd), dd.AddF(ap1, fi))
		w = dd.Add(w, d)
	}
	// k0: index of the largest term; the terms before it are all added
	k0 := 0
	if b > 1 {
		// (y > 1e-4 here: BUP is only called with b > 1 on the side where
		// x <= 0.7 or with b < 1)
		r := (b-1)*x/y - a
		if r >= 1 {
			k0 = int(math.Min(r, float64(nm1)))
		}
	}
	for i := 0; i < k0; i++ {
		next(i)
	}
	for i := k0; i < nm1; i++ {
		next(i)
		if d.Hi <= eps*0x1p-8*w.Hi {
			break
		}
	}
	r := dd.Mul(m, w)
	return math.Ldexp(r.Hi+r.Lo, k)
}

// ibetaBfrac returns I_x(a, b) by the continued fraction of TOMS 708's
// BFRAC (a, b > 1, λ = (a+b)y - b >= 0).
func ibetaBfrac(a, b float64, v ibetaXY, lambda, eps float64) float64 {
	m, k := ibetaPowMant(a, b, v)
	if m.Hi == 0 {
		return 0
	}
	x, y := v.x, v.y
	c := 1 + lambda
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
	for i := 0; i < 100000; i++ {
		n++
		t := n / a
		w := n * (b - n) * x
		e := a / s
		alpha := (p * (p + c0) * e * e) * (w * x)
		e = (1 + t) / (c1 + t + t)
		beta := n + w/s + e*(c+n*yp1)
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
	return ibetaScale(m, k, r)
}

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
	f := dd.Neg(dd.Add(dd.MulF(igamLog1pmxDD(dd.Neg(dd.DivF(lambda, a))), a),
		dd.MulF(igamLog1pmxDD(dd.DivF(lambda, b)), b)))
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
		t = dd.Mul(t, dd.Sub(dd.F(x), dd.DivF(bx, j)))
		aj := dd.DivF(t, j)
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
