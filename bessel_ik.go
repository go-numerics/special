package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Modified Bessel functions I and K of real order, after Boost.Math
// bessel_ik.hpp (Xiaogang Zhang, John Maddock; Boost Software License 1.0):
// Temme's series for K near the origin (N. M. Temme, "On the numerical
// evaluation of the modified Bessel function of the third kind", J. Comput.
// Phys. 19, 1975), Steed's continued fraction CF2 for K (I. J. Thompson and
// A. R. Barnett, J. Comput. Phys. 64, 1986), CF1 and the Wronskian for I.
// Everything is carried exponentially scaled (I e^-x, K e^x) so the scaled
// functions never overflow internally, and the large-x expansions (DLMF
// 10.40.1-2) are used wherever they converge.

// besselKSeries returns K_nu(x) = m * 2^e for non-integer 0 < nu < 2 and
// small x by the two power series of K = (pi/2)(I_{-nu} - I_nu)/sin(nu pi)
// (DLMF 10.27.4, 10.25.2):
//
//	K = Gamma(nu)/2 (x/2)^-nu sum (x^2/4)^k/(k! (1-nu)_k)
//	    - pi/(2 sin(nu pi)) (x/2)^nu/Gamma(nu+1) sum (x^2/4)^k/(k! (1+nu)_k),
//
// ok only where the second part is below 1/4 of the first (Temme's series
// loses about |nu log(x/2)| ulps through exp(+-nu log(x/2)) there).
func besselKSeries(nu, x float64) (m dd.DD, e int, ok bool) {
	if nu == 0 || nu == 1 {
		return dd.DD{}, 0, false
	}
	L, gs := besselLogPowGamma(nu, x) // log|(x/2)^nu/Gamma(nu+1)|
	c := math.Pi / (2 * sinPi(nu)) * gs
	// |second/first| = |c| 2 nu exp(2L)
	if math.Log(math.Abs(c)*2*nu)+2*L.Hi > math.Log(0.25) {
		return dd.DD{}, 0, false
	}
	q := x * x / 4
	ta, sa := 1.0, 0.0
	tb, sb := 1.0, 0.0
	for k := 1; k < 200; k++ {
		fk := float64(k)
		ta *= q / (fk * (fk - nu))
		tb *= q / (fk * (fk + nu))
		sa += ta
		sb += tb
		if math.Abs(ta) <= 0x1p-60 && math.Abs(tb) <= 0x1p-60 {
			break
		}
	}
	// first = exp(-L) (1+sa) / (2 nu) = 2^k exp(r) (1+sa)/(2 nu)
	nL := dd.Neg(L)
	k := math.Floor(nL.Hi / dd.Ln2Hi)
	k2 := dd.Add(dd.MulFF(k, dd.Ln2Hi), dd.MulFF(k, dd.Ln2Lo))
	first := dd.DivF(dd.MulF(dd.AddF(dd.F(1), sa), dd.Exp(dd.Sub(nL, k2))), 2*nu)
	// second / 2^k = c exp(L - k ln2) (1+sb)
	second := c * besselExpMul(dd.Sub(L, k2), 1+sb)
	return dd.AddF(first, -second), int(k), true
}

// besselTemmeIK returns K_u(x) and K_{u+1}(x) for |u| <= 1/2, 0 < x <= 2
// (Temme 1975; Boost temme_ik).
func besselTemmeIK(u, x float64) (ku, ku1 float64) {
	g1, g2 := besselRGammaPair(u)
	rgp := g2 - u*g1 // 1/Gamma(1+u)
	rgm := g2 + u*g1 // 1/Gamma(1-u)
	a := besselLogHalf(x)
	b := besselPowHalf(x, u)
	sigma := -a * u
	c := 1.0
	if math.Abs(u) >= 0x1p-52 {
		c = sinPi(u) / (u * math.Pi)
	}
	d := 1.0
	if math.Abs(sigma) >= 0x1p-52 {
		d = math.Sinh(sigma) / sigma
	}
	// Boost's gamma1 = g1, gamma2 = g2 (identically).
	p := 1 / rgp / (2 * b)
	q := 1 / rgm * b / 2
	f := (math.Cosh(sigma)*g1 + d*(-a)*g2) / c
	h := p
	coef := 1.0
	sum := coef * f
	sum1 := coef * h
	for k := 1; k < 1000; k++ {
		fk := float64(k)
		f = (fk*f + p + q) / (fk*fk - u*u)
		p /= fk - u
		q /= fk + u
		h = p - fk*f
		coef *= x * x / (4 * fk)
		sum += coef * f
		sum1 += coef * h
		if math.Abs(coef*f) < math.Abs(sum)*0x1p-53 {
			break
		}
	}
	return sum, 2 * sum1 / x
}

// besselTemmeIKMaxX is where Steed's CF2 (in double-double) takes over
// from Temme's series for K (Boost switches at 2; Temme's float64 sums
// cost several ulps, CF2 converges well down to about this x).
const besselTemmeIKMaxX = 0.5

// besselInv2PiDD is 1/(2 pi) as a double-double.
var besselInv2PiDD = dd.DD{Hi: 0.15915494309189535, Lo: -9.839338337591243e-18}

// besselCF1IK evaluates I_{v+1}/I_v by its continued fraction (A&S 9.1.73
// adapted) with the modified Lentz method.
func besselCF1IK(v, x float64) dd.DD {
	C, D := 0x1p-511, 0.0
	k := 1
	for ; k < besselMaxIter; k++ {
		b := 2 * (v + float64(k)) / x
		C = b + 1/C
		D = 1 / (b + D)
		if math.Abs(C*D-1) <= 0x1p-52 {
			break
		}
	}
	return besselCF1Tail(v, x, 2*k+20, 1)
}

// besselCF2IK returns K_u(x) e^x and K_{u+1}(x) e^x for x > 1 by Steed's
// algorithm for the continued fraction of U(u+1/2, 2u+1, 2x) (Thompson and
// Barnett 1986; Boost CF2_ik).
func besselCF2IK(u, x float64) (ku, ku1 float64) {
	// The recurrences in double-double: the float64 version of this sum
	// costs several ulps (the sum S of Steed's algorithm accumulates one
	// rounding per term). The terms are summed until they fall below 2^-64
	// of S (the result needs no more), and the divisions are reciprocals
	// with their FMA remainder (besselRecipDD) or divisions by a float64.
	a := besselAddFS(dd.MulFF(u, u), -0.25)
	b := dd.F(2 * (x + 1))
	D := besselRecipDD(b)
	f, delta := D, D
	prev, cur := dd.F(0), dd.F(1)
	C := dd.Neg(a)
	Q := C
	S := besselAddS(dd.F(1), dd.Mul(Q, delta))
	for k := 2; k < besselMaxIter; k++ {
		a = besselAddFS(a, -2*float64(k-1))
		b = besselAddFS(b, 2)
		D = besselRecipDD(besselAddS(b, dd.Mul(a, D)))
		delta = dd.Mul(delta, besselAddFS(dd.Mul(b, D), -1))
		f = besselAddS(f, delta)
		q := dd.Mul(besselAddS(prev, dd.Neg(dd.Mul(besselAddFS(b, -2), cur))), besselRecipDD(a))
		prev, cur = cur, q
		C = igamDivF(dd.Mul(C, dd.Neg(a)), float64(k))
		Q = besselAddS(Q, dd.Mul(C, q))
		S = besselAddS(S, dd.Mul(Q, delta))
		if q.Hi < 0x1p-52 { // rescale (Boost): C, prev, cur by q
			C = dd.Mul(C, q)
			r := besselRecipDD(q)
			prev = dd.Mul(prev, r)
			cur = dd.Mul(cur, r)
		}
		if math.Abs(Q.Hi*delta.Hi) < math.Abs(S.Hi)*0x1p-64 {
			break
		}
	}
	k := dd.Mul(dd.Sqrt(igamDivF(besselPiOver2DD, x)), besselRecipDD(S))
	t := besselAddFS(besselAddS(dd.Mul(a0(u), f), dd.F(0.5+u)), x)
	return k.Float(), igamDivF(dd.Mul(k, t), x).Float()
}

// besselAddS returns a + b for double-doubles with the low parts added in
// float64 (Dekker's "sloppy" addition, half the operations of dd.Add): its
// error is a rounding of a.Lo + b.Lo, which is 2^-105 of the result unless
// a.Hi and b.Hi cancel, and the recurrences that use it do not cancel by
// more than a few bits.
func besselAddS(a, b dd.DD) dd.DD {
	s, e := dd.TwoSum(a.Hi, b.Hi)
	h, l := dd.FastTwoSum(s, e+(a.Lo+b.Lo))
	return dd.DD{Hi: h, Lo: l}
}

// besselAddFS returns a + b for a double-double a and a float64 b.
func besselAddFS(a dd.DD, b float64) dd.DD {
	s, e := dd.TwoSum(a.Hi, b)
	h, l := dd.FastTwoSum(s, e+a.Lo)
	return dd.DD{Hi: h, Lo: l}
}

// besselRecipDD returns 1/w for a double-double w to about 2^-104: the
// float64 reciprocal corrected by its FMA remainder.
func besselRecipDD(w dd.DD) dd.DD {
	r := 1 / w.Hi
	h, l := dd.FastTwoSum(r, (math.FMA(-r, w.Hi, 1)-r*w.Lo)*r)
	return dd.DD{Hi: h, Lo: l}
}

// a0 returns u^2 - 1/4 as a double-double.
func a0(u float64) dd.DD { return dd.AddF(dd.MulFF(u, u), -0.25) }

// besselAsymIK returns I_v(x) e^-x (which = 'I') or K_v(x) e^x by the
// large-x expansions DLMF 10.40.1-2, with ok false when the terms do not
// fall below besselAsymTol before growing. For I the exponentially small
// companion term (of relative size e^-2x) is neglected, so this is used
// only for x > 20.
func besselAsymIK(v, x float64, which byte) (r float64, ok bool) {
	mu := 4 * v * v
	sgn := 1.0
	if which == 'I' {
		sgn = -1
	}
	if math.Abs(mu-1) > 4*x { // first term above 1/2: the sum would cancel
		return 0, false
	}
	// the terms after the first are summed apart and added to 1 once: one
	// rounding instead of one per term
	t, s := 1.0, 0.0
	prev := math.Inf(1)
	for k := 1; k < 200; k++ {
		o := float64(2*k - 1)
		t *= sgn * (mu - o*o) / (float64(k) * 8 * x)
		s += t
		at := math.Abs(t)
		if at < besselAsymTol*0x1p-6 {
			// 1/sqrt(2 pi x) and sqrt(pi/(2x)) in double-double
			c := besselInv2PiDD
			if which == 'K' {
				c = besselPiOver2DD
			}
			xs, sc := x, 1.0
			if x > 0x1p1000 { // keep c/x normal
				xs, sc = x*0x1p-200, 0x1p-100
			}
			return dd.Mul(dd.Sqrt(dd.DivF(c, xs)), dd.AddF(dd.F(1), s)).Float() * sc, true
		}
		if at > prev {
			break
		}
		prev = at
	}
	return 0, false
}

// besselIKCore returns I_v(x) e^-x = ie * 2^ei and K_v(x) e^x = ke * 2^ek
// for v >= 0 and finite x > 0 (what is not needed may be NaN): the large-x
// expansions where they converge; otherwise K by Temme's series (x <= 2) or
// Steed's CF2 at the fractional order u and forward recurrence in
// double-double, and I by its power series (small x against v) or by CF1
// and the Wronskian I_v K_{v+1} + I_{v+1} K_v = 1/x (Boost bessel_ik).
// The powers of two keep every intermediate in range.
func besselIKCore(v, x float64, needI, needK bool) (ie float64, ei int, ke float64, ek int) {
	ie, ke = math.NaN(), math.NaN()
	if x > 20 {
		okI, okK := !needI, !needK
		if needI {
			ie, okI = besselAsymIK(v, x, 'I')
		}
		if needK {
			ke, okK = besselAsymIK(v, x, 'K')
		}
		if okI && okK {
			return ie, 0, ke, 0
		}
		ie, ke = math.NaN(), math.NaN()
	}
	iSeries := v > 0 && (x/v < 0.25 || besselSeriesOK(v, x, 1))
	if !needK && iSeries {
		ie, ei = besselSeries(v, x, 1, true)
		return ie, ei, ke, 0
	}
	if v > besselMaxOrder {
		// Only the series is run this far out (K is then beyond overflow);
		// elsewhere the result is NaN.
		if !besselSeriesOK(v, x, 1) {
			return math.NaN(), 0, math.NaN(), 0
		}
		if needI {
			ie, ei = besselSeries(v, x, 1, true)
		}
		return ie, ei, math.Inf(1), 0
	}
	n := int(math.Floor(v + 0.5))
	u := v - float64(n)
	// K_u e^x and K_{u+1} e^x, as multiples of 2^e0
	var Ku, Ku1 dd.DD
	e0 := 0
	if a, ea, ok := besselKSeries(math.Abs(u), x); ok && x <= besselTemmeIKMaxX {
		b, eb, _ := besselKSeries(1+u, x)
		ex := math.Exp(x)
		Ku, Ku1, e0 = dd.MulF(a, ex), dd.MulF(besselScaleDD(b, eb-ea), ex), ea
	} else {
		var ku, ku1 float64
		if x <= besselTemmeIKMaxX {
			ku, ku1 = besselTemmeIK(u, x)
			ex := math.Exp(x)
			ku *= ex
			ku1 *= ex
		} else {
			ku, ku1 = besselCF2IK(u, x)
		}
		Ku, Ku1 = dd.F(ku), dd.F(ku1)
	}
	// K_{k+1} = 2k/x K_k + K_{k-1}, stable upwards (K is dominant)
	kv, kv1, e := besselRecur(u, x, n, Ku, Ku1, 1)
	e += e0
	if needK {
		ke, ek = kv.Float(), e
	}
	if needI {
		if iSeries {
			ie, ei = besselSeries(v, x, 1, true)
		} else {
			fv := besselCF1IK(v, x)
			den := dd.Add(dd.Mul(kv, fv), kv1)
			ie, ei = dd.Div(dd.F(1), dd.MulF(den, x)).Float(), -e
		}
	}
	return ie, ei, ke, ek
}

// besselIK returns I_v(x) and K_v(x), unscaled or scaled, for real v of
// either sign and finite x > 0. For v < 0 (non-integer), I_{-nu} =
// I_nu + (2/pi) sin(nu pi) K_nu (DLMF 10.27.2), or the power series of
// I_{-nu} directly where it converges without cancellation.
func besselIK(v, x float64, needI, needK, scaled bool) (I, K float64) {
	nu := math.Abs(v)
	refl := v < 0 && needI && v != math.Trunc(v)
	if refl && !needK && besselSeriesOK(v, x, 0.25) {
		m, e := besselSeries(v, x, 1, scaled)
		return math.Ldexp(m, e), math.NaN()
	}
	if !needI && nu > 0 && x < 1 && LogGamma(nu)-nu*besselLogHalf(x)-math.Ln2 > 710 {
		// K_nu(x) ~ Gamma(nu) (2/x)^nu / 2 (DLMF 10.30.2) overflows.
		return math.NaN(), math.Inf(1)
	}
	ie, ei, ke, ek := besselIKCore(nu, x, needI, needK || refl)
	ex := x
	if scaled {
		ex = 0
	}
	K = besselLdexpExp(ke, ek, -ex)
	I = besselLdexpExp(ie, ei, ex)
	if refl {
		t := 2 / math.Pi * sinPi(nu) * ke
		I += besselLdexpExp(t, ek, -2*x+ex)
	}
	return I, K
}

// I0 returns the modified Bessel function of the first kind of order zero,
// I0(x) = sum (x^2/4)^k / (k!)^2, for real x (even). I0(0) = 1,
// I0(±Inf) = +Inf, I0(NaN) = NaN; overflows to +Inf for |x| > ~713.99.
//
// Algorithm: the power series below 1/8; up to 512, e^x times I0(x) e^-x
// from degree-11 interpolants on 8 pieces per octave of x
// (zz_besselI0eOct_coeffs.go, fitted to mpmath), with e^x from dd.ExpMant
// and the product rounded once; DLMF 10.40.1 beyond.
func I0(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case x < 0.125:
		return besselI0Tiny(x)
	}
	if i := octIndex(x, besselI0eOctMin, besselI0eOctMax); i >= 0 {
		q := &besselI0eOct[i]
		return octExpMul(q, x-q.c, x)
	}
	return besselLdexpExp(besselI0eLarge(x), 0, x)
}

// The octaves of besselI0eOct (zz_besselI0eOct_coeffs.go): x in [1/8, 512).
const besselI0eOctMin, besselI0eOctMax = -3, 8

// besselI0Tiny is I0(x) = sum q^k / (k!)^2, q = x^2/4 <= 1/256, for
// x < 1/8: the terms to k = 5 (the next is below 2^-66 of the sum), all
// positive, added to 1 once.
func besselI0Tiny(x float64) float64 { return 1 + besselI0TinyTail(x) }

// besselI0TinyTail is besselI0Tiny(x) - 1.
func besselI0TinyTail(x float64) float64 {
	q := x * x / 4
	return q * (1 + q*(1.0/4+q*(1.0/36+q*(1.0/576+q*(1.0/14400)))))
}

// besselI0eLarge is I0(x) e^-x for x >= 512 by the asymptotic series DLMF
// 10.40.1 (summed to below 2^-64, prefactor in double-double), which
// converges there for order 0.
func besselI0eLarge(x float64) float64 {
	r, _ := besselAsymIK(0, x, 'I')
	return r
}

// I0e returns I0(x) e^-|x|, defined for all real x; I0e(±Inf) = 0.
func I0e(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1):
		return 0
	case x < 0.125:
		// e^-x (1 + T): one rounding of m (1 + T), m = e^-x from dd.ExpMant
		m, k := dd.ExpMant(dd.DD{Hi: -x})
		return math.Ldexp(m.Hi+(m.Lo+m.Hi*besselI0TinyTail(x)), k)
	}
	if i := octIndex(x, besselI0eOctMin, besselI0eOctMax); i >= 0 {
		q := &besselI0eOct[i]
		return octEval(q, x-q.c)
	}
	return besselI0eLarge(x)
}

// I1 returns the modified Bessel function of the first kind of order one,
// I1(x) = (x/2) sum (x^2/4)^k / (k! (k+1)!), for real x (odd). I1(0) = 0,
// I1(±Inf) = ±Inf, I1(NaN) = NaN; overflows for |x| > ~713.98.
//
// Algorithm: Boost.Math bessel_i1.hpp double-precision minimax
// approximations.
func I1(x float64) float64 {
	sign := 1.0
	if x < 0 {
		x, sign = -x, -1
	}
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return sign * x
	case x < 7.75:
		return sign * besselI1Small(x)
	}
	return sign * besselLdexpExp(besselI1eLarge(x), 0, x)
}

// besselI1eLarge is I1(x) e^-x for x >= 7.75 (as besselI0eLarge).
func besselI1eLarge(x float64) float64 {
	if x > 20 {
		if r, ok := besselAsymIK(1, x, 'I'); ok {
			return r
		}
	}
	return besselPoly(besselI1Pm[:], 1/x) / math.Sqrt(x)
}

// I1e returns I1(x) e^-|x| (odd in x); I1e(±Inf) = ±0.
func I1e(x float64) float64 {
	sign := 1.0
	if x < 0 {
		x, sign = -x, -1
	}
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1):
		return sign * 0
	case x < 7.75:
		return sign * besselLdexpExp(besselI1Small(x), 0, -x)
	}
	return sign * besselI1eLarge(x)
}

// Boost.Math bessel_i1.hpp (53-bit branch) coefficients.
var (
	besselI1Pm = [...]float64{3.989422804014406054e-01, -1.496033551613111533e-01, -4.675104253598537322e-02, -4.090895951581637791e-02, -5.719036414430205390e-02, -1.528189554374492735e-01, 3.458284470977172076e+00, -2.426181371595021021e+02, 1.178785865993440669e+04, -4.404655582443487334e+05, 1.277677779341446497e+07, -2.903390398236656519e+08, 5.192386898222206474e+09, -7.313784438967834057e+10, 8.087824484994859552e+11, -6.967602516005787001e+12, 4.614040809616582764e+13, -2.298849639457172489e+14, 8.325554073334618015e+14, -2.067285045778906105e+15, 3.146401654361325073e+15, -2.213318202179221945e+15}
)

func besselI1Small(x float64) float64 {
	// (x/2) sum (x^2/4)^k / (k! (k+1)!) in double-double
	q := dd.MulF(dd.MulFF(x, x), 0.25)
	t, s := dd.F(1), dd.F(1)
	for k := 1; k < 60; k++ {
		t = dd.DivF(dd.Mul(t, q), float64(k*(k+1)))
		s = dd.Add(s, t)
		if t.Hi < 0x1p-60*s.Hi {
			break
		}
	}
	return dd.MulF(s, x/2).Float()
}

// Boost.Math bessel_k0.hpp (53-bit branch) coefficients.
var (
	besselK0P1 = [...]float64{-1.372509002685546267e-01, 2.574916117833312855e-01, 1.395474602146869316e-02, 5.445476986653926759e-04, 7.125159422136622118e-06}
	besselK0Q1 = [...]float64{1.000000000000000000e+00, -5.458333438017788530e-02, 1.291052816975251298e-03, -1.367653946978586591e-05}
	besselK0P2 = [...]float64{1.159315156584124484e-01, 2.789828789146031732e-01, 2.524892993216121934e-02, 8.460350907213637784e-04, 1.491471924309617534e-05, 1.627106892422088488e-07, 1.208266102392756055e-09, 6.611686391749704310e-12}
)

// besselK0Small is K0 on (0, 1].
func besselK0Small(x float64) float64 {
	const Y = 1.137250900268554688
	a := x * x / 4
	a = (besselRational(besselK0P1[:], besselK0Q1[:], a)+Y)*a + 1
	return besselPoly(besselK0P2[:], x*x) - besselLog(x)*a
}

// The octaves of besselK0eOct (zz_besselK0eOct_coeffs.go): x in [1, 512).
const besselK0eOctMin, besselK0eOctMax = 0, 8

// besselK0e is K0(x) e^x for x > 1: the octave interpolants below 512, the
// asymptotic series DLMF 10.40.2 (which converges there for order 0)
// beyond.
func besselK0e(x float64) float64 {
	if i := octIndex(x, besselK0eOctMin, besselK0eOctMax); i >= 0 {
		q := &besselK0eOct[i]
		return octEval(q, x-q.c)
	}
	r, _ := besselAsymIK(0, x, 'K')
	return r
}

// K0 returns the modified Bessel function of the second kind of order
// zero, K0(x) = integral_0^Inf cos(x sinh t) dt, for x > 0. K0(0) = +Inf,
// K0(+Inf) = 0, K0(x < 0) = NaN, K0(NaN) = NaN; underflows to 0 beyond
// x ~ 745.
//
// Algorithm: Boost.Math bessel_k0.hpp's log-term form on (0, 1]; up to
// 512, e^-x times K0(x) e^x from degree-11 interpolants on 8 pieces per
// octave (fitted to mpmath), the product rounded once; DLMF 10.40.2
// beyond.
func K0(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	case x <= 1:
		return besselK0Small(x)
	}
	if i := octIndex(x, besselK0eOctMin, besselK0eOctMax); i >= 0 {
		q := &besselK0eOct[i]
		return octExpMul(q, x-q.c, -x)
	}
	return besselLdexpExp(besselK0e(x), 0, -x)
}

// K0e returns K0(x) e^x for x > 0 (NaN for x < 0, +Inf at 0, 0 at +Inf).
func K0e(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	case x <= 1:
		return besselLdexpExp(besselK0Small(x), 0, x)
	}
	return besselK0e(x)
}

// Boost.Math bessel_k1.hpp (53-bit branch) coefficients.
var (
	besselK1P1 = [...]float64{-3.62137953440350228e-03, 7.11842087490330300e-03, 1.00302560256614306e-05, 1.77231085381040811e-06}
	besselK1Q1 = [...]float64{1.00000000000000000e+00, -4.80414794429043831e-02, 9.85972641934416525e-04, -8.91196859397070326e-06}
	besselK1P2 = [...]float64{-3.07965757829206184e-01, -7.80929703673074907e-02, -2.70619343754051620e-03, -2.49549522229072008e-05}
	besselK1Q2 = [...]float64{1.00000000000000000e+00, -2.36316836412163098e-02, 2.64524577525962719e-04, -1.49749618004162787e-06}
	besselK1P3 = [...]float64{-1.97028041029226295e-01, -2.32408961548087617e+00, -7.98269784507699938e+00, -2.39968410774221632e+00, 3.28314043780858713e+01, 5.67713761158496058e+01, 3.30907788466509823e+01, 6.62582288933739787e+00, 3.08851840645286691e-01}
	besselK1Q3 = [...]float64{1.00000000000000000e+00, 1.41811409298826118e+01, 7.35979466317556420e+01, 1.77821793937080859e+02, 2.11014501598705982e+02, 1.19425262951064454e+02, 2.88448064302447607e+01, 2.27912927104139732e+00, 2.50358186953478678e-02}
)

// besselK1Small is K1 on (0, 1].
func besselK1Small(x float64) float64 {
	const Y = 8.69547128677368164e-02
	a := x * x / 4
	a = ((besselRational(besselK1P1[:], besselK1Q1[:], a)+Y)*a*a + a/2 + 1) * x / 2
	return besselRational(besselK1P2[:], besselK1Q2[:], x*x)*x + 1/x + besselLog(x)*a
}

// besselK1Large is K1(x) e^x sqrt(x) for x > 1.
func besselK1Large(x float64) float64 {
	return besselRational(besselK1P3[:], besselK1Q3[:], 1/x) + 1.45034217834472656
}

// besselK1e is K1(x) e^x for x > 1 (as besselK0e).
func besselK1e(x float64) float64 {
	if x > 20 {
		if r, ok := besselAsymIK(1, x, 'K'); ok {
			return r
		}
	}
	return besselK1Large(x) / math.Sqrt(x)
}

// K1 returns the modified Bessel function of the second kind of order one
// for x > 0. K1(0) = +Inf, K1(+Inf) = 0, K1(x < 0) = NaN, K1(NaN) = NaN;
// overflows to +Inf for x below about 5.6e-309 and underflows beyond ~745.
//
// Algorithm: Boost.Math bessel_k1.hpp double-precision minimax
// approximations.
func K1(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	case x <= 1:
		return besselK1Small(x)
	}
	return besselLdexpExp(besselK1e(x), 0, -x)
}

// K1e returns K1(x) e^x for x > 0 (NaN for x < 0, +Inf at 0, 0 at +Inf).
func K1e(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case math.IsInf(x, 1):
		return 0
	case x <= 1:
		return besselLdexpExp(besselK1Small(x), 0, x)
	}
	return besselK1e(x)
}

// besselIvArgs handles the special values shared by Iv and Ive; done
// reports that r is the answer.
func besselIvArgs(v, x float64, scaled bool) (r float64, done bool) {
	switch {
	case math.IsNaN(v) || math.IsNaN(x):
		return math.NaN(), true
	case math.IsInf(v, 1):
		if math.IsInf(x, 0) {
			return math.NaN(), true
		}
		return 0, true
	case math.IsInf(v, -1):
		return math.NaN(), true
	case x < 0 && v != math.Trunc(v):
		return math.NaN(), true
	case math.IsInf(x, 0):
		odd := x < 0 && math.Mod(v, 2) != 0
		r = math.Inf(1)
		if scaled {
			r = 0
		}
		if odd {
			r = -r
		}
		return r, true
	case x == 0:
		switch {
		case v == 0:
			return 1, true
		case v > 0 || v == math.Trunc(v):
			return 0, true
		}
		// I_v(x) ~ (x/2)^v / Gamma(v+1)
		return math.Copysign(math.Inf(1), -sinPi(v)), true
	}
	return 0, false
}

// Iv returns the modified Bessel function of the first kind of real order
// v, I_v(x) = sum_k (x/2)^(2k+v) / (k! Gamma(k+v+1)) (DLMF 10.25.2). For
// non-integer v it is real only for x >= 0 (NaN for x < 0); for integer v,
// I_v(-x) = (-1)^v I_v(x) and I_{-n} = I_n. Iv(v, 0) is 1 for v = 0, 0 for
// v > 0 or integer v, and ±Inf (the sign of 1/Gamma(v+1)) for non-integer
// v < 0; Iv(v, +Inf) = +Inf; Iv(+Inf, x) = 0, Iv(-Inf, x) = NaN; overflow
// gives +Inf.
//
// Algorithm: Boost.Math bessel_ik.hpp (Temme's series and Steed's CF2 for
// K, CF1 and the Wronskian for I, power series for small x), carried
// exponentially scaled, with the large-x expansion DLMF 10.40.1 and the
// reflection I_{-v} = I_v + (2/pi) sin(v pi) K_v (DLMF 10.27.2). I0 and I1
// serve the orders 0 and 1.
func Iv(v, x float64) float64 {
	if r, done := besselIvArgs(v, x, false); done {
		return r
	}
	sign := 1.0
	if x < 0 {
		x = -x
		if math.Mod(v, 2) != 0 {
			sign = -1
		}
	}
	if v == math.Trunc(v) {
		v = math.Abs(v)
		switch v {
		case 0:
			return I0(x)
		case 1:
			return sign * I1(x)
		}
	}
	i, _ := besselIK(v, x, true, false, false)
	return sign * i
}

// Ive returns Iv(v, x) e^-|x|, with the conventions of Iv; Ive(v, ±Inf)
// is ±0 (the limit), never NaN, and the result does not overflow.
//
// Algorithm: as Iv, computed scaled throughout.
func Ive(v, x float64) float64 {
	if r, done := besselIvArgs(v, x, true); done {
		return r
	}
	sign := 1.0
	if x < 0 {
		x = -x
		if math.Mod(v, 2) != 0 {
			sign = -1
		}
	}
	if v == math.Trunc(v) {
		v = math.Abs(v)
		switch v {
		case 0:
			return I0e(x)
		case 1:
			return sign * I1e(x)
		}
	}
	i, _ := besselIK(v, x, true, false, true)
	return sign * i
}

// In returns the modified Bessel function of the first kind of integer
// order n, Iv(float64(n), x).
func In(n int, x float64) float64 { return Iv(float64(n), x) }

// besselKvArgs handles the special values shared by Kv and Kve.
func besselKvArgs(v, x float64) (r float64, done bool) {
	switch {
	case math.IsNaN(v) || math.IsNaN(x) || x < 0:
		return math.NaN(), true
	case math.IsInf(x, 1):
		if math.IsInf(v, 0) {
			return math.NaN(), true
		}
		return 0, true
	case math.IsInf(v, 0) || x == 0:
		return math.Inf(1), true
	}
	return 0, false
}

// Kv returns the modified Bessel function of the second kind of real order
// v, K_v(x) = (pi/2) (I_{-v}(x) - I_v(x)) / sin(v pi) (DLMF 10.27.4), even
// in v, defined for x > 0. Kv(v, 0) = +Inf, Kv(v, +Inf) = 0, Kv(v, x < 0) =
// NaN, Kv(±Inf, x) = +Inf; NaN in, NaN out. Overflow gives +Inf and
// underflow 0.
//
// Algorithm: K0/K1 for v = 0, 1; otherwise Temme's series (x <= 2) or
// Steed's CF2 (in double-double) for the fractional order and forward
// recurrence in double-double (Boost.Math bessel_ik.hpp), or the large-x
// expansion DLMF 10.40.2 where it converges.
func Kv(v, x float64) float64 {
	if r, done := besselKvArgs(v, x); done {
		return r
	}
	v = math.Abs(v)
	switch {
	case v == 0:
		return K0(x)
	case v == 1:
		return K1(x)
	}
	_, k := besselIK(v, x, false, true, false)
	return k
}

// Kve returns Kv(v, x) e^x, with the conventions of Kv (Kve(v, +Inf) = 0).
//
// Algorithm: as Kv, computed scaled throughout.
func Kve(v, x float64) float64 {
	if r, done := besselKvArgs(v, x); done {
		return r
	}
	v = math.Abs(v)
	switch {
	case v == 0:
		return K0e(x)
	case v == 1:
		return K1e(x)
	}
	_, k := besselIK(v, x, false, true, true)
	return k
}

// Kn returns the modified Bessel function of the second kind of integer
// order n, Kv(float64(n), x).
func Kn(n int, x float64) float64 { return Kv(float64(n), x) }
