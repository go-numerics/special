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

// besselTemmeIK returns K_u(x) and K_{u+1}(x) for |u| <= 1/2, 0 < x <= 2
// (Temme 1975; Boost temme_ik).
func besselTemmeIK(u, x float64) (ku, ku1 float64) {
	g1, g2 := besselRGammaPair(u)
	rgp := g2 - u*g1 // 1/Gamma(1+u)
	rgm := g2 + u*g1 // 1/Gamma(1-u)
	a := math.Log(x / 2)
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

// besselInv2PiDD is 1/(2 pi) as a double-double.
var besselInv2PiDD = dd.DD{Hi: 0.15915494309189535, Lo: -9.839338337591243e-18}

// besselCF1IK evaluates I_{v+1}/I_v by its continued fraction (A&S 9.1.73
// adapted) with the modified Lentz method.
func besselCF1IK(v, x float64) dd.DD {
	const tiny = 0x1p-511
	C, f := tiny, tiny
	D := 0.0
	for k := 1; k < besselMaxIter; k++ {
		b := 2 * (v + float64(k)) / x
		C = b + 1/C
		D = b + D
		if D == 0 {
			D = tiny
		}
		D = 1 / D
		delta := C * D
		f *= delta
		if math.Abs(delta-1) <= 0x1p-52 {
			return besselCF1Tail(v, x, 2*k+20, 1)
		}
	}
	return dd.F(f)
}

// besselCF2IK returns K_u(x) e^x and K_{u+1}(x) e^x for x > 1 by Steed's
// algorithm for the continued fraction of U(u+1/2, 2u+1, 2x) (Thompson and
// Barnett 1986; Boost CF2_ik).
func besselCF2IK(u, x float64) (ku, ku1 float64) {
	a := u*u - 0.25
	b := 2 * (x + 1)
	D := 1 / b
	f, delta := D, D
	prev, cur := 0.0, 1.0
	C := -a
	Q := C
	S := 1 + Q*delta
	for k := 2; k < besselMaxIter; k++ {
		a -= 2 * float64(k-1)
		b += 2
		D = 1 / (b + a*D)
		delta *= b*D - 1
		f += delta
		q := (prev - (b-2)*cur) / a
		prev, cur = cur, q
		C *= -a / float64(k)
		Q += C * q
		S += Q * delta
		if q < 0x1p-52 {
			C *= q
			prev /= q
			cur /= q
		}
		if math.Abs(Q*delta) < math.Abs(S)*0x1p-53 {
			break
		}
	}
	ku = math.Sqrt(math.Pi/(2*x)) / S
	ku1 = ku * (0.5 + u + x + (u*u-0.25)*f) / x
	return ku, ku1
}

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
		if needI {
			ie, ei = besselSeries(v, x, 1, true)
		}
		return ie, ei, math.Inf(1), 0
	}
	n := int(math.Floor(v + 0.5))
	u := v - float64(n)
	var ku, ku1 float64
	if x <= 2 {
		ku, ku1 = besselTemmeIK(u, x)
		ex := math.Exp(x)
		ku *= ex
		ku1 *= ex
	} else {
		ku, ku1 = besselCF2IK(u, x)
	}
	// K_{k+1} = 2k/x K_k + K_{k-1}, stable upwards (K is dominant)
	kv, kv1, e := besselRecur(u, x, n, dd.F(ku), dd.F(ku1), 1)
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
// Algorithm: Boost.Math bessel_i0.hpp double-precision minimax
// approximations (polynomial in x^2/4 below 7.75, in 1/x above).
func I0(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x) || math.IsInf(x, 1):
		return x
	case x < 7.75:
		return besselI0Small(x)
	}
	return besselLdexpExp(besselI0eLarge(x), 0, x)
}

// besselI0eLarge is I0(x) e^-x for x >= 7.75: Boost's minimax polynomial in
// 1/x up to 20, the asymptotic series DLMF 10.40.1 (summed to below 2^-64,
// prefactor in double-double) beyond.
func besselI0eLarge(x float64) float64 {
	if x > 20 {
		if r, ok := besselAsymIK(0, x, 'I'); ok {
			return r
		}
	}
	return besselPoly(besselI0Pm[:], 1/x) / math.Sqrt(x)
}

// I0e returns I0(x) e^-|x|, defined for all real x; I0e(±Inf) = 0.
func I0e(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1):
		return 0
	case x < 7.75:
		return besselLdexpExp(besselI0Small(x), 0, -x)
	}
	return besselI0eLarge(x)
}

// Boost.Math bessel_i0.hpp (53-bit branch) coefficients.
var (
	besselI0Pm = [...]float64{3.98942280401425088e-01, 4.98677850604961985e-02, 2.80506233928312623e-02, 2.92211225166047873e-02, 4.44207299493659561e-02, 1.30970574605856719e-01, -3.35052280231727022e+00, 2.33025711583514727e+02, -1.13366350697172355e+04, 4.24057674317867331e+05, -1.23157028595698731e+07, 2.80231938155267516e+08, -5.01883999713777929e+09, 7.08029243015109113e+10, -7.84261082124811106e+11, 6.76825737854096565e+12, -4.49034849696138065e+13, 2.24155239966958995e+14, -8.13426467865659318e+14, 2.02391097391687777e+15, -3.08675715295370878e+15, 2.17587543863819074e+15}
)

func besselI0Small(x float64) float64 {
	// sum (x^2/4)^k / (k!)^2 (DLMF 10.25.2) in double-double: all terms
	// are positive, so the sum is as accurate as its terms.
	q := dd.MulF(dd.MulFF(x, x), 0.25)
	t, s := dd.F(1), dd.F(1)
	for k := 1; k < 60; k++ {
		t = dd.DivF(dd.Mul(t, q), float64(k*k))
		s = dd.Add(s, t)
		if t.Hi < 0x1p-60*s.Hi {
			break
		}
	}
	return s.Float()
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
	besselK0P3 = [...]float64{2.533141373155002416e-01, 3.628342133984595192e+00, 1.868441889406606057e+01, 4.306243981063412784e+01, 4.424116209627428189e+01, 1.562095339356220468e+01, -1.810138978229410898e+00, -1.414237994269995877e+00, -9.369168119754924625e-02}
	besselK0Q3 = [...]float64{1.000000000000000000e+00, 1.494194694879908328e+01, 8.265296455388554217e+01, 2.162779506621866970e+02, 2.845145155184222157e+02, 1.851714491916334995e+02, 5.486540717439723515e+01, 6.118075837628957015e+00, 1.586261269326235053e-01}
)

// besselK0Small is K0 on (0, 1].
func besselK0Small(x float64) float64 {
	const Y = 1.137250900268554688
	a := x * x / 4
	a = (besselRational(besselK0P1[:], besselK0Q1[:], a)+Y)*a + 1
	return besselPoly(besselK0P2[:], x*x) - math.Log(x)*a
}

// besselK0Large is K0(x) e^x sqrt(x) for x > 1.
func besselK0Large(x float64) float64 {
	return besselRational(besselK0P3[:], besselK0Q3[:], 1/x) + 1
}

// besselK0e is K0(x) e^x for x > 1: Boost's rational approximation up to
// 20, the asymptotic series DLMF 10.40.2 beyond.
func besselK0e(x float64) float64 {
	if x > 20 {
		if r, ok := besselAsymIK(0, x, 'K'); ok {
			return r
		}
	}
	return besselK0Large(x) / math.Sqrt(x)
}

// K0 returns the modified Bessel function of the second kind of order
// zero, K0(x) = integral_0^Inf cos(x sinh t) dt, for x > 0. K0(0) = +Inf,
// K0(+Inf) = 0, K0(x < 0) = NaN, K0(NaN) = NaN; underflows to 0 beyond
// x ~ 745.
//
// Algorithm: Boost.Math bessel_k0.hpp double-precision minimax
// approximations (log-term form on (0, 1], rational in 1/x above).
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
	return besselRational(besselK1P2[:], besselK1Q2[:], x*x)*x + 1/x + math.Log(x)*a
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
// Algorithm: K0/K1 and forward recurrence for integer v (Boost.Math
// bessel_kn.hpp); otherwise Temme's series (x <= 2) or Steed's CF2 for the
// fractional order and forward recurrence (Boost.Math bessel_ik.hpp), or
// the large-x expansion DLMF 10.40.2 where it converges.
func Kv(v, x float64) float64 {
	if r, done := besselKvArgs(v, x); done {
		return r
	}
	v = math.Abs(v)
	if besselIsInt(v) && v <= 50 {
		return besselKn(int(v), x, false)
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
	if besselIsInt(v) && v <= 50 {
		return besselKn(int(v), x, true)
	}
	_, k := besselIK(v, x, false, true, true)
	return k
}

// Kn returns the modified Bessel function of the second kind of integer
// order n, Kv(float64(n), x).
func Kn(n int, x float64) float64 { return Kv(float64(n), x) }

// besselKn returns K_n(x) (or K_n(x) e^x) for 0 <= n <= 50 and finite x > 0
// by forward recurrence from K0 and K1 (Boost.Math bessel_kn.hpp), which is
// stable since K grows with the order.
func besselKn(n int, x float64, scaled bool) float64 {
	var prev, cur float64
	if scaled {
		prev, cur = K0e(x), K1e(x)
	} else {
		prev, cur = K0(x), K1(x)
	}
	if n == 0 {
		return prev
	}
	for k := 1; k < n; k++ {
		prev, cur = cur, 2*float64(k)/x*cur+prev
	}
	return cur
}
