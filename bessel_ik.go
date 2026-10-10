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
	b := math.Exp(u * a)
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

// besselCF1IK evaluates I_{v+1}/I_v by its continued fraction (A&S 9.1.73
// adapted) with the modified Lentz method.
func besselCF1IK(v, x float64) float64 {
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
			return besselCF1Tail(v, x, k+8, 1).Float()
		}
	}
	return f
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
	t, s := 1.0, 1.0
	prev := math.Inf(1)
	for k := 1; k < 200; k++ {
		o := float64(2*k - 1)
		t *= sgn * (mu - o*o) / (float64(k) * 8 * x)
		s += t
		at := math.Abs(t)
		if at < besselAsymTol*0x1p-6 {
			if which == 'I' {
				return s / math.Sqrt(2*math.Pi*x), true
			}
			return s * math.Sqrt(math.Pi/(2*x)), true
		}
		if at > prev {
			break
		}
		prev = at
	}
	return 0, false
}

// besselISeries returns I_v(x) for v >= 0 by its power series (DLMF
// 10.25.2), times e^-x when scaled.
func besselISeries(v, x float64, scaled bool) float64 {
	L := besselLogPowGamma(v, x)
	if scaled {
		L = dd.AddF(L, -x)
	}
	m := x * x / 4
	t, s := 1.0, 1.0
	for k := 1; k < 100000; k++ {
		fk := float64(k)
		t *= m / (fk * (fk + v))
		s += t
		if t <= 0x1p-54*s {
			break
		}
	}
	if L.Hi < -746 {
		return 0
	}
	return besselExpMul(L, s)
}

// besselIKScaled returns I_v(x) e^-x and K_v(x) e^x for v >= 0 and finite
// x > 0 (what is not needed may be NaN). Kv may be +Inf (overflow).
func besselIKScaled(v, x float64, needI, needK bool) (Ie, Ke float64) {
	Ie, Ke = math.NaN(), math.NaN()
	if x > 20 {
		okI, okK := !needI, !needK
		if needI {
			Ie, okI = besselAsymIK(v, x, 'I')
		}
		if needK {
			Ke, okK = besselAsymIK(v, x, 'K')
		}
		if okI && okK {
			return Ie, Ke
		}
	}
	if !needK && v > 0 && (x/v < 0.25 || x*x/4 <= v+1) {
		return besselISeries(v, x, true), Ke
	}
	if v > besselMaxOrder {
		if needI {
			Ie = besselISeries(v, x, true)
		}
		return Ie, math.Inf(1)
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
	// forward recurrence K_{k+1} = 2k/x K_k + K_{k-1} (scaled alike)
	scale := 1.0 // K_v e^x = prev / scale
	prev, cur := ku, ku1
	for k := 1; k <= n; k++ {
		fact := 2 * (u + float64(k)) / x
		if (math.MaxFloat64-math.Abs(prev))/fact < math.Abs(cur) {
			prev /= cur
			scale /= cur
			cur = 1
		}
		prev, cur = cur, fact*cur+prev
	}
	kv, kv1 := prev, cur
	if needK {
		if math.MaxFloat64*scale < kv {
			Ke = math.Inf(1)
		} else {
			Ke = kv / scale
		}
	}
	if needI {
		if v > 0 && (x/v < 0.25 || x*x/4 <= v+1) {
			Ie = besselISeries(v, x, true)
		} else {
			fv := besselCF1IK(v, x)
			Ie = scale / x / (kv*fv + kv1) // Wronskian I K' - I' K = -1/x
		}
	}
	return Ie, Ke
}

// besselIK returns I_v(x) and K_v(x), unscaled or scaled, for real v of
// either sign and finite x > 0.
func besselIK(v, x float64, needI, needK, scaled bool) (I, K float64) {
	nu := math.Abs(v)
	refl := v < 0 && needI && v != math.Trunc(v)
	ie, ke := besselIKScaled(nu, x, needI, needK || refl)
	if refl {
		// I_{-nu} = I_nu + (2/pi) sin(nu pi) K_nu (DLMF 10.27.2)
		t := 2 / math.Pi * sinPi(nu) * ke
		if scaled {
			// K e^x * e^-2x
			ie += besselExpScale(t, -2*x)
		} else {
			I = besselExpScale(ie, x) + besselExpScale(t, -x)
			ie = math.NaN()
		}
	}
	if scaled {
		return ie, ke
	}
	if !refl {
		I = besselExpScale(ie, x)
	}
	K = besselExpScale(ke, -x)
	return I, K
}

// besselExpScale returns r * e^t without intermediate overflow.
func besselExpScale(r, t float64) float64 {
	if math.IsNaN(r) || r == 0 || math.IsInf(r, 0) {
		return r
	}
	if t > 700 || t < -700 {
		h := math.Exp(t / 2)
		return r * h * h
	}
	return r * math.Exp(t)
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
	case math.IsNaN(x):
		return x
	case x < 7.75:
		return besselI0Small(x)
	case x < 500:
		return math.Exp(x) * besselPoly(besselI0Pm[:], 1/x) / math.Sqrt(x)
	}
	ex := math.Exp(x / 2)
	return ex * besselPoly(besselI0Pl[:], 1/x) / math.Sqrt(x) * ex
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
		return besselI0Small(x) * math.Exp(-x)
	case x < 500:
		return besselPoly(besselI0Pm[:], 1/x) / math.Sqrt(x)
	}
	return besselPoly(besselI0Pl[:], 1/x) / math.Sqrt(x)
}

// Boost.Math bessel_i0.hpp (53-bit branch) coefficients.
var (
	besselI0Ps = [...]float64{1.00000000000000000e+00, 2.49999999999999909e-01, 2.77777777777782257e-02, 1.73611111111023792e-03, 6.94444444453352521e-05, 1.92901234513219920e-06, 3.93675991102510739e-08, 6.15118672704439289e-10, 7.59407002058973446e-12, 7.59389793369836367e-14, 6.27767773636292611e-16, 4.34709704153272287e-18, 2.63417742690109154e-20, 1.13943037744822825e-22, 9.07926920085624812e-25}
	besselI0Pm = [...]float64{3.98942280401425088e-01, 4.98677850604961985e-02, 2.80506233928312623e-02, 2.92211225166047873e-02, 4.44207299493659561e-02, 1.30970574605856719e-01, -3.35052280231727022e+00, 2.33025711583514727e+02, -1.13366350697172355e+04, 4.24057674317867331e+05, -1.23157028595698731e+07, 2.80231938155267516e+08, -5.01883999713777929e+09, 7.08029243015109113e+10, -7.84261082124811106e+11, 6.76825737854096565e+12, -4.49034849696138065e+13, 2.24155239966958995e+14, -8.13426467865659318e+14, 2.02391097391687777e+15, -3.08675715295370878e+15, 2.17587543863819074e+15}
	besselI0Pl = [...]float64{3.98942280401432905e-01, 4.98677850491434560e-02, 2.80506308916506102e-02, 2.92179096853915176e-02, 4.53371208762579442e-02}
)

func besselI0Small(x float64) float64 {
	a := x * x / 4
	return a*besselPoly(besselI0Ps[:], a) + 1
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
	case math.IsNaN(x):
		return x
	case x < 7.75:
		return sign * besselI1Small(x)
	case x < 500:
		return sign * math.Exp(x) * besselPoly(besselI1Pm[:], 1/x) / math.Sqrt(x)
	}
	ex := math.Exp(x / 2)
	return sign * ex * besselPoly(besselI1Pl[:], 1/x) / math.Sqrt(x) * ex
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
		return sign * besselI1Small(x) * math.Exp(-x)
	case x < 500:
		return sign * besselPoly(besselI1Pm[:], 1/x) / math.Sqrt(x)
	}
	return sign * besselPoly(besselI1Pl[:], 1/x) / math.Sqrt(x)
}

// Boost.Math bessel_i1.hpp (53-bit branch) coefficients.
var (
	besselI1Ps = [...]float64{8.333333333333333803e-02, 6.944444444444341983e-03, 3.472222222225921045e-04, 1.157407407354987232e-05, 2.755731926254790268e-07, 4.920949692800671435e-09, 6.834657311305621830e-11, 7.593969849687574339e-13, 6.904822652741917551e-15, 5.220157095351373194e-17, 3.410720494727771276e-19, 1.625212890947171108e-21, 1.332898928162290861e-23}
	besselI1Pm = [...]float64{3.989422804014406054e-01, -1.496033551613111533e-01, -4.675104253598537322e-02, -4.090895951581637791e-02, -5.719036414430205390e-02, -1.528189554374492735e-01, 3.458284470977172076e+00, -2.426181371595021021e+02, 1.178785865993440669e+04, -4.404655582443487334e+05, 1.277677779341446497e+07, -2.903390398236656519e+08, 5.192386898222206474e+09, -7.313784438967834057e+10, 8.087824484994859552e+11, -6.967602516005787001e+12, 4.614040809616582764e+13, -2.298849639457172489e+14, 8.325554073334618015e+14, -2.067285045778906105e+15, 3.146401654361325073e+15, -2.213318202179221945e+15}
	besselI1Pl = [...]float64{3.989422804014314820e-01, -1.496033551467584157e-01, -4.675105322571775911e-02, -4.090421597376992892e-02, -5.843630344778927582e-02}
)

func besselI1Small(x float64) float64 {
	a := x * x / 4
	q := [3]float64{1, 0.5, besselPoly(besselI1Ps[:], a)}
	return x * besselPoly(q[:], a) / 2
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
	case x <= 1:
		return besselK0Small(x)
	case x < 700:
		return besselK0Large(x) * math.Exp(-x) / math.Sqrt(x)
	}
	ex := math.Exp(-x / 2)
	return besselK0Large(x) * ex / math.Sqrt(x) * ex
}

// K0e returns K0(x) e^x for x > 0 (NaN for x < 0, +Inf at 0, 0 at +Inf).
func K0e(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case x <= 1:
		return besselK0Small(x) * math.Exp(x)
	}
	return besselK0Large(x) / math.Sqrt(x)
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
	case x <= 1:
		return besselK1Small(x)
	case x < 700:
		return besselK1Large(x) * math.Exp(-x) / math.Sqrt(x)
	}
	ex := math.Exp(-x / 2)
	return besselK1Large(x) * ex / math.Sqrt(x) * ex
}

// K1e returns K1(x) e^x for x > 0 (NaN for x < 0, +Inf at 0, 0 at +Inf).
func K1e(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(1)
	case x <= 1:
		return besselK1Small(x) * math.Exp(x)
	}
	return besselK1Large(x) / math.Sqrt(x)
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
