package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Bessel functions J and Y of real order. The structure is that of
// Boost.Math bessel_jy.hpp (Xiaogang Zhang, John Maddock; Boost Software
// License 1.0): Temme's series for Y near the origin, Steed's method with
// the continued fractions CF1 and CF2 elsewhere, small-argument series where
// they converge without cancellation. The large-argument branch is replaced
// by besselAsymJY (modulus-phase, phase in double-double), and the Gamma
// function ratios of Temme's method come from the Taylor series of 1/Gamma
// rather than from Gamma itself.

// besselMaxIter bounds every continued fraction and series; CF1 needs
// O(x) steps when x > v.
const besselMaxIter = 100_000_000

// besselMaxOrder is the largest order the recurrences are run for; beyond
// it only the small-x series and the large-x expansion are available.
const besselMaxOrder = 1e7

// besselLogPowGamma returns log((x/2)^v / Gamma(v+1)) as a double-double,
// for v >= 0 and finite x > 0.
func besselLogPowGamma(v, x float64) dd.DD {
	lx := dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo})
	var lg dd.DD
	if v <= 1 {
		lg = log1pDD(gamma1pm1(v))
	} else {
		lg = dd.Add(logGammaPos(v), dd.Log(v)) // log Gamma(v+1), v exact
	}
	return dd.Sub(dd.MulF(lx, v), lg)
}

// besselRGammaPair returns g1 = (1/Gamma(1-u) - 1/Gamma(1+u))/(2u) and
// g2 = (1/Gamma(1-u) + 1/Gamma(1+u))/2 for |u| <= 1/2, Temme's Gamma_1 and
// Gamma_2 (N. M. Temme, "On the numerical evaluation of the ordinary Bessel
// function of the second kind", J. Comput. Phys. 21, 1976), from the Taylor
// series 1/Gamma(z) = sum c_k z^k (A&S 6.1.34): g1 = -sum c_{2m} u^(2m-2),
// g2 = sum c_{2m+1} u^(2m). Both are free of cancellation.
func besselRGammaPair(u float64) (g1, g2 float64) {
	u2 := u * u
	c := besselRGammaTaylor[:]
	for k := len(c) - 1; k >= 1; k -= 2 { // c[k] = c_{k+1}: even index k+1
		g1 = g1*u2 + c[k]
	}
	for k := len(c) - 2; k >= 0; k -= 2 {
		g2 = g2*u2 + c[k]
	}
	return -g1, g2
}

// besselTemmeJY returns Y_u(x) and Y_{u+1}(x) for |u| <= 1/2 and
// 0 < x <= 2 by Temme's series (Temme 1976; Boost temme_jy).
func besselTemmeJY(u, x float64) (yu, yu1 float64) {
	g1, g2 := besselRGammaPair(u)
	rgp := g2 - u*g1 // 1/Gamma(1+u)
	rgm := g2 + u*g1 // 1/Gamma(1-u)
	spv2 := sinPi(u / 2)
	xp := math.Pow(x/2, u)
	a := math.Log(x / 2)
	sigma := -a * u
	d := 1.0
	if math.Abs(sigma) >= 0x1p-52 {
		d = math.Sinh(sigma) / sigma
	}
	e := u * math.Pi * math.Pi / 2
	if math.Abs(u) >= 0x1p-52 {
		e = 2 * spv2 * spv2 / u
	}
	vspv := 1 / math.Pi
	if math.Abs(u) >= 0x1p-52 {
		vspv = u / sinPi(u)
	}
	f := (g1*math.Cosh(sigma) - g2*a*d) * 2 * vspv
	p := vspv / xp * rgm
	q := vspv * xp * rgp
	g := f + e*q
	h := p
	coef := 1.0
	sum := coef * g
	sum1 := coef * h
	v2 := u * u
	cm := -x * x / 4
	for k := 1; k < 1000; k++ {
		fk := float64(k)
		f = (fk*f + p + q) / (fk*fk - v2)
		p /= fk - u
		q /= fk + u
		g = f + e*q
		h = p - fk*g
		coef *= cm / fk
		sum += coef * g
		sum1 += coef * h
		if math.Abs(coef*g) < math.Abs(sum)*0x1p-53 {
			break
		}
	}
	return -sum, -2 * sum1 / x
}

// besselCF1JY evaluates the continued fraction for J_{v+1}/J_v (A&S
// 9.1.73) by the modified Lentz method (W. J. Lentz, Applied Optics 15,
// 1976); s is the sign of J_v relative to the starting value.
func besselCF1JY(v, x float64) (fv dd.DD, s int) {
	const tiny = 0x1p-511
	C, f := tiny, tiny
	D := 0.0
	s = 1
	for k := 1; k < besselMaxIter; k++ {
		b := 2 * (v + float64(k)) / x
		C = b - 1/C
		D = b - D
		if C == 0 {
			C = tiny
		}
		if D == 0 {
			D = tiny
		}
		D = 1 / D
		delta := C * D
		f *= delta
		if D < 0 {
			s = -s
		}
		if math.Abs(delta-1) < 0x1p-52 {
			// Re-evaluate the converged fraction from its tail: the
			// backward evaluation rounds once per level instead of
			// accumulating one rounding per level in the product f.
			return besselCF1Tail(v, x, 2*k+20, -1), s
		}
	}
	return dd.F(-f), s
}

// besselCF1Tail evaluates 1/(b_1 + sg/(b_2 + sg/(b_3 + ...))) with
// b_j = 2(v+j)/x down from level n, in double-double when n is moderate
// (near the turning point x ~ v the levels neither damp nor amplify
// rounding errors, and the ratio is then differenced against v/x).
func besselCF1Tail(v, x float64, n int, sg float64) dd.DD {
	if n > 100000 {
		r := 0.0
		for j := n; j >= 1; j-- {
			r = 1 / (2*(v+float64(j))/x + sg*r)
		}
		return dd.F(r)
	}
	var r dd.DD
	for j := n; j >= 1; j-- {
		h, l := dd.TwoSum(v, float64(j))
		b := dd.DivF(dd.DD{Hi: 2 * h, Lo: 2 * l}, x)
		r = dd.Div(dd.F(1), dd.Add(b, dd.MulF(r, sg)))
	}
	return r
}

// besselCF2JY evaluates p + iq = (J'_v + iY'_v)/(J_v + iY_v) for x > 1 by
// the complex continued fraction of Steed's method (A. R. Barnett et al.,
// Comput. Phys. Commun. 8, 1974; Boost CF2_jy), in real arithmetic.
func besselCF2JY(v, x float64) (p, q dd.DD) {
	const tiny = 0x1p-511
	fr, fi := -0.5/x, 1.0
	Cr, Ci := fr, fi
	v2 := v * v
	a := (0.25 - v2) / x
	br, bi := 2*x, 2.0
	temp := Cr*Cr + 1
	Ci = bi + a*Cr/temp
	Cr = br + a/temp
	Dr, Di := br, bi
	if math.Abs(Cr)+math.Abs(Ci) < tiny {
		Cr = tiny
	}
	temp = Dr*Dr + Di*Di
	Dr = Dr / temp
	Di = -Di / temp
	dr := Cr*Dr - Ci*Di
	di := Ci*Dr + Cr*Di
	temp = fr
	fr = temp*dr - fi*di
	fi = temp*di + fi*dr
	for k := 2; k < besselMaxIter; k++ {
		a = float64(k) - 0.5
		a = a*a - v2
		bi += 2
		temp = Cr*Cr + Ci*Ci
		Cr = br + a*Cr/temp
		Ci = bi - a*Ci/temp
		Dr = br + a*Dr
		Di = bi + a*Di
		if math.Abs(Cr)+math.Abs(Ci) < tiny {
			Cr = tiny
		}
		if math.Abs(Dr)+math.Abs(Di) < tiny {
			Dr = tiny
		}
		temp = Dr*Dr + Di*Di
		Dr = Dr / temp
		Di = -Di / temp
		dr = Cr*Dr - Ci*Di
		di = Ci*Dr + Cr*Di
		temp = fr
		fr = temp*dr - fi*di
		fi = temp*di + fi*dr
		if math.Abs(dr-1)+math.Abs(di) < 0x1p-52 {
			return besselCF2Tail(v, x, 2*k+20)
		}
	}
	return dd.F(fr), dd.F(fi)
}

// besselCF2Tail evaluates the fraction of besselCF2JY from level n down in
// double-double complex arithmetic: p + iq = -1/(2x) + i + (i/x) T with
// T = a_1/(b_1 + a_2/(b_2 + ...)), a_k = (k-1/2)^2 - v^2, b_k = 2(x + ik).
func besselCF2Tail(v, x float64, n int) (p, q dd.DD) {
	if n > 100000 {
		n = 100000
	}
	v2 := dd.MulFF(v, v)
	var tr, ti dd.DD
	for k := n; k >= 1; k-- {
		h := float64(k) - 0.5
		a := dd.Sub(dd.F(h*h), v2)
		dr := dd.AddF(tr, 2*x)
		di := dd.AddF(ti, 2*float64(k))
		m := dd.Add(dd.Mul(dr, dr), dd.Mul(di, di))
		tr = dd.Div(dd.Mul(a, dr), m)
		ti = dd.Neg(dd.Div(dd.Mul(a, di), m))
	}
	p = dd.Neg(dd.DivF(dd.AddF(ti, 0.5), x))
	q = dd.AddF(dd.DivF(tr, x), 1)
	return p, q
}

// besselJSeries returns J_v(x) for v >= 0 by its power series
// (x/2)^v/Gamma(v+1) * sum (-x^2/4)^k / (k! (v+1)_k) (DLMF 10.2.2), the
// prefix formed as exp of a double-double logarithm.
func besselJSeries(v, x float64) float64 {
	L := besselLogPowGamma(v, x)
	if L.Hi < -746 {
		return 0
	}
	m := -x * x / 4
	t, s := 1.0, 1.0
	for k := 1; k < 1000; k++ {
		fk := float64(k)
		t *= m / (fk * (fk + v))
		s += t
		if math.Abs(t) <= 0x1p-54*math.Abs(s) {
			break
		}
	}
	return besselExpMul(L, s)
}

// besselExpMul returns exp(L) * s, without spurious overflow or underflow
// of exp(L) when the product is representable.
func besselExpMul(L dd.DD, s float64) float64 {
	if L.Hi > 700 || L.Hi < -700 {
		k := math.Floor(L.Hi / dd.Ln2Hi)
		L = dd.Sub(L, dd.Add(dd.MulFF(k, dd.Ln2Hi), dd.MulFF(k, dd.Ln2Lo)))
		return math.Ldexp(dd.Exp(L)*s, int(k))
	}
	return dd.Exp(L) * s
}

// besselYSeries returns Y_v(x) for non-integer v > 0 and small x by the
// two power series of (J_v cos(v pi) - J_{-v})/sin(v pi) (Boost
// bessel_y_small_z_series), usable where the second is negligible against
// the first or both are accurate.
func besselYSeries(v, x float64) float64 {
	L := besselLogPowGamma(v, x)
	m := -x * x / 4
	ta, sa := 1.0, 1.0
	tb, sb := 1.0, 1.0
	for k := 1; k < 1000; k++ {
		fk := float64(k)
		ta *= m / (fk * (fk - v))
		tb *= m / (fk * (fk + v))
		sa += ta
		sb += tb
		if math.Abs(ta) <= 0x1p-54*math.Abs(sa) && math.Abs(tb) <= 0x1p-54*math.Abs(sb) {
			break
		}
	}
	// -Gamma(v)/(pi (x/2)^v) sa + (x/2)^v/Gamma(v+1) cot(v pi) sb
	first := -besselExpMul(dd.Neg(L), sa/(math.Pi*v))
	second := 0.0
	if L.Hi > -746 {
		second = besselExpMul(L, sb*cosPi(v)/sinPi(v))
	}
	return first + second
}

// besselJYPos returns J_v(x) and Y_v(x) for v >= 0 and finite x > 0 (only
// what is asked for is guaranteed). Y may be -Inf (overflow).
func besselJYPos(v, x float64, needJ, needY bool) (J, Y float64) {
	if x > 20 {
		if j, y, ok := besselAsymJY(v, x); ok {
			return j, y
		}
	}
	if !needY && x*x/4 <= v+1 {
		return besselJSeries(v, x), math.NaN()
	}
	if v > besselMaxOrder {
		// Only the series is available this far out.
		if math.Log(x*x/4/v) < -1 {
			return besselJSeries(v, x), math.Inf(-1)
		}
		return math.NaN(), math.NaN()
	}
	n := int(math.Floor(v + 0.5))
	u := v - float64(n) // -1/2 <= u < 1/2
	if x < 1 && u != 0 && math.Log(0x1p-54) > v*math.Log(x*x/4/v) {
		J = math.NaN()
		if needJ {
			J = besselJSeries(v, x)
		}
		return J, besselYSeries(v, x)
	}
	W := 2 / (x * math.Pi) // Wronskian
	scale := 1.0           // Y_v = Yv / scale
	var Yv, Yv1 float64
	if x <= 2 {
		prev, cur := besselTemmeJY(u, x)
		for k := 1; k <= n; k++ {
			fact := 2 * (u + float64(k)) / x
			if (math.MaxFloat64-math.Abs(prev))/fact < math.Abs(cur) {
				scale /= cur
				prev /= cur
				cur = 1
			}
			prev, cur = cur, fact*cur-prev
		}
		Yv, Yv1 = prev, cur
		J = math.NaN()
		if needJ {
			fv, _ := besselCF1JY(v, x)
			J = scale * W / (Yv*fv.Float() - Yv1)
		}
	} else if x >= v {
		// Steed's method at the order v itself (Barnett et al. 1974): with
		// x >= v CF2 converges quickly and no recurrence crosses the
		// oscillatory region, where it would accumulate error.
		fv, s := besselCF1JY(v, x)
		j, g := besselSteed(v, x, fv)
		J = float64(s) * j.Float()
		Yv = float64(s) * dd.Mul(g, j).Float()
	} else {
		fvd, s := besselCF1JY(v, x)
		fv := fvd.Float()
		const init = 0x1p-511
		prev := fv * float64(s) * init
		cur := float64(s) * init
		over := false
		for k := n; k > 0; k-- {
			t := 2 * (u + float64(k)) / x
			if t > 1 && math.MaxFloat64/t < math.Abs(cur) {
				over = true
				break
			}
			next := t*cur - prev
			if next == 0 {
				next = prev * 0x1p-53
			}
			prev, cur = cur, next
		}
		ratio, fu := 0.0, 1.0
		if !over {
			ratio = float64(s) * init / cur
			fu = prev / cur
		}
		pd, qd := besselCF2JY(u, x)
		p, q := pd.Float(), qd.Float()
		t := u/x - fu // J'_u/J_u
		gamma := (p - t) / q
		if gamma == 0 {
			gamma = u * 0x1p-52 / x
		}
		Ju := math.Copysign(math.Sqrt(W/(q+gamma*(p-t))), cur)
		J = Ju * ratio
		Yu := gamma * Ju
		Yu1 := Yu * (u/x - p - q/gamma)
		prev, cur = Yu, Yu1
		if needY {
			for k := 1; k <= n; k++ {
				fact := 2 * (u + float64(k)) / x
				if (math.MaxFloat64-math.Abs(prev))/fact < math.Abs(cur) {
					prev /= cur
					scale /= cur
					cur = 1
				}
				prev, cur = cur, fact*cur-prev
			}
		}
		Yv = prev
	}
	if math.MaxFloat64*math.Abs(scale) < math.Abs(Yv) {
		Y = math.Inf(-1) // Y_v -> -Inf as x -> 0 for v > 0
		if Yv*scale > 0 {
			Y = math.Inf(1)
		}
	} else {
		Y = Yv / scale
	}
	return J, Y
}

// besselSteed returns |J_nu(x)| and gamma = Y_nu/J_nu from the ratio
// fv = J_{nu+1}/J_nu (CF1) and p + iq (CF2), in double-double: with
// t = J'/J = nu/x - fv, gamma = (p - t)/q and the Wronskian gives
// J^2 = (2/(pi x)) / (q + gamma (p - t)) (Barnett et al. 1974).
func besselSteed(nu, x float64, fv dd.DD) (j, gamma dd.DD) {
	p, q := besselCF2JY(nu, x)
	t := dd.Sub(dd.DivF(dd.F(nu), x), fv)
	pt := dd.Sub(p, t)
	gamma = dd.Div(pt, q)
	den := dd.Add(q, dd.Mul(gamma, pt))
	W := dd.DivF(besselTwoOverPiDD, x)
	return dd.Sqrt(dd.Div(W, den)), gamma
}

// besselJYRefl returns J_v and Y_v for v < 0 (non-integer) from the
// functions of order nu = -v: J_v = cos(nu pi) J_nu - sin(nu pi) Y_nu,
// Y_v = sin(nu pi) J_nu + cos(nu pi) Y_nu (DLMF 10.4.6/7, as reflection).
func besselJYRefl(v, x float64) (J, Y float64) {
	if x > 20 {
		if j, y, ok := besselAsymJY(v, x); ok {
			return j, y
		}
	}
	nu := -v
	jn, yn := besselJYPos(nu, x, true, true)
	c, s := cosPi(nu), sinPi(nu)
	J = c*jn - s*yn
	Y = s*jn + c*yn
	if c == 0 {
		Y = s * jn
	}
	return J, Y
}

// besselIsInt reports whether v is an integer small enough for the
// integer-order routines.
func besselIsInt(v float64) bool {
	return v == math.Trunc(v) && math.Abs(v) <= besselMaxOrder
}

// Jv returns the Bessel function of the first kind of real order v,
//
//	J_v(x) = sum_k (-1)^k (x/2)^(2k+v) / (k! Gamma(k+v+1))   (DLMF 10.2.2).
//
// For integer v it is Jn(int(v), x). For non-integer v the function is
// real only for x >= 0: Jv(v, x < 0) = NaN. Jv(v, 0) is 0 for v > 0 and
// ±Inf (the sign of 1/Gamma(v+1)) for non-integer v < 0; Jv(v, +Inf) = 0;
// Jv(+Inf, x) = 0 and Jv(-Inf, x) = NaN; NaN in, NaN out.
//
// Algorithm: Boost.Math bessel_jy.hpp (Temme's method for x <= 2, Steed's
// CF1/CF2 method above, power series for small x), with negative orders by
// the reflection formula J_{-nu} = cos(nu pi) J_nu - sin(nu pi) Y_nu with
// exact sinPi/cosPi, and Hankel's expansion in modulus-phase form with a
// double-double phase for large x (besselAsymJY). Orders beyond 1e7 are
// supported only where the power series or the large-x expansion applies;
// elsewhere the result is NaN.
func Jv(v, x float64) float64 {
	switch {
	case math.IsNaN(v) || math.IsNaN(x):
		return math.NaN()
	case besselIsInt(v):
		return Jn(int(v), x)
	case math.IsInf(v, 1):
		if math.IsInf(x, 0) {
			return math.NaN()
		}
		if x < 0 {
			return math.NaN()
		}
		return 0
	case math.IsInf(v, -1) || x < 0:
		return math.NaN()
	case math.IsInf(x, 1):
		return 0
	case x == 0:
		if v > 0 {
			return 0
		}
		return math.Copysign(math.Inf(1), -sinPi(v))
	case v > 0:
		j, _ := besselJYPos(v, x, true, false)
		return j
	}
	j, _ := besselJYRefl(v, x)
	return j
}

// Yv returns the Bessel function of the second kind of real order v,
// Y_v(x) = (J_v(x) cos(v pi) - J_{-v}(x)) / sin(v pi) (DLMF 10.2.3), and
// its limit for integer v (Yn). It is real only for x >= 0: Yv(v, x < 0) =
// NaN. Yv(v, 0) = -Inf for v > 0 or v = 0; for v < 0 it is the limit
// -Inf*cos(v pi) (0 at half-integers, where Y_{-n-1/2} = ±J_{n+1/2});
// Yv(v, +Inf) = 0; Yv(+Inf, x > 0) = -Inf, Yv(-Inf, x) = NaN; NaN in, NaN
// out. Large negative values overflow to -Inf.
//
// Algorithm: as Jv (Boost.Math bessel_jy.hpp with the large-x branch
// replaced by the double-double modulus-phase expansion).
func Yv(v, x float64) float64 {
	switch {
	case math.IsNaN(v) || math.IsNaN(x) || x < 0:
		return math.NaN()
	case besselIsInt(v):
		return Yn(int(v), x)
	case math.IsInf(v, 1):
		if math.IsInf(x, 1) {
			return math.NaN()
		}
		return math.Inf(-1)
	case math.IsInf(v, -1):
		return math.NaN()
	case math.IsInf(x, 1):
		return 0
	case x == 0:
		if v > 0 {
			return math.Inf(-1)
		}
		c := cosPi(-v)
		if c == 0 {
			return 0
		}
		return math.Copysign(math.Inf(1), -c)
	case v > 0:
		_, y := besselJYPos(v, x, false, true)
		return y
	}
	_, y := besselJYRefl(v, x)
	return y
}

// Jn returns the Bessel function of the first kind of integer order n,
// for any real x: J_{-n}(x) = (-1)^n J_n(x) and J_n(-x) = (-1)^n J_n(x).
// Jn(n, 0) = 0 for n != 0, Jn(n, ±Inf) = 0, Jn(n, NaN) = NaN.
//
// Algorithm: J0 and J1 for n = 0, 1; the double-double modulus-phase Hankel
// expansion for large x; forward recurrence from J0 and J1 for n < x;
// otherwise the power series (x < 5 or n > x^2/4) or backward recurrence
// from Steed's CF1, normalised by whichever of J0(x), J1(x) is larger
// (Boost.Math bessel_jn.hpp normalises by J0 only, which fails next to its
// zeros).
func Jn(n int, x float64) float64 {
	sign := 1.0
	if n < 0 {
		n = -n
		if n&1 == 1 {
			sign = -sign
		}
	}
	if x < 0 {
		x = -x
		if n&1 == 1 {
			sign = -sign
		}
	}
	switch {
	case math.IsNaN(x):
		return x
	case n == 0:
		return sign * J0(x)
	case n == 1:
		return sign * J1(x)
	case math.IsInf(x, 0):
		return 0
	case x == 0:
		return sign * 0
	}
	v := float64(n)
	if x*x/4 <= v+1 {
		return sign * besselJSeries(v, x)
	}
	j, _ := besselJYPos(v, x, true, false)
	return sign * j
}

// Yn returns the Bessel function of the second kind of integer order n for
// x >= 0, with Y_{-n}(x) = (-1)^n Y_n(x). Yn(n, 0) = -Inf (+Inf for odd
// negative n), Yn(n, x < 0) = NaN, Yn(n, +Inf) = 0, Yn(n, NaN) = NaN; large
// negative values overflow to -Inf.
//
// Algorithm: Y0 and Y1 for n = 0, 1; the double-double modulus-phase Hankel
// expansion for large x; otherwise forward recurrence from Y0 and Y1, which
// is stable for Y (Boost.Math bessel_yn.hpp).
func Yn(n int, x float64) float64 {
	sign := 1.0
	if n < 0 {
		n = -n
		if n&1 == 1 {
			sign = -1
		}
	}
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case n == 0:
		return Y0(x)
	case x == 0:
		return sign * math.Inf(-1)
	case n == 1:
		return sign * Y1(x)
	case math.IsInf(x, 1):
		return 0
	}
	_, y := besselJYPos(float64(n), x, false, true)
	return sign * y
}
