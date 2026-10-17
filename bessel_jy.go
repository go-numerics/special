package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Bessel functions J and Y of real order. The structure is that of
// Boost.Math bessel_jy.hpp (Xiaogang Zhang, John Maddock; Boost Software
// License 1.0): Temme's series for Y near the origin, Steed's method with
// the continued fractions CF1 and CF2 elsewhere, small-argument series where
// they converge without cancellation. Differences, all for accuracy:
//   - the continued fractions are re-evaluated backwards in double-double
//     from the depth at which Lentz's iteration converged, and Steed's
//     normalisation and the recurrences run in double-double;
//   - Steed's method is applied at the order v itself when x >= v (no
//     recurrence through the oscillatory region), and at negative orders
//     directly (no reflection formula and its cancellation);
//   - the large-argument branch is besselAsymJY (modulus-phase, phase in
//     double-double), with a fallback to Steed's method next to a zero;
//   - the Gamma function ratios of Temme's method come from the Taylor
//     series of 1/Gamma, the series prefixes from double-double logarithms.

// besselMaxIter bounds every continued fraction and series; CF1 needs
// O(x) steps when x > v.
const besselMaxIter = 100_000_000

// besselMaxOrder is the largest order the recurrences are run for; beyond
// it only the small-x series and the large-x expansion are available.
const besselMaxOrder = 1e7

// besselSteedMaxX is the largest x for which Steed's method (O(x) steps in
// CF1) stands in for the large-x expansion next to the zeros.
const besselSteedMaxX = 1e6

// besselTemmeMaxX is where Steed's method takes over from Temme's series
// (Boost switches at 2). Temme's sums cancel next to the zeros of Y_u and
// are evaluated in float64; Steed's method in double-double keeps the
// relative accuracy there, at the price of a slower CF2 for small x.
const besselTemmeMaxX = 0.25

// besselLogPowGamma returns log|(x/2)^v / Gamma(v+1)| as a double-double
// and the sign of Gamma(v+1), for finite x > 0 and v >= 0 or v < 0 not an
// integer.
func besselLogPowGamma(v, x float64) (dd.DD, float64) {
	lx := dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo})
	var lg dd.DD
	sg := 1.0
	switch {
	case v < 0:
		lg = logGammaAbsDD(v + 1)
		sg = GammaSign(v + 1)
	case v <= 1:
		lg = log1pDD(gamma1pm1(v))
	default:
		lg = dd.Add(logGammaPos(v), dd.Log(v)) // log Gamma(v+1), v exact
	}
	return dd.Sub(dd.MulF(lx, v), lg), sg
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
	xp := besselPowHalf(x, u)
	a := besselLogHalf(x)
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
	// Lentz's iteration finds the depth k at which the fraction has
	// converged and the sign; the value is then re-evaluated from that
	// depth (doubled) backwards in double-double by besselCF1Tail, which
	// rounds once per level instead of accumulating one rounding per level
	// in a running product.
	C, D := 0x1p-511, 0.0
	s = 1
	k := 1
	for ; k < besselMaxIter; k++ {
		b := 2 * (v + float64(k)) / x
		C = b - 1/C
		D = 1 / (b - D)
		if D < 0 {
			s = -s
		}
		if math.Abs(C*D-1) < 0x1p-52 {
			break
		}
	}
	return besselCF1Tail(v, x, 2*k+20, -1), s
}

// besselCF1Tail evaluates 1/(b_1 + sg/(b_2 + sg/(b_3 + ...))) with
// b_j = 2(v+j)/x down from level n, to about double-double accuracy when n
// is moderate (near the turning point x ~ v the levels neither damp nor
// amplify rounding errors, and the ratio is then differenced against v/x).
//
// Each level is carried as an unrenormalised pair: b_j from TwoSum(v, j)
// and the FMA remainder of 2(v+j)·(1/x), the denominator from TwoSum
// (renormalised: it cancels next to a zero), and its reciprocal corrected
// by its FMA remainder (one division per level, about a third of the dd
// package's operations).
func besselCF1Tail(v, x float64, n int, sg float64) dd.DD {
	if n > 100000 {
		r := 0.0
		for j := n; j >= 1; j-- {
			r = 1 / (2*(v+float64(j))/x + sg*r)
		}
		return dd.F(r)
	}
	ix := 1 / x
	ixl := math.FMA(-ix, x, 1) * ix // 1/x = ix + ixl to about 2^-104
	rh, rl := 0.0, 0.0
	for j := n; j >= 1; j-- {
		h, l := dd.TwoSum(v, float64(j))
		h, l = 2*h, 2*l
		bh := float64(h * ix) // rounded: it feeds the TwoSum below
		bl := math.FMA(h, ix, -bh) + (h*ixl + l*ix)
		wh, we := dd.TwoSum(bh, sg*rh)
		// renormalised: b_j and r cancel next to a zero of the function
		wh, wl := dd.FastTwoSum(wh, we+bl+sg*rl)
		rh = 1 / wh
		rl = (math.FMA(-rh, wh, 1) - rh*wl) * rh
	}
	h, l := dd.FastTwoSum(rh, rl)
	return dd.DD{Hi: h, Lo: l}
}

// besselCF2JY evaluates p + iq = (J'_v + iY'_v)/(J_v + iY_v) for x > 1 by
// the complex continued fraction of Steed's method (A. R. Barnett et al.,
// Comput. Phys. Commun. 8, 1974; Boost CF2_jy), in real arithmetic.
func besselCF2JY(v, x float64) (p, q dd.DD) {
	// Lentz's iteration in float64 to find the depth of convergence, then
	// the double-double evaluation from (four times) that depth.
	fr, fi := -0.5/x, 1.0
	Cr, Ci := fr, fi
	v2 := v * v
	a := (0.25 - v2) / x
	br, bi := 2*x, 2.0
	temp := Cr*Cr + 1
	Ci = bi + a*Cr/temp
	Cr = br + a/temp
	Dr, Di := br, bi
	k := 2
	for ; k < besselMaxIter; k++ {
		a = float64(k) - 0.5
		a = a*a - v2
		bi += 2
		temp = Cr*Cr + Ci*Ci
		Cr = br + a*Cr/temp
		Ci = bi - a*Ci/temp
		Dr = br + a*Dr
		Di = bi + a*Di
		temp = Dr*Dr + Di*Di
		Dr = Dr / temp
		Di = -Di / temp
		dr := Cr*Dr - Ci*Di
		di := Ci*Dr + Cr*Di
		if math.Abs(dr-1)+math.Abs(di) < 0x1p-52 {
			break
		}
	}
	return besselCF2Tail(v, x, 4*k+40)
}

// besselCF2Tail evaluates the fraction of besselCF2JY from level n down in
// double-double complex arithmetic: p + iq = -1/(2x) + i + (i/x) T with
// T = a_1/(b_1 + a_2/(b_2 + ...)), a_k = (k-1/2)^2 - v^2, b_k = 2(x + ik).
func besselCF2Tail(v, x float64, n int) (p, q dd.DD) {
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

// besselSeries returns (x/2)^v / Gamma(v+1) * sum_k (sg x^2/4)^k / (k! (v+1)_k),
// times e^-x when scaled, as m * 2^e (so that it neither overflows nor
// underflows): with sg = -1 this is J_v(x) (DLMF 10.2.2), with sg = +1
// I_v(x) (DLMF 10.25.2). v >= 0, or v < 0 not an integer; the prefix is
// formed as the exponential of a double-double logarithm.
func besselSeries(v, x, sg float64, scaled bool) (m float64, e int) {
	L, gs := besselLogPowGamma(v, x)
	if scaled {
		L = dd.AddF(L, -x)
	}
	switch { // beyond any representable value (keeps int(k) small)
	case L.Hi > 1e5:
		return gs * math.Inf(1), 0
	case L.Hi < -1e5:
		return 0, 0
	}
	// terms and sum in double-double: for I the terms may grow a long way
	// before they fall, and one rounding per term would add up
	q := dd.MulF(dd.MulFF(x, x), sg/4)
	t, s := dd.F(1), dd.F(1)
	for k := 1; k < 100000; k++ {
		fk := float64(k)
		h, l := dd.TwoSum(fk, v)
		t = dd.Div(dd.Mul(t, q), dd.MulF(dd.DD{Hi: h, Lo: l}, fk))
		s = dd.Add(s, t)
		if math.Abs(t.Hi) <= 0x1p-60*math.Abs(s.Hi) {
			break
		}
	}
	k := math.Floor(L.Hi / dd.Ln2Hi)
	r := dd.Sub(L, dd.Add(dd.MulFF(k, dd.Ln2Hi), dd.MulFF(k, dd.Ln2Lo)))
	return gs * dd.MulF(s, dd.Exp(r)).Float(), int(k)
}

// besselJSeries returns J_v(x) by its power series (see besselSeries).
func besselJSeries(v, x float64) float64 {
	m, e := besselSeries(v, x, -1, false)
	return math.Ldexp(m, e)
}

// besselSeriesOK reports whether the power series of J_v or I_v at x
// converges without cancellation worth speaking of: x^2/4 must be small
// against every k (k + v), k >= 1, which for v < 0 includes the levels
// where k + v passes near 0.
func besselSeriesOK(v, x, ratio float64) bool {
	m := v + 1
	if v < 0 {
		// |k (k+v)| over k >= 1 is least at k = 1 or next to k = -v
		m = math.Abs(1 + v)
		for _, k := range []float64{math.Floor(-v), math.Ceil(-v)} {
			if k >= 1 {
				m = math.Min(m, math.Abs(k*(k+v)))
			}
		}
	}
	return x*x/4 <= ratio*m
}

// besselExpMul returns exp(L) * s, without spurious overflow or underflow
// of exp(L) when the product is representable.
func besselExpMul(L dd.DD, s float64) float64 {
	// beyond any representable product (keeps int(k) small on 32-bit)
	switch {
	case L.Hi > 1e5:
		return s * math.Inf(1)
	case L.Hi < -1e5:
		return s * 0
	}
	k := math.Floor(L.Hi / dd.Ln2Hi)
	r := dd.Sub(L, dd.Add(dd.MulFF(k, dd.Ln2Hi), dd.MulFF(k, dd.Ln2Lo)))
	return math.Ldexp(dd.Exp(r)*s, int(k))
}

// besselLdexpExp returns m * 2^e * e^t without intermediate overflow or
// underflow, e^t reduced in double-double.
func besselLdexpExp(m float64, e int, t float64) float64 {
	switch {
	case m == 0 || math.IsNaN(m) || math.IsInf(m, 0):
		return m
	case t > 1e5: // overflows whatever m and e are (|e| stays far below)
		return math.Copysign(math.Inf(1), m)
	case t < -1e5:
		return math.Copysign(0, m)
	}
	k := math.Floor(t / dd.Ln2Hi)
	r := dd.Sub(dd.F(t), dd.Add(dd.MulFF(k, dd.Ln2Hi), dd.MulFF(k, dd.Ln2Lo)))
	fm, fe := math.Frexp(m)
	return math.Ldexp(fm*dd.Exp(r), fe+e+int(k))
}

// besselYSeries returns Y_v(x) for non-integer v > 0 and small x by the
// two power series of (J_v cos(v pi) - J_{-v})/sin(v pi) (Boost
// bessel_y_small_z_series), usable where the second is negligible against
// the first or both are accurate.
func besselYSeries(v, x float64) float64 {
	L, _ := besselLogPowGamma(v, x)
	m := -x * x / 4
	ta, sa := 1.0, 0.0 // tails after the leading 1, added once at the end
	tb, sb := 1.0, 0.0
	for k := 1; k < 1000; k++ {
		fk := float64(k)
		ta *= m / (fk * (fk - v))
		tb *= m / (fk * (fk + v))
		sa += ta
		sb += tb
		if math.Abs(ta) <= 0x1p-56*math.Abs(1+sa) && math.Abs(tb) <= 0x1p-56*math.Abs(1+sb) {
			break
		}
	}
	sa, sb = 1+sa, 1+sb
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
		// Steed's CF1 costs O(x) steps: beyond besselSteedMaxX the
		// expansion is used even next to a zero.
		j, y, ok := besselAsymJY(v, x, needJ, needY)
		if ok || (x > besselSteedMaxX && !math.IsNaN(j)) {
			return j, y
		}
	}
	if !needY && besselSeriesOK(v, x, besselJSeriesRatio) {
		return besselJSeries(v, x), math.NaN()
	}
	if v > besselMaxOrder {
		// Only the series is available this far out.
		if 2*besselLogHalf(x)-besselLog(v) < -1 {
			return besselJSeries(v, x), math.Inf(-1)
		}
		return math.NaN(), math.NaN()
	}
	if needY && x < 1 && v >= 1 && LogGamma(v)-v*besselLogHalf(x)-math.Log(math.Pi) > 710.5 {
		// |Y_v(x)| ~ Gamma(v) (2/x)^v / pi overflows (DLMF 10.7.4).
		J = math.NaN()
		if needJ {
			J = besselJSeries(v, x)
		}
		return J, math.Inf(-1)
	}
	n := int(math.Floor(v + 0.5))
	u := v - float64(n) // -1/2 <= u < 1/2
	// The two power series of Y (besselYSeries) where the second, of
	// relative size (x/2)^(2v) cot(v pi), is negligible or nearly so;
	// Temme's sum loses accuracy there through exp(+-u log(x/2)).
	if x < 1 && u != 0 && 2*v*besselLogHalf(x)+math.Log1p(math.Abs(cosPi(v)/sinPi(v))) < -10 {
		J = math.NaN()
		if needJ {
			J = besselJSeries(v, x)
		}
		return J, besselYSeries(v, x)
	}
	if x <= besselTemmeMaxX {
		// Temme's series for Y_u, Y_{u+1}, forward recurrence to Y_v,
		// Y_{v+1}; J from CF1 and the Wronskian J_{v+1} Y_v - J_v Y_{v+1}
		// = 2/(pi x).
		yu, yu1 := besselTemmeJY(u, x)
		yv, yv1, e := besselRecur(u, x, n, dd.F(yu), dd.F(yu1), -1)
		J = math.NaN()
		if needJ {
			fv, _ := besselCF1JY(v, x)
			den := dd.Sub(dd.Mul(yv, fv), yv1)
			J = math.Ldexp(dd.Div(dd.DivF(besselTwoOverPiDD, x), den).Float(), -e)
		}
		return J, math.Ldexp(yv.Float(), e)
	}
	switch {
	case x >= v:
		// Steed's method at the order v itself (Barnett et al. 1974): with
		// x >= v CF2 converges quickly and no recurrence crosses the
		// oscillatory region, where it would accumulate error.
		fv, s := besselCF1JY(v, x)
		j, g, _, _ := besselSteed(v, x, fv)
		return float64(s) * j.Float(), float64(s) * dd.Mul(g, j).Float()
	}
	// x < v: CF1 at v, backward recurrence (J is minimal there) down to u,
	// Steed's method at u, forward recurrence for Y (Boost bessel_jy).
	fv, s := besselCF1JY(v, x)
	sg := float64(s)
	ju1, ju, eJ := besselRecurDown(u, x, n, dd.MulF(fv, sg), dd.F(sg))
	fu := dd.Div(ju1, ju) // J_{u+1}/J_u
	jabs, g, p, q := besselSteed(u, x, fu)
	Ju := jabs
	if ju.Hi < 0 {
		Ju = dd.Neg(jabs)
	}
	// J_v = J_u * sg / ju * 2^-eJ
	J = math.Ldexp(dd.Div(dd.MulF(Ju, sg), ju).Float(), -eJ)
	if !needY {
		return J, math.NaN()
	}
	Yu := dd.Mul(g, Ju)
	// Y_{u+1} = Y_u (u/x - p - q/gamma)
	Yu1 := dd.Mul(Yu, dd.Sub(dd.Sub(dd.DivF(dd.F(u), x), p), dd.Div(q, g)))
	yv, _, e := besselRecur(u, x, n, Yu, Yu1, -1)
	return J, math.Ldexp(yv.Float(), e)
}

// besselSteed returns |J_nu(x)| and gamma = Y_nu/J_nu from the ratio
// fv = J_{nu+1}/J_nu (CF1) and p + iq (CF2), in double-double: with
// t = J'/J = nu/x - fv, gamma = (p - t)/q and the Wronskian gives
// J^2 = (2/(pi x)) / (q + gamma (p - t)) (Barnett et al. 1974).
func besselSteed(nu, x float64, fv dd.DD) (j, gamma, p, q dd.DD) {
	p, q = besselCF2JY(nu, x)
	t := dd.Sub(dd.DivF(dd.F(nu), x), fv)
	pt := dd.Sub(p, t)
	gamma = dd.Div(pt, q)
	den := dd.Add(q, dd.Mul(gamma, pt))
	W := dd.DivF(besselTwoOverPiDD, x)
	return dd.Sqrt(dd.Div(W, den)), gamma, p, q
}

// besselRecur runs the three-term recurrence
// f_{k+1} = 2(u+k)/x f_k + sg f_{k-1} (sg = -1 for J and Y, +1 for K)
// upwards from f_u = a, f_{u+1} = b to f_{u+n}, f_{u+n+1}, in double-double;
// the values are kept in range by exact power-of-two rescaling and the
// result is (fn * 2^e, fn1 * 2^e).
func besselRecur(u, x float64, n int, a, b dd.DD, sg float64) (fn, fn1 dd.DD, e int) {
	for k := 1; k <= n; k++ {
		h, l := dd.TwoSum(u, float64(k))
		f := dd.DivF(dd.DD{Hi: 2 * h, Lo: 2 * l}, x)
		if math.Abs(b.Hi)*f.Hi > 0x1p1000 || math.Abs(a.Hi) > 0x1p1000 {
			ex := math.Ilogb(b.Hi)
			a, b = besselScaleDD(a, -ex), besselScaleDD(b, -ex)
			e += ex
		}
		a, b = b, dd.Add(dd.Mul(f, b), dd.MulF(a, sg))
	}
	return a, b, e
}

// besselRecurDown runs J_{k-1} = 2k/x J_k - J_{k+1} downwards from
// (J_{u+n+1}, J_{u+n}) = (a, b) to (J_{u+1}, J_u), with exact rescaling;
// the result is (ju1 * 2^e, ju * 2^e) in the units of the input.
func besselRecurDown(u, x float64, n int, a, b dd.DD) (ju1, ju dd.DD, e int) {
	for k := n; k >= 1; k-- {
		h, l := dd.TwoSum(u, float64(k))
		f := dd.DivF(dd.DD{Hi: 2 * h, Lo: 2 * l}, x)
		if math.Abs(b.Hi)*f.Hi > 0x1p1000 || math.Abs(a.Hi) > 0x1p1000 {
			ex := math.Ilogb(b.Hi)
			a, b = besselScaleDD(a, -ex), besselScaleDD(b, -ex)
			e += ex
		}
		a, b = b, dd.Sub(dd.Mul(f, b), a)
	}
	return a, b, e
}

// besselScaleDD returns a * 2^k (exact barring underflow).
func besselScaleDD(a dd.DD, k int) dd.DD {
	return dd.DD{Hi: math.Ldexp(a.Hi, k), Lo: math.Ldexp(a.Lo, k)}
}

// besselJYRefl returns J_v and Y_v for v < 0 (non-integer) from the
// functions of order nu = -v: J_v = cos(nu pi) J_nu - sin(nu pi) Y_nu,
// Y_v = sin(nu pi) J_nu + cos(nu pi) Y_nu (DLMF 10.4.6/7, as reflection).
func besselJYRefl(v, x float64, needJ, needY bool) (J, Y float64) {
	if x > 20 {
		j, y, ok := besselAsymJY(v, x, needJ, needY)
		if ok || (x > besselSteedMaxX && !math.IsNaN(j)) {
			return j, y
		}
	}
	if x > 2 && x >= -v {
		// Steed's method holds for negative orders as well; used directly
		// it avoids the cancellation of the reflection formula.
		fv, s := besselCF1JY(v, x)
		j, g, _, _ := besselSteed(v, x, fv)
		return float64(s) * j.Float(), float64(s) * dd.Mul(g, j).Float()
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
// Algorithm: after Boost.Math bessel_jy.hpp (N. M. Temme, J. Comput. Phys.
// 21, 1976, for Y at x <= 1/4; Steed's method with the continued fractions
// CF1 and CF2 of A. R. Barnett et al., Comput. Phys. Commun. 8, 1974,
// above; the power series for small x), carried in double-double; the
// large-x expansion in modulus-phase form (DLMF 10.18.17-18) with the
// phase in double-double and Payne-Hanek reduction. Negative orders use
// Steed's method directly when x >= |v|, else the reflection formula
// J_{-nu} = cos(nu pi) J_nu - sin(nu pi) Y_nu with exact sinPi/cosPi.
// Orders beyond 1e7 are supported only where the power series or the
// large-x expansion applies; elsewhere the result is NaN. The cost of
// Steed's method grows like x, so above x = 1e6 the large-x expansion is
// used even next to a zero.
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
	j, _ := besselJYRefl(v, x, true, false)
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
// Algorithm: as Jv; for small x and non-integer v, the two power series of
// the definition where the second is negligible.
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
	_, y := besselJYRefl(v, x, false, true)
	return y
}

// Jn returns the Bessel function of the first kind of integer order n,
// for any real x: J_{-n}(x) = (-1)^n J_n(x) and J_n(-x) = (-1)^n J_n(x).
// Jn(n, 0) = 0 for n != 0, Jn(n, ±Inf) = 0, Jn(n, NaN) = NaN.
//
// Algorithm: J0 and J1 for n = 0, 1; otherwise as Jv: the power series for
// small x, the double-double modulus-phase Hankel expansion for large x,
// and Steed's method in double-double between (at the order n itself for
// x >= n, after backward recurrence from n for x < n). The forward
// recurrence from J0 and J1 used by Cephes and Boost is not used: it loses
// the relative accuracy next to the zeros of J_n.
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
	if besselSeriesOK(v, x, besselJSeriesRatio) {
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
// Algorithm: Y0 and Y1 for n = 0, 1; otherwise as Yv (Temme's series and
// forward recurrence for small x, Steed's method in double-double, the
// modulus-phase Hankel expansion for large x).
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

// besselPowHalf returns (x/2)^u as exp(u log(x/2)) with the logarithm in
// double-double: math.Pow rounds u*log(x) to double first, which costs
// |u log x| ulps when x is tiny.
func besselPowHalf(x, u float64) float64 {
	lx := dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo})
	return dd.Exp(dd.MulF(lx, u))
}

// besselJSeriesRatio bounds x^2/4 / (v+1), the ratio of the first two terms
// of the power series of J, where the series is used: the alternating sum
// then cancels by less than a factor 2.
const besselJSeriesRatio = 0.02

// besselLog returns log(x) for x > 0 through dd.Log, which normalises
// subnormal arguments (math.Log on amd64 does not).
func besselLog(x float64) float64 { return dd.Log(x).Hi }

// besselLogHalf returns log(x/2) for x > 0, exact in x (x/2 may round).
func besselLogHalf(x float64) float64 {
	return dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}).Float()
}
