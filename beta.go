package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// sumDD returns a + b exactly as a double-double.
func sumDD(a, b float64) dd.DD {
	h, l := dd.TwoSum(a, b)
	return dd.DD{Hi: h, Lo: l}
}

// log1pmxDD returns log(1+x) - x for a double-double x > -1. For
// |x| < 2^-10 it sums the series -x²/2 + x³/3 - x⁴/4 + ... with the first
// two terms in double-double (the float64 rest is below 2^-10 of the
// result); beyond, log1p(x) - x with log1pDD accurate to 2^-90, which loses
// at most log2(2/|x|) < 11 bits to the cancellation.
func log1pmxDD(x dd.DD) dd.DD {
	if math.Abs(x.Hi) >= 0x1p-10 {
		return dd.Sub(log1pDD(x), x)
	}
	x2 := dd.Mul(x, x)
	r := dd.Add(dd.MulF(x2, -0.5), dd.DivF(dd.Mul(x2, x), 3))
	// tail: Σ_{k>=4} (-1)^(k+1) x^k / k; |x| < 2^-10 so 8 terms reach
	// 2^-80 of the result.
	xh := x.Hi
	t := 0.0
	for k := 11.0; k >= 4; k-- {
		t = 1/k - xh*t
	}
	return dd.AddF(r, -xh*xh*xh*xh*t)
}

// logGammaDiff returns log Γ(b) - log Γ(a+b) for b >= 10 and 0 < a <= b
// as a double-double, without forming either logarithm (they cancel for
// b >> a). With x = a/b, L = log1p(x) and Stirling's series S:
//
//	log Γ(b) - log Γ(b+a) = -a log b - b (L - x) - (a - 1/2) L + S(b) - S(a+b)
//
// (from DLMF 5.11.1); every term is well conditioned.
func logGammaDiff(a, b float64) dd.DD {
	x := dd.Div(dd.F(a), dd.F(b))
	L := log1pDD(x)
	t := dd.Neg(dd.MulF(dd.Log(b), a))
	t = dd.Sub(t, dd.MulF(log1pmxDD(x), b))
	t = dd.Sub(t, dd.Mul(sumDD(a, -0.5), L))
	s := sumDD(a, b)
	return dd.AddF(t, stirlingSeries(b)-stirlingSeries(s.Hi))
}

// logBetaPos returns log B(a, b) for 0 < a <= b (finite) as a double-double.
func logBetaPos(a, b float64) dd.DD {
	if b >= 10 {
		return dd.Add(logGammaPos(a), logGammaDiff(a, b))
	}
	// Both below 10: log Γ(a) + log Γ(b) - log Γ(a+b), with a+b = s + e
	// exactly and log Γ(s+e) = log Γ(s) + ψ(s) e to first order.
	s := sumDD(a, b)
	lg := dd.Add(logGammaPos(a), logGammaPos(b))
	ls := dd.AddF(logGammaPos(s.Hi), Digamma(s.Hi)*s.Lo)
	return dd.Sub(lg, ls)
}

// LogBeta returns log|B(a, b)|, the logarithm of the absolute value of the
// beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
//
// Special values: NaN if either argument is NaN; +Inf when a or b is 0 or a
// negative integer (and a+b is not); -Inf when B(a, b) = 0 (a+b a
// non-positive integer, a and b not).
//
// Algorithm: for positive arguments with max(a, b) >= 10, log Γ(b) -
// log Γ(a+b) is evaluated as a single cancellation-free expression from
// Stirling's series (logGammaDiff) and added to log Γ(a) in double-double;
// otherwise the three log-gammas are combined in double-double with the
// rounding of a+b corrected to first order. Negative arguments go through
// LogGamma with the same correction.
func LogBeta(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	if a > b {
		a, b = b, a
	}
	if a > 0 {
		if b >= 10 && b <= 1e300 {
			h, l := logBetaHL(a, b)
			return h + l
		}
		if math.IsInf(b, 1) {
			return math.Inf(-1)
		}
		return logBetaPos(a, b).Float()
	}
	return logBetaGeneral(a, b)
}

// logBetaGeneral handles a <= 0 or b <= 0 (a <= b).
func logBetaGeneral(a, b float64) float64 {
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return math.NaN()
	}
	s := sumDD(a, b)
	pa, pb := a == math.Trunc(a) && a <= 0, b == math.Trunc(b) && b <= 0
	ps := s.Lo == 0 && s.Hi == math.Trunc(s.Hi) && s.Hi <= 0
	switch {
	case pa || pb:
		if ps && pa && pb {
			return math.NaN()
		}
		return math.Inf(1)
	case ps:
		return math.Inf(-1)
	}
	return logBetaNegDD(a, b, s).Float()
}

// logBetaNegDD returns log|B(a,b)| in double-double for arguments with no
// pole involved (s = a+b exactly).
func logBetaNegDD(a, b float64, s dd.DD) dd.DD {
	l := dd.Add(logGammaAbsDD(a), logGammaAbsDD(b))
	ls := dd.AddF(logGammaAbsDD(s.Hi), Digamma(s.Hi)*s.Lo)
	return dd.Sub(l, ls)
}

// Beta returns the beta function B(a, b) = Γ(a)Γ(b)/Γ(a+b).
//
// Special values: NaN if either argument is NaN; +Inf when a or b is 0 or a
// negative integer and a+b is not a non-positive integer; 0 when a+b is a
// non-positive integer and neither a nor b is; overflow to ±Inf and
// underflow to 0 as the true value dictates.
//
// Algorithm: for positive arguments below 10 the three gamma values come
// from Gamma's double-double core and the rounding of a+b is corrected to
// first order (Γ(s+e) = Γ(s)(1 + ψ(s)e)); otherwise B = exp(log B) with
// log B from LogBeta's cancellation-free double-double evaluation.
func Beta(a, b float64) float64 {
	if math.IsNaN(a) || math.IsNaN(b) {
		return math.NaN()
	}
	if a > b {
		a, b = b, a
	}
	if a > 0 {
		if b < 10 && a > 0x1p-56 {
			return betaSmall(a, b)
		}
		if b >= 10 && a < 10 && a > 0x1p-56 && b <= 1e300 {
			// Γ(a) exp(log Γ(b) - log Γ(a+b))
			gh, gl := gammaHL(a)
			dh, dl := logGammaDiffHL(a, b, a+b)
			hn := dh + dl
			eh, el, k := expHL(hn, dl-(hn-dh))
			if k >= -1021 && k <= 1022 {
				ph := float64(gh * eh)
				pl := math.FMA(gh, eh, -ph) + math.FMA(gh, el, gl*eh)
				return (ph + pl) * pow2(k)
			}
		}
		if b >= 10 && b <= 1e300 {
			h, l := logBetaHL(a, b)
			hn := h + l
			eh, el, k := expHL(hn, l-(hn-h))
			if k >= -1021 && k <= 1022 {
				return (eh + el) * pow2(k)
			}
		}
		if math.IsInf(b, 1) {
			return 0
		}
		return dd.Exp(logBetaPos(a, b))
	}
	l := logBetaGeneral(a, b)
	if math.IsNaN(l) || math.IsInf(l, 0) {
		if math.IsInf(l, -1) {
			return 0
		}
		return l
	}
	s := sumDD(a, b)
	sign := GammaSign(a) * GammaSign(b) * GammaSign(s.Hi)
	return sign * dd.Exp(logBetaNegDD(a, b, s))
}

// Float64 fast paths (kernels.go): the same decompositions as logBetaPos
// and Beta's double-double path, with the logarithms and exponentials from
// the float64 kernels, products formed exactly up to their low-order terms
// and the sums of leading parts by TwoSum.

// gammaHL returns Γ(x) as hi + lo for 0 < x <= 171 (x not tiny: x >= 2^-56).
func gammaHL(x float64) (hi, lo float64) {
	switch {
	case x >= 10:
		h, l := logGammaLargeHL(x)
		hn := h + l
		eh, el, k := expHL(hn, l-(hn-h))
		s := pow2(k) // k <= 1022 for x <= 171
		return eh * s, el * s
	case x >= 1:
		return gammaMid(x)
	}
	// Γ(1+x)/x
	gh, gl := gamma1pHL(x)
	q := gh / x
	return q, (math.FMA(-q, x, gh) + gl) / x
}

// betaSmall returns B(a, b) for 2^-56 <= a <= b < 10: Γ(a)Γ(b)/Γ(a+b)
// with the three values as hi + lo, the rounding of a+b = s + sl corrected
// to first order (Γ(s+sl) = Γ(s)(1 + ψ(s) sl)), one rounding.
func betaSmall(a, b float64) float64 {
	s := a + b
	bb := s - a
	sl := (a - (s - bb)) + (b - bb)
	ah, al := gammaHL(a)
	bh, bl := gammaHL(b)
	ch, cl := gammaHL(s)
	if sl != 0 {
		cl += ch * Digamma(s) * sl
	}
	nh := float64(ah * bh)
	nl := math.FMA(ah, bh, -nh) + math.FMA(ah, bl, al*bh)
	q := nh / ch
	r := math.FMA(-q, ch, nh) + (nl - q*cl)
	return q + r/ch
}

// logBetaHL returns log B(a, b) as hi + lo for 0 < a <= b, 10 <= b and
// a + b finite. For a >= 10 and b <= 64a the three log-gammas are combined
// directly (each about 2^-68 x absolute; the cancellation is at most a
// factor b/a); otherwise log Γ(a) + [log Γ(b) - log Γ(a+b)] with the
// bracket from logGammaDiffHL, which does not cancel.
func logBetaHL(a, b float64) (hi, lo float64) {
	s := a + b
	bb := s - a
	sl := (a - (s - bb)) + (b - bb)
	if a >= 10 && b <= 64*a {
		return logBetaDirect(a, b, s, sl)
	}
	var gh, gl float64
	if a >= 10 {
		gh, gl = logGammaLargeHL(a)
	} else {
		g := logGammaPos(a)
		gh, gl = g.Hi, g.Lo
	}
	dh, dl := logGammaDiffHL(a, b, s)
	v := gh + dh // TwoSum
	bb = v - gh
	ve := (gh - (v - bb)) + (dh - bb)
	return v, ve + (gl + dl)
}

// logBetaDirect returns log B(a, b) for 10 <= a <= b <= 64a, s + sl = a + b
// exactly: log Γ(a) + log Γ(b) - log Γ(s + sl) from logGammaLargeHL2 (its
// logarithm and Stirling correction written out, error about 2^-66 x
// absolute), the leading parts summed by Fast2Sum, below 2^-62 of |log B|
// in this range (the cancellation is at most a factor b/a).
func logBetaDirect(a, b, s, sl float64) (hi, lo float64) {
	ah, al := logGammaLargeHL2(a, 0)
	bh, bl := logGammaLargeHL2(b, 0)
	ch, cl := logGammaLargeHL2(s, sl)
	u := bh + ah // Fast2Sum: log Γ is increasing here and b >= a
	ue := ah - (u - bh)
	v := u - ch // Fast2Sum: log B < 0 here, so log Γ(a+b) > u
	ve := u - (v + ch)
	return v, (ve + ue) + (al + bl - cl)
}

// logGammaDiffHL returns log Γ(b) - log Γ(a+b) as hi + lo for 0 < a <= b,
// b >= 10, s = RN(a+b), by logGammaDiff's cancellation-free expression
//
//	-a log b - b (L - x) - (a - 1/2) L + S(b) - S(a+b),  x = a/b, L = log1p(x)
//
// with L - x from its series below x = 1/64 and S(b) - S(a+b) as
// d Σ c_k T_{2k-1}, d = 1/b - 1/(a+b) = a/(b(a+b)) and
// T_n = (b^-n - (a+b)^-n)/d = Σ_i b^-i (a+b)^-(n-1-i) (the differences of
// powers factored, so that a small a costs no cancellation).
func logGammaDiffHL(a, b, s float64) (hi, lo float64) {
	xh := a / b
	xl := math.FMA(-xh, b, a) / b
	var lh, ll, mh, ml float64 // L = log1p(x), M = L - x
	if xh < 1.0/64 {
		// M = -x²/2 + x³(1/3 - x/4 + ...), to x^12/12 (next below 2^-62 M)
		x2 := float64(xh * xh)
		x2l := math.FMA(xh, xh, -x2) + 2*xh*xl
		x4 := x2 * x2
		p := math.FMA(math.FMA(-1.0/6, xh, 1.0/5), x2, math.FMA(-1.0/4, xh, 1.0/3))
		q := math.FMA(math.FMA(-1.0/10, xh, 1.0/9), x2, math.FMA(-1.0/8, xh, 1.0/7))
		r := math.FMA(math.FMA(-1.0/12, xh, 1.0/11), x4, q)
		t := math.FMA(r, x4, p) * xh // x/3 - x²/4 + ... - x^10/12
		mh = -0.5 * x2
		ml = math.FMA(x2, t, -0.5*x2l)
		lh = xh + mh // Fast2Sum: |M| < x/100
		ll = ((xh - lh) + mh) + (xl + ml)
	} else {
		wh := 1 + xh // Fast2Sum: x <= 1
		wl := ((1 - wh) + xh) + xl
		var l2 float64
		lh, l2 = logHL(wh)
		ll = l2 + wl/wh
		mh = lh - xh // exact: x/2 <= L <= x
		ml = ll - xl
	}
	bh, bl := logHL(b)
	t1h := float64(a * bh) // a log b
	t1l := math.FMA(a, bh, -t1h) + a*bl
	t2h := float64(b * mh) // b M
	t2l := math.FMA(b, mh, -t2h) + b*ml
	ah := a - 0.5 // (a - 1/2) L
	bb := ah - a
	al := (a - (ah - bb)) + (-0.5 - bb)
	t3h := float64(ah * lh)
	t3l := math.FMA(ah, lh, -t3h) + math.FMA(ah, ll, al*lh)
	// S(b) - S(a+b)
	ib := 1 / b
	is := 1 / s
	d := a * ib * is
	ib2 := ib * ib
	sg := ib + is
	is2 := is * is
	tn, pw := 1.0, is
	sum := 0.0
	for _, c := range stirlingCoef {
		sum = math.FMA(c, tn, sum)
		tn = math.FMA(ib2, tn, pw*sg)
		pw *= is2
	}
	ds := d * sum
	// -t1 - t3 - t2 + ds
	u := -t1h - t3h // TwoSum
	bb = u + t1h
	ue := (-t1h - (u - bb)) + (-t3h - bb)
	v := u - t2h // TwoSum
	bb = v - u
	ve := (u - (v - bb)) + (-t2h - bb)
	return v, (ue + ve) + (ds - t1l - t2l - t3l)
}

// stirlingCoef[k-1] = B_2k/(2k(2k-1)), k = 1..9 (stirlingSeries).
var stirlingCoef = [...]float64{1.0 / 12, -1.0 / 360, 1.0 / 1260, -1.0 / 1680, 1.0 / 1188,
	-691.0 / 360360, 1.0 / 156, -3617.0 / 122400, 43867.0 / 244188}
