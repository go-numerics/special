package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Double-double constants (mpmath, 300 bits).
var (
	twoOverSqrtPi = dd.DD{Hi: 1.1283791670955126, Lo: 1.533545961316588e-17}
	invSqrt2      = dd.DD{Hi: 0.7071067811865476, Lo: -4.833646656726457e-17}
	sqrtPiOver2   = dd.DD{Hi: 1.2533141373155003, Lo: -9.164289990229583e-17}
	sqrtPiHalf    = dd.DD{Hi: 0.886226925452758, Lo: -3.8332932499128993e-17}
	invSqrt2Pi    = dd.DD{Hi: 0.3989422804014327, Lo: -2.49232720227773e-17}
	ln2DD         = dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}
)

// erfSmallDD returns erf(x) for |x| <= 1/2 as x E(x²), E fitted to mpmath
// (zz_erfSmallPieces_coeffs.go) with its constant term in double-double.
func erfSmallDD(x float64) dd.DD {
	return dd.MulF(evalPiecesDD(erfSmallPieces[:], x*x), x)
}

// erfcxDD returns erfcx(x) = exp(x²) erfc(x) for x >= 1/2 (finite) as a
// double-double whose error is about 0.1 ulp: on [1/2, 8] a piecewise
// Chebyshev interpolant of erfcx itself; beyond, x erfcx(x) as a function
// of u = 1/x² (smooth, tending to 1/sqrt(pi)), both fitted to mpmath
// (zz_erfcxPieces_coeffs.go, zz_erfcxTailPieces_coeffs.go). Fitting erfcx
// rather than erfc removes the Gaussian factor from the approximation; it is
// reapplied exactly by the callers (the classical device of Cody, "Rational
// Chebyshev approximations for the error function", Math. Comp. 23, 1969).
func erfcxDD(x float64) dd.DD {
	if x < 8 {
		return evalPiecesDD(erfcxPieces[:], x)
	}
	g := evalPiecesDD(erfcxTailPieces[:], 1/(x*x))
	return dd.DivF(g, x)
}

// erfcxCorrected returns erfcx(z) for a double-double z with z.Hi >= 1/2,
// correcting the evaluation at z.Hi to first order through the logarithmic
// derivative erfcx'/erfcx = 2x - 2/(sqrt(pi) erfcx(x)), which tends to
// -1/x + 1/x³ for large x (taken from the asymptotic series there, since the
// difference cancels). The correction is relative and below 2^-52, so a few
// correct digits of the derivative are ample.
func erfcxCorrected(z dd.DD) dd.DD {
	e := erfcxDD(z.Hi)
	x := z.Hi
	var ld float64
	if x < 8 {
		ld = 2*x - twoOverSqrtPi.Hi/e.Hi
	} else {
		ld = -1/x + 1/(x*x*x)
	}
	return dd.AddF(e, e.Hi*z.Lo*ld)
}

// erfcMant returns m, k with erfc(z) = m * 2^k for a double-double
// z >= 1/2: exp(-z²) is formed from z² in double-double (dd.ExpMant) and
// multiplied by erfcx(z) before the single final rounding, so that the
// Gaussian factor does not amplify the rounding of z² (the error a naive
// exp(-x*x) incurs grows like 2x² ulp).
func erfcMant(z dd.DD) (dd.DD, int) {
	m, k, _ := erfcMantX(z)
	return m, k
}

// erfcMantX is erfcMant also returning erfcx(z) (the inverse functions need
// it for their Newton step).
func erfcMantX(z dd.DD) (dd.DD, int, dd.DD) {
	em, k := dd.ExpMant(dd.Neg(dd.Mul(z, z)))
	ex := erfcxCorrected(z)
	return dd.Mul(em, ex), k, ex
}

// erfcDD returns erfc(z) as a double-double for finite z.
func erfcDD(z dd.DD) dd.DD {
	switch {
	case z.Hi >= 0.5:
		m, k := erfcMant(z)
		return dd.Ldexp(m, k)
	case z.Hi <= -0.5:
		return dd.Sub(dd.F(2), erfcDD(dd.Neg(z)))
	}
	// 1 - erf(z), erf(z.Hi + z.Lo) = erf(z.Hi) + erf'(z.Hi) z.Lo
	e := erfSmallDD(z.Hi)
	e = dd.AddF(e, z.Lo*twoOverSqrtPi.Hi*math.Exp(-z.Hi*z.Hi))
	return dd.Sub(dd.F(1), e)
}

// Float64 fast paths (kernels.go, fastpoly.go): the same approximations as
// the double-double helpers above — the fitted erfcx and E, exp(-z²) from
// z² carried exactly — with the exponential and the products formed by the
// float64 kernels as hi + lo and one final rounding. The double-double
// helpers remain for the callers that need their extra digits (the
// incomplete gamma and beta functions) and for the subnormal tails.

var (
	logErfcxEst     = estPieces(logErfcxPieces[:])
	logErfcxTailEst = estPieces(logErfcxTailPieces[:])
	erfcxEst        = estPieces(erfcxPieces[:])
	erfcxTailEst    = estPieces(erfcxTailPieces[:])
	erfSmallEst     = estPieces(erfSmallPieces[:])
)

// erfcxHL returns erfcx(x) as hi + lo for 0.5 <= x < 1e300 (error about
// 2^-55 relative, the rounding of s·q(s) in the fitted pieces): on [0.5, 8)
// the piece of erfcxPieces; beyond, G(u)/x with G = x·erfcx(x) in
// u = 1/x² (the rounding of u moves the evaluation point by 2^-52 u, below
// 2^-59 in G) and the quotient corrected by its FMA residual.
func erfcxHL(x float64) (hi, lo float64) {
	if x < 8 {
		return erfcxEst[int((x-0.5)*2)].evalLow(x, 0)
	}
	inv := 1 / x
	u := inv * inv
	gh, gl := erfcxTailEst[min(int(u*128), 1)].evalShort(u)
	y := gh * inv
	r := math.FMA(-y, x, gh) + gl
	return y, r * inv
}

// erfcHL returns hi, lo, k with erfc(zh + zl) = (hi + lo)·2^k (and
// erfcx(zh) to about 2^-53, for the inverse functions' Newton steps) for
// 0.5 <= zh <= 26.5 and |zl| <= ulp(zh): exp(-z²) from z² = sh + sl exact
// to 2^-106 (the exp kernel, 2^-68), erfcx(zh) corrected to first order in
// zl through erfcx'(x) = 2x erfcx(x) - 2/sqrt(pi), and the product formed
// exactly up to its low-order terms.
//
// This is the hot path of Erf, Erfc and Ndtr, so expHL and the erfcx piece
// evaluation are written in (a call would spill every live register under
// Go's ABI); they are the same operations as expHL and estPiece.eval.
func erfcHL(zh, zl float64) (hi, lo float64, k int, erfcx float64) {
	sh := float64(zh * zh) // float64(): not fused below
	sl := math.FMA(zh, zh, -sh) + 2*zh*zl
	// exp(-sh - sl): expHL
	kd := math.RoundToEven(-sh * expInvLn2N)
	ki := int(kd)
	r := -sh - kd*expC1
	dr := -sl - kd*expC2
	u := r + dr
	ue := dr - (u - r)
	u2 := u * u
	pa := math.FMA(u, 1.0/6, 0.5)
	pb := math.FMA(u, 1.0/120, 1.0/24)
	pb = math.FMA(u2, 1.0/720, pb)
	pu := u2 * math.FMA(u2, pb, pa)
	t := &expT[ki&127]
	th, tl := t[0], t[1]
	tu := float64(th * u)
	e0 := th + tu
	el := ((tu - (e0 - th)) + math.FMA(th, u, -tu)) + math.FMA(th, ue+pu, math.FMA(tl, u, tl))
	eh := e0 + el
	el -= eh - e0
	// erfcx(zh): estPiece.evalLow on [0.5, 8), erfcxHL beyond
	var ch, cl float64
	if zh < 8 {
		e := &erfcxEst[int((zh-0.5)*2)]
		s := zh - e.c
		q := &e.q
		s2 := s * s
		s4 := s2 * s2
		a0 := math.FMA(q[5], s, q[4])
		a1 := math.FMA(q[7], s, q[6])
		a2 := math.FMA(q[9], s, q[8])
		a3 := math.FMA(q[11], s, q[10])
		a4 := math.FMA(q[13], s, q[12])
		a5 := math.FMA(q[15], s, q[14])
		b0 := math.FMA(a1, s2, a0)
		b1 := math.FMA(a3, s2, a2)
		b2 := math.FMA(a5, s2, a4)
		p := math.FMA(b2, s4*s4, math.FMA(b1, s4, b0))
		p = math.FMA(math.FMA(math.FMA(p, s, q[3]), s, q[2]), s, q[1])
		r1 := float64(s * q[0])
		ch = e.c0 + r1
		cl = (r1 - (ch - e.c0)) + ((math.FMA(s, q[0], -r1) + s*e.c1lo) + math.FMA(s2, p, e.c0lo))
	} else {
		// erfcxHL's tail written out: G(u)/z, u = 1/z²
		inv := 1 / zh
		u := inv * inv
		e := &erfcxTailEst[min(int(u*128), 1)]
		s := u - e.c
		q := &e.q
		s2 := s * s
		a0 := math.FMA(q[4], s, q[3])
		a1 := math.FMA(q[6], s, q[5])
		a2 := math.FMA(q[8], s, q[7])
		b0 := math.FMA(a1, s2, a0)
		b1 := math.FMA(q[9], s2, a2)
		p := math.FMA(b1, s2*s2, b0)
		p = math.FMA(math.FMA(math.FMA(p, s, q[2]), s, q[1]), s, q[0])
		rr := float64(s * p) // |s p| < 2^-9 G: its rounding is negligible
		gh := e.c0 + rr
		gl := (rr - (gh - e.c0)) + e.c0lo
		ch = gh * inv
		cl = (math.FMA(-ch, zh, gh) + gl) * inv
	}
	ex := ch + cl // erfcx(zh); ch alone is not normalised
	if zl != 0 {
		cl += zl * math.FMA(2*zh, ex, -twoOverSqrtPi.Hi)
	}
	ph := float64(eh * ch)
	pl := math.FMA(eh, ch, -ph) + math.FMA(eh, cl, el*ch)
	hi = ph + pl // normalised: callers add 1 or take log1p
	return hi, pl - (hi - ph), ki >> 7, ex
}

// erfSmallHL returns erf(x) as hi + lo for |x| <= 1/2 (x E(x²); the
// product left unevaluated).
func erfSmallHL(x float64) (hi, lo float64) {
	eh, el := erfSmallEst[0].evalShort(x * x)
	hi = float64(x * eh)
	return hi, math.FMA(x, eh, -hi) + x*el
}

// Erf returns the error function erf(x) = 2/sqrt(pi) ∫₀ˣ exp(-t²) dt.
//
// Special values: erf(±0) = ±0, erf(±Inf) = ±1, erf(NaN) = NaN.
//
// Algorithm: for |x| <= 1/2, x E(x²) with E fitted to mpmath; beyond,
// 1 - erfc(|x|) in double-double with erfc from erfcMant.
func Erf(x float64) float64 {
	ax := math.Abs(x)
	switch {
	case ax <= 0.5:
		eh, el := erfSmallEst[0].evalShort(x * x)
		return math.FMA(x, eh, x*el)
	case ax < 6:
		h, l, k, _ := erfcHL(ax, 0)
		s := pow2(k)
		h *= s
		l *= s
		// 1 - erfc: Fast2Sum (erfc(1/2) < 1/2).
		v := 1 - h
		return math.Copysign(v+(((1-v)-h)-l), x)
	case math.IsNaN(x):
		return x
	}
	// erfc(6) < 2.2e-17 < ulp(1)/4: the result rounds to ±1.
	return math.Copysign(1, x)
}

// Erfc returns the complementary error function erfc(x) = 1 - erf(x).
//
// Special values: erfc(+Inf) = 0, erfc(-Inf) = 2, erfc(NaN) = NaN; erfc
// underflows to 0 for x > 27.2.
//
// Algorithm: for x >= 1/2, exp(-x²) erfcx(x) with x² in double-double and
// erfcx fitted to mpmath, multiplied before the single rounding; for
// |x| < 1/2, 1 - erf(x) in double-double; for x <= -1/2, 2 - erfc(-x).
func Erfc(x float64) float64 {
	switch {
	case x >= 0.5 && x <= 26.5:
		h, l, k, _ := erfcHL(x, 0)
		return (h + l) * pow2(k) // erfc(26.5) > 2^-1020: normal
	case x > -0.5 && x < 0.5:
		eh, el := erfSmallHL(x)
		v := 1 - eh // Fast2Sum: |erf| < 1/2
		return v + (((1 - v) - eh) - el)
	case x <= -0.5 && x > -6:
		h, l, k, _ := erfcHL(-x, 0)
		s := pow2(k)
		h *= s
		l *= s
		v := 2 - h
		return v + (((2 - v) - h) - l)
	}
	return erfcTail(x)
}

// erfcTail is Erfc for x > 26.5 (subnormal results), x <= -6 and NaN.
func erfcTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 27.3:
		return 0
	case x > 0:
		m, k := erfcMant(dd.F(x))
		return math.Ldexp(m.Float(), k)
	}
	// erfc(-6) = 2 - 2.2e-17: rounds to 2.
	return 2
}

// Erfcx returns the scaled complementary error function
// erfcx(x) = exp(x²) erfc(x).
//
// Special values: erfcx(+Inf) = 0, erfcx(-Inf) = +Inf, erfcx(NaN) = NaN;
// erfcx overflows to +Inf for x < -26.628.
//
// Algorithm: for x >= 1/2 the fitted erfcx directly (no exponential at
// all); for |x| < 1/2, exp(x²)(1 - erf(x)) in double-double; for
// x <= -1/2, 2 exp(x²) - erfcx(-x).
func Erfcx(x float64) float64 {
	switch {
	case x >= 0.5 && x < 1e300:
		h, l := erfcxHL(x)
		return h + l
	case x > -0.5 && x < 0.5:
		// exp(x²)(1 - erf(x))
		sh := float64(x * x)
		eh, el, k := expHL(sh, math.FMA(x, x, -sh))
		fh, fl := erfSmallHL(x)
		wh := 1 - fh // Fast2Sum
		wl := ((1 - wh) - fh) - fl
		ph := float64(eh * wh)
		pl := math.FMA(eh, wh, -ph) + math.FMA(eh, wl, el*wh)
		return (ph + pl) * pow2(k) // k = 0 or 1
	case x <= -0.5 && x >= -26.6:
		// 2 exp(x²) - erfcx(-x); exp(x²) < 2^1021 here.
		sh := float64(x * x)
		eh, el, k := expHL(sh, math.FMA(x, x, -sh))
		s := pow2(k + 1)
		eh *= s
		el *= s
		ch, cl := erfcxHL(-x)
		v := eh - ch // Fast2Sum: 2 exp(x²) > 2.5 > erfcx(-x)
		return v + (((eh - v) - ch) + (el - cl))
	}
	return erfcxTail(x)
}

// erfcxTail is Erfcx for x >= 1e300, x < -26.6 and NaN.
func erfcxTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 1):
		return 0
	case x > 0:
		return erfcxDD(x).Float()
	}
	return math.Inf(1)
}

// ndtrZ returns z = -x/sqrt(2) as a double-double.
func ndtrZ(x float64) dd.DD {
	return dd.MulF(invSqrt2, -x)
}

// Ndtr returns the standard normal cumulative distribution function
// Φ(x) = (1/sqrt(2π)) ∫_{-∞}^x exp(-t²/2) dt = erfc(-x/sqrt 2)/2.
//
// Special values: Φ(-Inf) = 0, Φ(+Inf) = 1, Φ(NaN) = NaN.
//
// Algorithm: the argument -x/sqrt(2) is carried in double-double into
// erfc (erfcMant), so the lower tail keeps full relative accuracy down to
// the subnormal range; the rounding of x/sqrt(2) that limits
// implementations computing erfc(-x*M_SQRT1_2) (an error growing like
// x² ulp) does not occur.
func Ndtr(x float64) float64 {
	if !(x > -37.4 && x < 8.5) {
		return ndtrTail(x)
	}
	zh, zl := ndtrZHL(x)
	switch {
	case zh >= 0.5:
		h, l, k, _ := erfcHL(zh, zl)
		return (h + l) * pow2(k-1) // Φ(-37.4) > 2^-1021: normal
	case zh <= -0.5:
		h, l, k, _ := erfcHL(-zh, -zl)
		s := pow2(k - 1)
		h *= s
		l *= s
		v := 1 - h // Fast2Sum: Φ(-x) < 1/4
		return v + (((1 - v) - h) - l)
	}
	// (1 - erf(z))/2, erf(zh + zl) = erf(zh) + zl·2/sqrt(pi)·exp(-zh²)
	eh, el := erfSmallHL(zh)
	el += zl * erfDeriv(zh)
	v := 1 - eh
	return 0.5 * (v + (((1 - v) - eh) - el))
}

// erfDeriv returns erf'(z) = 2/sqrt(pi) exp(-z²) for |z| < 1/2 to about
// 2^-12 relative (exp by 1 - z² + z⁴/2 - z⁶/6): enough for a first-order
// correction by a z.lo below 2^-53 z.
func erfDeriv(z float64) float64 {
	u := z * z
	return twoOverSqrtPi.Hi * (1 - u*(1-u*(0.5-u*(1.0/6))))
}

// ndtrZHL returns -x/sqrt(2) as hi + lo (error below 2^-104 relative).
func ndtrZHL(x float64) (hi, lo float64) {
	hi = float64(-x * invSqrt2.Hi)
	return hi, math.FMA(-x, invSqrt2.Hi, -hi) - x*invSqrt2.Lo
}

// ndtrTail is Ndtr for x <= -37.4 (subnormal results), x >= 8.5 (where
// Φ(-x) < 2^-56 and the result rounds to 1) and NaN.
func ndtrTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 0:
		return 1
	case x < -40:
		// Φ(-40) < 4e-350.
		return 0
	}
	m, k := erfcMant(ndtrZ(x))
	return math.Ldexp(m.Float(), k-1)
}

// LogNdtr returns log Φ(x), the logarithm of the standard normal cdf,
// accurate in both tails: for x -> -Inf it is -x²/2 - log(-x) - ... with no
// underflow, for large x it is -Φ(-x) with full relative accuracy.
//
// Special values: LogNdtr(-Inf) = -Inf, LogNdtr(+Inf) = 0,
// LogNdtr(NaN) = NaN.
//
// Algorithm: for x <= -1/sqrt(2), log erfcx(z) - z² - log 2 with
// z = -x/sqrt(2) in double-double; for x >= 1/sqrt(2), log1p(-Φ(-x)) with
// Φ(-x) in double-double; in between, the logarithm of Φ in double-double.
func LogNdtr(x float64) float64 {
	if !(x > -1e150 && x < 37) {
		return logNdtrTail(x)
	}
	zh, zl := ndtrZHL(x)
	switch {
	case zh >= 0.5:
		// log erfcx(z) - z² - log 2, log erfcx fitted directly (no
		// logarithm of erfcx): on [0.5, 8) as a function of z, beyond as
		// H(u) - log z, H = log(z erfcx(z)) in u = 1/z².
		var ch, cl float64
		if zh < 8 {
			var i int
			switch {
			case zh < 2:
				i = int((zh - 0.5) * 2)
			case zh < 4:
				i = 3 + int(zh-2)
			default:
				i = 5 + int((zh-4)*0.5)
			}
			ch, cl = logErfcxEst[i].evalLow(zh, zl)
		} else {
			iz := 1 / zh
			u := iz * iz
			hh, hl := logErfcxTailEst[min(int(u*128), 1)].evalShort(u)
			// log z: logHL written out (z >= 8 is normal)
			ixb := math.Float64bits(zh)
			tmp := ixb - 0x3fe8000000000000
			kf := float64(int64(tmp) >> 52)
			m := math.Float64frombits(ixb - uint64(int64(tmp)>>52)<<52)
			t := &logT[int(m*256+0.5)-192]
			invc := t[0]
			p := float64(m * invc) // float64(): not fused into r below
			pe := math.FMA(m, invc, -p)
			r := p - 1
			r2 := r * r
			q0 := math.FMA(r, 1.0/3, -0.5)
			q1 := math.FMA(r, 1.0/5, -0.25)
			q2 := math.FMA(r, 1.0/7, -1.0/6)
			q2 = math.FMA(r2, -1.0/8, q2)
			q1 = math.FMA(r2, q2, q1)
			q := r2 * math.FMA(r2, q1, q0)
			l0 := kf*ln2Hi42 + t[1]
			gh := l0 + r
			gl := (r - (gh - l0)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q))
			gl += zl * iz
			ch = hh - gh // TwoSum
			bb := ch - hh
			cl = ((hh - (ch - bb)) + (-gh - bb)) + (hl - gl)
		}
		sh := float64(zh * zh)
		sl := math.FMA(zh, zh, -sh) + 2*zh*zl
		a := ch - ln2DD.Hi // TwoSum
		bb := a - ch
		ae := (ch - (a - bb)) + (-ln2DD.Hi - bb)
		v := a - sh // TwoSum
		bb = v - a
		ve := (a - (v - bb)) + (-sh - bb)
		return v + (ve + (ae + (cl - sl - ln2DD.Lo)))
	case zh <= -0.5:
		// log1p(-Φ(-x)), Φ(-x) = erfc(-z)/2 < 1/4
		h, l, k, _ := erfcHL(-zh, -zl)
		s := pow2(k - 1)
		h *= s
		l *= s
		if h < 0x1p-20 {
			// -q - q²/2 - q³/3 - q⁴/4 (next term below 2^-80 q)
			return -h - (l + h*h*math.FMA(h, math.FMA(h, 0.25, 1.0/3), 0.5))
		}
		wh := 1 - h // Fast2Sum
		wl := ((1 - wh) - h) - l
		lh, ll := logHL(wh)
		return lh + (ll + wl/wh)
	}
	// log Φ, Φ = (1 - erf(z))/2 in (1/4, 3/4)
	eh, el := erfSmallHL(zh)
	el += zl * erfDeriv(zh)
	ph := 1 - eh
	pl := ((1 - ph) - eh) - el
	lh, ll := logHL(ph)
	a := lh - ln2DD.Hi // TwoSum
	bb := a - lh
	ae := (lh - (a - bb)) + (-ln2DD.Hi - bb)
	return a + (ae + (ll + pl/ph - ln2DD.Lo))
}

// logNdtrTail is LogNdtr for x <= -1e150, x >= 37 and NaN.
func logNdtrTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		// log Φ(x) = -Φ(-x) + ..., and Φ(-40) < 4e-350 rounds to -0.
		return math.Copysign(0, -1)
	case x > 0:
		m, k := erfcMant(dd.Neg(ndtrZ(x)))
		q := dd.Ldexp(m, k-1)
		return log1pDD(dd.Neg(q)).Float()
	}
	// log Φ(x) = -x²/2 - log(-x) - log sqrt(2π) + O(x^-2); the first
	// term alone is the correctly rounded result here.
	return -0.5 * x * x
}
