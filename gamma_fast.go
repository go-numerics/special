package special

import "math"

// Certified float64 fast paths for Gamma on 0 < x < 1 and x < 0.
//
// There Gamma's reference evaluation is double-double (gammaPos,
// gammaNegDD). Each fast path below evaluates the same formula in float64
// arithmetic as an unevaluated sum h + l, reusing bit for bit every
// intermediate value that the reference rounds to float64 (the fitted
// polynomial of Γ(1+t), sin(πx)), and has an a-priori bound e on the distance
// between h + l and the reference's own double-double value D. When every
// real within e of h + l rounds to h (zivRound), the reference's result,
// which is the rounding of D, is h; otherwise the reference runs. The output
// of Gamma is therefore the same float64 as before, bit for bit (Ziv's
// strategy, with the reference as the accurate step). The derivation of
// the bounds is in docs/gamma-fastpath.md; the tests check fast == reference
// on more than 10^7 points.

// Relative bounds on |h + l - D| / |h| (docs/gamma-fastpath.md).
const (
	gammaErrRecip = 0x1p-99               // 2^-56 <= x < 1, -1/2 < x <= -2^-56 (derived: 42u²)
	gammaErrProd  = 0x1p-94               // -34 < x <= -1/2 (derived: 1087u²)
	gammaErrRefl  = 0x1.6a09e667f3bcdp-59 // -190 <= x <= -34: 2^-58.5 (derived: 2^-58.91)
)

// zivRound reports whether every real within e (>= 0) of h + l rounds to
// h under round-to-nearest-even: whether h + l + [-e, e] lies strictly
// inside the rounding cell of h, which extends ulp(h)/2 on either side, or
// ulp(h)/4 on the side towards zero when |h| is a power of two. The two
// comparisons are floating-point sums against representable bounds, which
// rounding cannot carry across them (rounding is monotonic). It refuses h
// subnormal or within 2^54 of it, infinite or NaN.
func zivRound(h, l, e float64) bool {
	b := math.Float64bits(h)
	ex := b >> 52 & 0x7ff
	if ex <= 54 || ex == 0x7ff {
		return false
	}
	if b>>63 != 0 {
		l = -l // work on |h|
	}
	half := math.Float64frombits((ex - 53) << 52) // ulp(h)/2
	low := half
	if b&(1<<52-1) == 0 {
		low = half / 2
	}
	return l+e < half && l-e > -low
}

// gamma1pFast returns Γ(1+t) = 1 + t(t-1)F(t) for t in [-1/2, 1] as
// hi + lo, with F = evalPiecesDD(gamma1pPieces, t), the very value gamma1p
// uses, and the rest formed with exact products (FMA) and one Fast2Sum:
// relative error below 8.3u² with respect to 1 + t(t-1)F.
func gamma1pFast(t float64) (hi, lo float64) {
	f := evalPiecesDD(gamma1pPieces[:], t)
	// t - 1 = uh + ul exactly (TwoSum, as gamma1pm1)
	uh := t - 1
	bb := uh - t
	ul := (t - (uh - bb)) + (-1 - bb)
	ph := float64(uh * t) // float64(): not fused below
	pl := math.FMA(uh, t, -ph) + ul*t
	gh := float64(ph * f.Hi)
	gl := math.FMA(ph, f.Hi, -gh) + math.FMA(ph, f.Lo, pl*f.Hi)
	hi = 1 + gh // Fast2Sum: |t(t-1)F| <= 0.773 on [-1/2, 1]
	lo = ((1 - hi) + gh) + gl
	return hi, lo
}

// gammaRecipFast returns Γ(x) = Γ(1+x)/x for 2^-56 <= x < 1 and
// -1/2 < x <= -2^-56, and whether it is certified equal to
// dd.DivF(gamma1p(x), x).Float().
func gammaRecipFast(x float64) (float64, bool) {
	gh, gl := gamma1pFast(x)
	qh := gh / x
	r := math.FMA(-qh, x, gh) // exact remainder
	ql := (r + gl) / x
	h := qh + ql
	l := ql - (h - qh)
	return h, zivRound(h, l, gammaErrRecip*math.Abs(h))
}

// gammaProdFast returns Γ(x) = Γ(1+t) / (t Π_{k<n} (x+k)), n = ceil(-x),
// t = x + n, for -34 < x <= -1/2 non-integer, and whether it is certified
// equal to gammaNegDD(x). Every x + k is exact (gammaNegDD); the product is
// carried as hi + lo in two interleaved chains (half the latency of one)
// and normalised before the division.
func gammaProdFast(x float64) (float64, bool) {
	n := math.Ceil(-x)
	t := x + n
	ah, al := t, 0.0
	bh, bl := 1.0, 0.0
	k := 0.0
	for ; k+1 < n; k += 2 {
		f := x + k
		p := float64(ah * f)
		al = math.FMA(ah, f, -p) + al*f
		ah = p
		g := x + (k + 1)
		q := float64(bh * g)
		bl = math.FMA(bh, g, -q) + bl*g
		bh = q
	}
	if k < n {
		f := x + k
		p := float64(ah * f)
		al = math.FMA(ah, f, -p) + al*f
		ah = p
	}
	dh := float64(ah * bh)
	dl := math.FMA(ah, bh, -dh) + (ah*bl + al*bh)
	s := dh + dl // Fast2Sum
	dl -= s - dh
	dh = s
	gh, gl := gamma1pFast(t)
	qh := gh / dh
	r := math.FMA(-qh, dh, gh) // exact remainder
	ql := (r + gl - qh*dl) / dh
	h := qh + ql
	l := ql - (h - qh)
	return h, zivRound(h, l, gammaErrProd*math.Abs(h))
}

// gammaReflFast returns Γ(x) = -π / (x sin(πx) Γ(-x)) for -190 <= x <= -34
// non-integer, and whether it is certified equal to gammaNegDD(x):
// log|Γ(x)| = log π - log|x sin πx| - log Γ(-x) from the float64 log kernel
// and logGammaLargeHL, then the exp kernel. sin(πx) is sinPiDD(x), the
// reference's own value. Subnormal results go through math.Ldexp as in the
// reference, results below 2^-1076 are its zero, and the strip
// e^-746 < |Γ(x)| < e^-745 (outside the exp kernel's range) is left to the
// reference.
func gammaReflFast(x float64) (float64, bool) {
	s := sinPiDD(x)
	dh := float64(x * s.Hi)
	dl := math.FMA(x, s.Hi, -dh) + x*s.Lo
	lh, ll := logHLn(math.Abs(dh)) // |x sin πx| > 2^-46: normal
	ll += dl / dh                  // log|dh + dl| = log|dh| + dl/dh + O(u²)
	gh, gl := logGammaLargeHL(-x)
	// L = log π - (lh + ll) - (gh + gl): two TwoSums for the leading parts
	a := lnPi.Hi - lh
	bb := a - lnPi.Hi
	ea := (lnPi.Hi - (a - bb)) + (-lh - bb)
	c := a - gh
	bb = c - a
	ec := (a - (c - bb)) + (-gh - bb)
	cl := (ea + ec) + ((lnPi.Lo - ll) - gl)
	h := c + cl // Fast2Sum: |cl| < 2^-9 < |c|
	l := cl - (h - c)
	switch {
	case h < -746:
		// |D| < e^(-746 + 2^-58) < 2^-1076: the reference's math.Ldexp
		// returns a zero.
		return math.Copysign(0, s.Hi), true
	case h < -745:
		return 0, false // outside the exp kernel's range
	}
	eh, el, k := expHL(h, l)
	if !zivRound(eh, el, gammaErrRefl*eh) {
		return 0, false
	}
	if k >= -1021 {
		return math.Copysign(eh*pow2(k), s.Hi), true // exact scaling
	}
	// Subnormal: the reference returns math.Ldexp(RN(M), k') with
	// RN(M) = eh·2^(k-k'), certified above; math.Ldexp depends only on the
	// normalised fraction and the total exponent, so this is its result.
	return math.Copysign(math.Ldexp(eh, k), s.Hi), true
}

// gammaUnit returns Γ(x) for 0 < x < 1 (non-integer by definition).
func gammaUnit(x float64) float64 {
	if x >= 0x1p-56 {
		if v, ok := gammaRecipFast(x); ok {
			return v
		}
	}
	return gammaPos(x).Float()
}

// gammaNeg returns Γ(x) for non-integer x < 0: the certified fast path of
// the region, else gammaNegDD.
func gammaNeg(x float64) float64 {
	var v float64
	ok := false
	switch {
	case x > -0x1p-56:
	case x > -0.5:
		v, ok = gammaRecipFast(x)
	case x > -34:
		v, ok = gammaProdFast(x)
	case x >= -190:
		v, ok = gammaReflFast(x)
	}
	if ok {
		return v
	}
	return gammaNegDD(x)
}
