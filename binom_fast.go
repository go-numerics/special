package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Certified float64 fast path of Binom for n+1 > 0, n-k+1 > 0, k > -1 (the
// beta-function branch), the strategy of gamma_fast.go: the reference
// (binomBetaDD) evaluates
//
//	exp(-(log(n+1) + log B(a, b) + ψ-corrections))
//
// with a = n-k+1, b = k+1 rounded, in double-double. Here the same sum is
// formed as hi + lo from the float64 kernels, where it is cheap to do so:
//
//   - log(n+1) from the log kernel (np1.Hi) plus np1.Lo/np1.Hi;
//   - log B(p, q), p = min(a, b), q = max(a, b), for 1/2 <= p < 2 and
//     10 <= q <= 2^40: log Γ(p) + [log Γ(q) - log Γ(p+q)], the first term
//     from the fitted core of Γ(1+t) (gamma1pFast, whose polynomial value
//     is the one the reference's gamma1pm1 uses, at the same t) and the log
//     kernel, the bracket from logGammaDiffHL;
//   - the first-order corrections for the rounding of a and b, float64
//     values formed exactly as the reference forms them;
//   - the exponential from the exp kernel.
//
// Every other (p, q) is the reference's. The result is returned when
// zivRound certifies that every real within binomErr·|h| of h + l rounds to
// h: the reference's value D is within that distance of h + l (the
// sweeps of binom_fast_test.go measure |h + l - D|/D below 2^-64.5; the
// bound allows for the sum of the kernels' stated errors, 2^-68 each, times
// the multipliers a, p and the log's value, with a margin), so the
// reference's rounding of D is h. Otherwise the reference runs: the output
// is bit for bit the reference's.
const binomErr = 0x1p-58

// logGammaUnitHL returns log Γ(x) as hi + lo for 1/2 <= x < 2, from the
// same Γ(1+t) core as logGammaPos: t = x-1 below 3/2 (gamma1pm1(x-1)),
// otherwise Γ(x) = (x-1) Γ(1+t) with t = x-2 (the reference's
// log((1+t)Γ(1+t))), the two logarithms summed by TwoSum.
func logGammaUnitHL(x float64) (hi, lo float64) {
	if x < 1.5 {
		gh, gl := gamma1pFast(x - 1)
		lh, ll := logHLn(gh)
		return lh, ll + gl/gh
	}
	gh, gl := gamma1pFast(x - 2)
	ah, al := logHLn(gh)
	bh, bl := logHLn(x - 1)
	s := ah + bh // TwoSum
	bb := s - ah
	se := (ah - (s - bb)) + (bh - bb)
	return s, se + (al + bl + gl/gh)
}

// logGammaMidHL is logGammaUnitHL for 1/2 <= x < 10: (x-1)Γ(x-1) below 5/2
// as there, and above it the reference's product Γ(z) Π (x-j), z = x-n in
// (1, 2], n = ceil(x-2), with Γ(z) from the same core and every x-j exact,
// carried as hi + lo.
func logGammaMidHL(x float64) (hi, lo float64) {
	if x < 2.5 {
		return logGammaUnitHL(x)
	}
	n := math.Ceil(x - 2)
	gh, gl := gamma1pFast(x - n - 1)
	for j := 1.0; j <= n; j++ {
		f := x - j
		p := float64(gh * f)
		gl = math.FMA(gh, f, -p) + float64(gl*f)
		gh = p
	}
	lh, ll := logHLn(gh)
	return lh, ll + gl/gh
}

// binomFastHL returns hi, lo, k with the unevaluated sum
// (hi + lo)·2^k ≈ Binom(n, k) for the branch above, and whether the
// arguments are in the region it covers.
func binomFastHL(np1, nk1 dd.DD, k float64) (hi, lo float64, ke int, ok bool) {
	k1 := sumDD(k, 1)
	a, b := nk1.Float(), k1.Float()
	p, q := math.Min(a, b), math.Max(a, b)
	small := q < 10 && p+q < 10
	if !(p >= 0.5 && (small || (p < 2 && q >= 10 && q <= 0x1p40))) {
		return 0, 0, 0, false
	}
	da, db := dd.AddF(nk1, -a).Float(), dd.AddF(k1, -b).Float()
	s := p + q
	ps := Digamma(s)
	corr := float64(da*(Digamma(a)-ps)) + float64(db*(Digamma(b)-ps))
	var v, ve, gl, dl float64
	var bb float64
	if small {
		// log B = log Γ(p) + log Γ(q) - [log Γ(s) + ψ(s)·sl], s + sl = p + q
		sl := p - (s - q) // Fast2Sum remainder (p <= q), as sumDD's low part
		ph, pl := logGammaMidHL(p)
		qh, ql := logGammaMidHL(q)
		sh, sg := logGammaMidHL(s)
		u := ph + qh // TwoSum
		bb = u - ph
		ue := (ph - (u - bb)) + (qh - bb)
		v = u - sh // TwoSum
		bb = v - u
		ve = (u - (v - bb)) + (-sh - bb) + ue
		gl = (pl + ql) - (sg + float64(ps*sl))
	} else {
		// log B(p, q) = log Γ(p) + [log Γ(q) - log Γ(p+q)]
		var gh, dh float64
		gh, gl = logGammaUnitHL(p)
		dh, dl = logGammaDiffRef(p, q, s)
		v = gh + dh // TwoSum
		bb = v - gh
		ve = (gh - (v - bb)) + (dh - bb)
	}
	// + log(n+1)
	nh, nl := logHL(np1.Hi)
	w := v + nh // TwoSum
	bb = w - v
	we := (v - (w - bb)) + (nh - bb)
	t := (ve + we) + ((gl + dl) + (nl + np1.Lo/np1.Hi) + corr)
	xh := w + t // TwoSum
	bb = xh - w
	xl := (w - (xh - bb)) + (t - bb)
	if !(math.Abs(xh) <= 700) {
		return 0, 0, 0, false
	}
	hi, lo, ke = expHL(-xh, -xl)
	return hi, lo, ke, true
}

// binomFast returns Binom(n, k) of the beta branch and whether it is
// certified equal to binomBetaDD.
func binomFast(np1, nk1 dd.DD, k float64) (float64, bool) {
	h, l, ke, ok := binomFastHL(np1, nk1, k)
	if !ok {
		return 0, false
	}
	s := pow2(ke) // |ke| <= 1010
	h = float64(h * s)
	l = float64(l * s)
	if !zivRound(h, l, float64(binomErr*h)) {
		return 0, false
	}
	return h, true
}

// logGammaDiffRef is logGammaDiffHL with the Stirling difference S(b) - S(s)
// taken as the reference's float64 value stirlingSeries(b) -
// stirlingSeries(s) (logGammaDiff rounds it so: a term of 2^-60 absolute
// error near b = 10 that the fast path has to reproduce, not improve on).
// For 0 < a, b >= 10, a < 2.
func logGammaDiffRef(a, b, s float64) (hi, lo float64) {
	xh := a / b
	xl := math.FMA(-xh, b, a) / b
	var lh, ll, mh, ml float64
	if a/b < 1.0/64 {
		x2 := float64(xh * xh)
		x2l := math.FMA(xh, xh, -x2) + float64(2*xh*xl)
		x4 := float64(x2 * x2)
		p := math.FMA(math.FMA(-1.0/6, xh, 1.0/5), x2, math.FMA(-1.0/4, xh, 1.0/3))
		q := math.FMA(math.FMA(-1.0/10, xh, 1.0/9), x2, math.FMA(-1.0/8, xh, 1.0/7))
		r := math.FMA(math.FMA(-1.0/12, xh, 1.0/11), x4, q)
		t := float64(math.FMA(r, x4, p) * xh)
		mh = float64(-0.5 * x2)
		ml = math.FMA(x2, t, -0.5*x2l)
		lh = xh + mh
		ll = ((xh - lh) + mh) + (xl + ml)
	} else {
		wh := 1 + xh
		wl := ((1 - wh) + xh) + xl
		var l2 float64
		lh, l2 = logHLn(wh)
		ll = l2 + wl/wh
		mh = lh - xh
		ml = ll - xl
	}
	bh, bl := logHLn(b)
	t1h := float64(a * bh) // a log b
	t1l := math.FMA(a, bh, -t1h) + float64(a*bl)
	t2h := float64(b * mh) // b M
	t2l := math.FMA(b, mh, -t2h) + float64(b*ml)
	ah := a - 0.5 // (a - 1/2) L
	bb := ah - a
	al := (a - (ah - bb)) + (-0.5 - bb)
	t3h := float64(ah * lh)
	t3l := math.FMA(ah, lh, -t3h) + math.FMA(ah, ll, al*lh)
	ds := stirlingSeries(b) - stirlingSeries(s)
	u := -t1h - t3h // TwoSum
	bb = u + t1h
	ue := (-t1h - (u - bb)) + (-t3h - bb)
	v := u - t2h // TwoSum
	bb = v - u
	ve := (u - (v - bb)) + (-t2h - bb)
	return v, (ue + ve) + (ds - t1l - t2l - t3l)
}
