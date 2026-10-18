package special

import "math"

// The inverse functions are evaluated directly, without a guess-and-refine
// iteration, from two fitted approximations (tools/genfit):
//
//   - erfinv(p) = p R(p²) for |p| <= 1/2, R fitted in t = p² (the form of
//     the central rational of Boost.Math's erf_inv_imp and of Giles,
//     "Approximating the erfinv function", GPU Computing Gems, 2011);
//   - erfcinv(q) = X(w), w = sqrt(-log q), for q < 1/2: X is smooth and
//     close to the identity in w (w is the variable of Boost's tail
//     rationals), fitted on pieces up to w = 31 (q down to the smallest
//     subnormal).
//
// -log q comes from the log kernel (2^-68 absolute) and w = sqrt(-log q) is
// carried as hi + lo; both polynomials keep their constant term exact and
// take the low part of their argument to first order, so the result is
// within about half an ulp plus the rounding of q(s) scaled by |s q(s)|/x.

var (
	erfInvSmallEst = estPieces(erfInvSmallPieces[:])
	erfcInvEst     = estPieces(erfcInvPieces[:])
)

// evalLow returns the piece's polynomial at t + tl (|tl| <= ulp(t)) as
// hi + lo: c0 + s c1 + s² Q(s), s = t - c, with c0 and c1 double-doubles
// and s c1 formed exactly, so that the rounding of Q(s) is scaled down by
// s²|Q(s)| relative to the result (below 2^-58 on every piece used here);
// tl enters through the derivative c1 + 2 s q[1] (its next term, 3 s² q[2]
// tl, is below 2^-60 of the result).
func (e *estPiece) evalLow(t, tl float64) (hi, lo float64) {
	s := t - e.c
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
	p = math.FMA(math.FMA(math.FMA(p, s, q[3]), s, q[2]), s, q[1]) // Q(s)
	r1 := float64(s * q[0])                                        // float64(): not fused into hi
	r1l := math.FMA(s, q[0], -r1) + math.FMA(s, e.c1lo, tl*math.FMA(2*s, q[1], q[0]))
	hi = e.c0 + r1 // Fast2Sum: |s c1| < |c0| on every piece
	lo = (r1 - (hi - e.c0)) + (r1l + math.FMA(s2, p, e.c0lo))
	return hi, lo
}

// erfInvSmallHL returns erfinv(p) as hi + lo for 0 <= p <= 1/2.
func erfInvSmallHL(p float64) (hi, lo float64) {
	t := float64(p * p) // float64(): not fused below
	rh, rl := erfInvSmallEst[min(int(t*8), 1)].evalLow(t, math.FMA(p, p, -t))
	hi = float64(p * rh)
	return hi, math.FMA(p, rh, -hi) + p*rl
}

// erfcInvHL returns erfcinv(q) as hi + lo for 0 < q < 1/2.
func erfcInvHL(q float64) (hi, lo float64) {
	lh, ll := logHL(q) // q may be subnormal
	return erfcInvLog(lh, ll)
}

// erfcInvN is erfcInvHL for a normal q.
func erfcInvN(q float64) (hi, lo float64) {
	lh, ll := logHLn(q)
	return erfcInvLog(lh, ll)
}

// erfcInvLog returns erfcinv(q) as hi + lo from log q = lh + ll (q < 1/2):
// w = sqrt(-log q) carried as hi + lo and X(w) from its piece, located
// without a branch (erfcInvIdx: per unit of w - 1/16, the segment's start,
// inverse width and first piece; the segments are [0.8125, 3.0625) by
// 1/4, [3.0625, 7.0625) by 1, [7.0625, 11.0625) by 2, then by 4).
func erfcInvLog(lh, ll float64) (hi, lo float64) {
	s := lh + ll // log q = s + e (Fast2Sum)
	e := ll - (s - lh)
	wh := math.Sqrt(-s)
	wl := (math.FMA(-wh, wh, -s) - e) / (2 * wh) // w = sqrt(-log q) = wh + wl
	u := wh - 0.0625                             // exact
	t := &erfcInvIdx[int(u)]
	return erfcInvEst[int(t[2])+int((u-t[0])*t[1])].evalLow(wh, wl)
}

// erfcInvIdx[f] = {segment start in u = w - 1/16, pieces per unit, first
// piece of the segment} for f = floor(u) (w < 27.3 for every q > 0).
var erfcInvIdx = [31][3]float64{
	{0.75, 4, 0}, {0.75, 4, 0}, {0.75, 4, 0},
	{3, 1, 9}, {3, 1, 9}, {3, 1, 9}, {3, 1, 9},
	{7, 0.5, 13}, {7, 0.5, 13}, {7, 0.5, 13}, {7, 0.5, 13},
	{11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15},
	{11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15},
	{11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15},
	{11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15}, {11, 0.25, 15},
}

// ErfInv returns the inverse error function: ErfInv(Erf(x)) = x.
//
// Special values: ErfInv(±1) = ±Inf, ErfInv(±0) = ±0; NaN for |y| > 1 or
// NaN.
//
// Algorithm: for |y| <= 1/2, y R(y²) with R fitted to mpmath; beyond,
// erfcinv(1 - |y|) (1 - |y| exact) as X(sqrt(-log(1 - |y|))) with X fitted
// to mpmath on pieces, the logarithm and the square root carried beyond
// float64; about half an ulp.
func ErfInv(y float64) float64 {
	p := math.Abs(y)
	switch {
	case math.IsNaN(y) || p > 1:
		return math.NaN()
	case p == 1:
		return math.Copysign(math.Inf(1), y)
	case y == 0:
		return y
	case p <= 0.5:
		h, l := erfInvSmallHL(p)
		return math.Copysign(h+l, y)
	}
	h, l := erfcInvN(1 - p) // 1-p exact (and normal) for 1/2 <= p < 1
	return math.Copysign(h+l, y)
}

// ErfcInv returns the inverse complementary error function:
// ErfcInv(Erfc(x)) = x, for 0 <= q <= 2.
//
// Special values: ErfcInv(0) = +Inf, ErfcInv(2) = -Inf, ErfcInv(1) = 0;
// NaN outside [0, 2] or for NaN.
//
// Algorithm: as ErfInv, with X evaluated at sqrt(-log q) directly, so that
// small q (down to the smallest subnormal) keep full relative accuracy.
func ErfcInv(q float64) float64 {
	switch {
	case math.IsNaN(q) || q < 0 || q > 2:
		return math.NaN()
	case q == 0:
		return math.Inf(1)
	case q == 2:
		return math.Inf(-1)
	case q > 1:
		return -ErfcInv(2 - q) // exact
	case q >= 0.5:
		h, l := erfInvSmallHL(1 - q) // 1 - q exact
		return h + l
	}
	h, l := erfcInvHL(q)
	return h + l
}

// Ndtri returns the inverse of the standard normal cdf: Ndtri(Ndtr(x)) = x,
// for 0 <= p <= 1.
//
// Special values: Ndtri(0) = -Inf, Ndtri(1) = +Inf, Ndtri(1/2) = 0; NaN
// outside [0, 1] or for NaN.
//
// Algorithm: -sqrt(2) erfcinv(2p) with erfcinv from ErfcInv's fitted
// approximations as hi + lo and sqrt(2) as a double-double, multiplied
// before the single rounding (so p in the far lower tail, down to
// subnormals, keeps full relative accuracy). For p > 1/2,
// Ndtri(p) = -Ndtri(1-p) with 1-p exact.
func Ndtri(p float64) float64 {
	var h, l float64
	switch {
	case p > 0.25 && p < 0.75:
		// sqrt(2) erfinv(2p - 1), 2p - 1 exact here
		t := 2*p - 1
		h, l = erfInvSmallHL(math.Abs(t))
		if t < 0 {
			h, l = -h, -l
		}
	case p > 0 && p <= 0.25:
		h, l = erfcInvHL(2 * p) // 2p exact (or an exact subnormal doubling)
		h, l = -h, -l
	case p >= 0.75 && p < 1:
		h, l = erfcInvN(2 * (1 - p)) // exact, normal
	case p == 0:
		return math.Inf(-1)
	case p == 1:
		return math.Inf(1)
	default:
		return math.NaN()
	}
	// sqrt(2) (h + l), one rounding
	v := float64(math.Sqrt2 * h)
	return v + (math.FMA(math.Sqrt2, h, -v) + math.FMA(math.Sqrt2, l, sqrt2Lo*h))
}

// sqrt2Lo = sqrt(2) - math.Sqrt2 (mpmath).
const sqrt2Lo = -9.667293313452913e-17
