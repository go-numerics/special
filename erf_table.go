package special

import "math"

// Direct table kernels of Erf, Erfc and Ndtr (special#33): the function
// itself fitted piece by piece, with no exponential, no erfcx and no division.
//
// erfc(x) and Φ(-t) = erfc(t/√2)/2 decay like exp(-x²), so a piece of fixed
// width in x would need an ever higher degree; pieces of fixed width in x²
// (equivalently, of fixed range of the logarithmic slope 2x·dx) keep the
// degree near 14: step 1/4 in z² below z² = 16 (index int(4z²), degree <= 9)
// and step 1 beyond (index int(z²) + 48, degree about 14), z = x for erfc
// and z = t/√2 for Φ. Both rules are one expression, min(int(4z²),
// int(z²)+48): the first is the smaller below z² = 16 and the second beyond.
// The index is computed in float64, so the pieces are fitted on a range
// wider by 2^-40 (relative) on both sides (tools/genfit).

// evalTab returns the piece's polynomial at t as hi + lo, for the fast-decay
// pieces, where the linear and quadratic terms are a sizeable part of the
// value (|s q0| up to c0/2, |s² q1| up to c0/8, against 2^-3 and 2^-8 of it
// for the pieces of estPiece.eval): c0 + s q0 + s² q1 is formed exactly
// (c0, q0, q1, q2 as hi + lo, the products by FMA residuals, s² and s³ as
// hi + lo), and only s⁴ q[3..15](s), at most 0.3% of the value, is rounded,
// by Estrin's scheme. The error before the final rounding is below 2^-60
// relative.
func (e *estPiece) evalTab(t float64) (hi, lo float64) {
	s := t - e.c // exact: t and c are within a factor 2
	q := &e.q
	s2 := float64(s * s) // float64(): not fused below
	s2e := math.FMA(s, s, -s2)
	s4 := float64(s2 * s2)
	a0 := math.FMA(q[4], s, q[3])
	a1 := math.FMA(q[6], s, q[5])
	a2 := math.FMA(q[8], s, q[7])
	a3 := math.FMA(q[10], s, q[9])
	a4 := math.FMA(q[12], s, q[11])
	a5 := math.FMA(q[14], s, q[13])
	b0 := math.FMA(a1, s2, a0)
	b1 := math.FMA(a3, s2, a2)
	b2 := math.FMA(a5, s2, a4)
	c0 := math.FMA(b1, s4, b0)
	c1 := math.FMA(q[15], s4, b2)
	r := math.FMA(c1, float64(s4*s4), c0)
	t4 := float64(s4 * r) // s⁴ q[3..15](s)
	s3 := float64(s2 * s)
	s3e := math.FMA(s2, s, -s3) + float64(s2e*s)
	th := float64(s2 * q[1])
	tl := math.FMA(s2, q[1], -th) + float64(s2e*q[1])
	uh := float64(s3 * q[2])
	ul := math.FMA(s3, q[2], -uh) + float64(s3e*q[2])
	r1 := float64(s * q[0])
	h0 := e.c0 + r1 // Fast2Sum: |s q0| < |c0|
	l0 := r1 - (h0 - e.c0)
	h1 := h0 + th // Fast2Sum: |s² q1| < |h0|
	l1 := th - (h1 - h0)
	h2 := h1 + uh // Fast2Sum: |s³ q2| < |h1|
	l2 := uh - (h2 - h1)
	l := ((l0 + l1) + l2) + ((math.FMA(s, q[0], -r1) + tl) + (ul + (math.FMA(s, e.c1lo, e.c0lo) + math.FMA(s2, e.c2lo, math.FMA(s3, e.c3lo, t4)))))
	hi = h2 + l
	return hi, l - (hi - h2)
}

// gaussEst copies fitted pieces (the piece of index i is p[i-1]) into a table
// indexed by min(int(4z²), int(z²)+48); index 0 is never used.
func gaussEst(p []polyPiece) []estPiece {
	e := estPieces(p)
	return append([]estPiece{e[0]}, e...)
}

var (
	erfcTabEst    = gaussEst(erfcPieces[:])
	ndtrNegTabEst = gaussEst(ndtrNegPieces[:])
	ndtrPosTabEst = estPieces(ndtrPosPieces[:])
)

// erfcTabHL returns erfc(x) as hi + lo for 1/2 <= x <= 25 (error about 2^-59
// relative; the value is at least 2^-909, so no part underflows).
func erfcTabHL(x float64) (hi, lo float64) {
	z2 := float64(x * x)
	return erfcTabEst[min(int(4*z2), int(z2)+48)].evalTab(x)
}

// ndtrNegHL returns Φ(-t) as hi + lo for 3/4 <= t <= 36.
func ndtrNegHL(t float64) (hi, lo float64) {
	t2 := float64(t * t)
	return ndtrNegTabEst[min(int(2*t2), int(0.5*t2)+48)].evalTab(t)
}

// ndtrPosHL returns Φ(x) as hi + lo for -3/4 <= x < 8.5.
func ndtrPosHL(x float64) (hi, lo float64) {
	return ndtrPosTabEst[int((x+0.75)*8)].evalTab(x)
}

// logNdtrTailBase is the top 15 bits of the float64 pattern of 8: the piece
// of L(t) = log Φ(-t) + t²/2 (8 per octave of t on [8, 2048)) is the top 15
// bits of t minus this.
const logNdtrTailBase = (1023 + 3) << 3

var logNdtrTailEst = [64]estPiece(estPieces(logNdtrTailPieces[:]))
