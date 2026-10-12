package special

import "math"

// Estrin evaluation of the fitted piecewise polynomials.
//
// evalPiecesDD runs Horner's rule over a slice: up to 19 dependent
// multiply-adds per call, a latency chain that dominates the cost of the
// functions built on it. The pieces are copied here, at initialisation,
// into fixed arrays (zero-padded at the high-degree end) and evaluated by
// Estrin's scheme (Knuth, TAOCP vol. 2, 4.6.4), whose dependency depth is
// logarithmic in the degree, finished by three Horner steps for accuracy. The value is the same polynomial with a
// different rounding pattern: the constant term stays an exact
// double-double and only q(s), whose contribution s·q(s) is a fraction of
// the result, is rounded.

// estPiece is a polyPiece with q(s) = (p(s) - p(0))/s in ascending order.
type estPiece struct {
	c, c0, c0lo float64
	c1lo        float64 // low part of q[0] (zero for tables that do not emit it)
	q           [16]float64
}

// estPieces converts fit.py pieces (coefficients descending, constant
// last, at most 16 non-constant coefficients).
func estPieces(p []polyPiece) []estPiece {
	out := make([]estPiece, len(p))
	for i := range p {
		n := len(p[i].coef) - 1
		e := &out[i]
		e.c, e.c0, e.c0lo, e.c1lo = p[i].c, p[i].coef[n], p[i].c0lo, p[i].c1lo
		for k := 0; k < n; k++ {
			e.q[k] = p[i].coef[n-1-k]
		}
	}
	return out
}

// eval returns the piece's polynomial at t as hi + lo: c0 + s·q(s) with
// the product (TwoProd) and the sum (Fast2Sum: on every fitted piece
// |s·q(s)| < |c0|) exact, so that only the rounding of q(s) itself remains,
// scaled down by |s·q(s)/c0|. q is evaluated by Horner for its three
// lowest-order steps, so that the rounding errors of the Estrin part are
// scaled down by s³ (|s| <= 1/4 on every piece in its own units), and by
// Estrin's scheme for q[3..15]. (Written out as one leaf function: a call
// costs the spilling of every live register under Go's ABI.)
func (e *estPiece) eval(t float64) (hi, lo float64) {
	s := t - e.c
	q := &e.q
	s2 := s * s
	s4 := s2 * s2
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
	p := math.FMA(c1, s4*s4, c0)
	p = math.FMA(math.FMA(math.FMA(p, s, q[2]), s, q[1]), s, q[0])
	r := float64(s * p) // float64(): not fused into hi
	hi = e.c0 + r
	lo = (r - (hi - e.c0)) + (math.FMA(s, p, -r) + e.c0lo)
	return hi, lo
}
