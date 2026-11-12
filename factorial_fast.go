package special

import "math"

// logGammaLargeHLL is logGammaLargeHL2(x, 0) that also returns the logarithm
// it forms on the way, log x = lh + ll (the very values logHLn(x) returns:
// the same kernel, written out as there). Factorial needs log Γ(x) + log x
// and would otherwise run the logarithm kernel twice. It is a copy rather
// than a change of logGammaLargeHL2 because geninline expands that function
// into the array variants.
func logGammaLargeHLL(x float64) (hi, lo, lh, ll float64) {
	ixb := math.Float64bits(x)
	tmp := ixb - 0x3fe8000000000000
	kf := float64(int64(tmp) >> 52)
	m := math.Float64frombits(ixb - uint64(int64(tmp)>>52)<<52)
	t := &logT[int(float64(m*256)+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1
	r2 := float64(r * r)
	q0 := math.FMA(r, 1.0/3, -0.5)
	q1 := math.FMA(r, 1.0/5, -0.25)
	q2 := math.FMA(r, 1.0/7, -1.0/6)
	q2 = math.FMA(r2, -1.0/8, q2)
	q1 = math.FMA(r2, q2, q1)
	q := float64(r2 * math.FMA(r2, q1, q0))
	l0 := float64(kf*ln2Hi42) + t[1]
	lh = l0 + r
	ll = (r - (lh - l0)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q))
	// (x - 1/2) log x
	ix := 1 / x
	mh := x - 0.5
	ml := (x - mh) - 0.5 // nonzero only above 2^52
	ph := float64(mh * lh)
	pl := math.FMA(mh, lh, -ph) + math.FMA(mh, ll, ml*lh)
	// S(x): stirlingHL
	const c1lo = 4.625929269271485e-18 // 1/12 - RN(1/12)
	c := &stirlingCoef
	ixl := float64(math.FMA(-ix, x, 1) * ix)
	z := float64(ix * ix)
	z2 := float64(z * z)
	z4 := float64(z2 * z2)
	sa := math.FMA(math.FMA(c[4], z, c[3]), z2, math.FMA(c[2], z, c[1]))
	sb := math.FMA(math.FMA(c[8], z, c[7]), z2, math.FMA(c[6], z, c[5]))
	tail := float64(math.FMA(sb, z4, sa) * z * ix)
	sh := float64(ix * c[0])
	sl := math.FMA(ix, c[0], -sh) + math.FMA(ixl, c[0], math.FMA(ix, c1lo, tail))
	v := ph - x // Fast2Sum: (x - 1/2) log x >= 2.2 x here
	vl := (ph - v) - x
	h1 := v + lnSqrt2Pi.Hi // Fast2Sum: v > 13
	e1 := (v - h1) + lnSqrt2Pi.Hi
	hi = h1 + sh // Fast2Sum: sh < 0.01
	e2 := (h1 - hi) + sh
	return hi, (e1 + e2) + (vl + pl + (lnSqrt2Pi.Lo + sl)), lh, ll // xl = 0: no first-order term
}
