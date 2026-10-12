package special

import "math"

// Float64 exp and log kernels returning an unevaluated sum hi + lo.
//
// The double-double functions of internal/dd carry about 2^-72 relative
// accuracy and cost 10–20 ns per call, most of it in renormalisations that
// the error analysis of a final float64 result does not need. These kernels
// keep only the error-free steps that matter: an exact argument reduction,
// the table value as a double-double, and one rounding of a term that is
// already 2^-8 smaller than the result. Their output hi + lo is accurate to
// about 2^-61 (exp) and 2^-68 absolute (log) — far below the half ulp the
// caller's final rounding adds — at the cost of a few floating-point
// operations and one table lookup.
//
// Method: the table-driven algorithms of Tang (ACM TOMS 15(2), 1989 and
// 16(4), 1990) in the arrangement of ARM's optimized-routines exp/log
// (Szabolcs Nagy; also glibc's): exp with N = 128 and a Cody–Waite
// reduction, log with x = 2^k m, m in [0.75, 1.5), r = m·(1/c) - 1 formed
// exactly through an FMA (TwoProd) and -log(1/c) tabulated for the rounded
// reciprocal itself (testdata/gen/kernel_tables.py).

// pow2 returns 2^k for -1022 <= k <= 1023.
func pow2(k int) float64 { return math.Float64frombits(uint64(k+1023) << 52) }

// expHL returns hi, lo, k with exp(xh + xl) = (hi + lo)·2^k, hi in
// [0.99, 2.02), |lo| <= ulp(hi)/2 (normalised), relative error below 2^-60 for |xh| <= 745 and
// |xl| <= 2^-40 |xh| (or |xl| <= 2^-50). Callers check the range.
//
// x = (128k' + j) ln2/128 + u, |u| <= ln2/256 + 2^-60: kd·expC1 is exact
// and so is xh - kd·expC1 (Cody–Waite); exp(u) - 1 = u + u²P(u) with P to
// u⁴/720 (truncation 2^-72); 2^(j/128) from a 400-bit table.
func expHL(xh, xl float64) (hi, lo float64, k int) {
	kd := math.RoundToEven(xh * expInvLn2N)
	ki := int(kd)
	r := xh - kd*expC1  // exact
	dr := xl - kd*expC2 // |dr| <= 2^-24, rounding below 2^-77
	u := r + dr         //
	ue := dr - (u - r)  // Fast2Sum (or below 2^-77 when |r| < |dr|)
	u2 := u * u
	a := math.FMA(u, 1.0/6, 0.5)
	b := math.FMA(u, 1.0/120, 1.0/24)
	b = math.FMA(u2, 1.0/720, b)
	p := u2 * math.FMA(u2, b, a) // exp(u) - 1 - u, |p| < 2^-17.4
	t := &expT[ki&127]
	th, tl := t[0], t[1]
	// 2^(j/128)·(1 + u + ue + p) = th + [th·u + th·(ue+p) + tl·(1+u)]
	lo = math.FMA(th, u, math.FMA(th, ue+p, math.FMA(tl, u, tl)))
	hi = th + lo // Fast2Sum: |lo| < 2^-7 th
	lo -= hi - th
	return hi, lo, ki >> 7
}

// logHL returns hi + lo = log(x) for finite x > 0 (normal or subnormal),
// lo not normalised (|lo| < 2^-15 |hi|), with absolute error below 2^-68
// and relative error below 2^-62 (inside the central interval
// |x - 1| < 1/512 the error is relative to x - 1).
//
// x = 2^k m with m in [0.75, 1.5) (the exponent taken from the bits of x
// minus those of 0.75, as in optimized-routines); c = j/256 the grid point
// nearest m; invc = RN(1/c); r = m·invc - 1 = rh + pe exactly (TwoProd,
// then a Sterbenz subtraction; |r| < 2^-8.5); log x = k ln2 - log(invc) +
// log1p(r), the table holding -log(invc) for the float64 invc itself, with
// k·ln2Hi42 + logc_hi exact (both multiples of 2^-42) and log1p(r) - r
// summed to r^8/8 (truncation below 2^-76).
func logHL(x float64) (hi, lo float64) {
	ix := math.Float64bits(x)
	ke := 0
	if ix < 1<<52 { // subnormal
		ix = math.Float64bits(x * 0x1p54)
		ke = -54
	}
	tmp := ix - 0x3fe8000000000000 // bits of 0.75
	ke += int(int64(tmp) >> 52)
	m := math.Float64frombits(ix - uint64(int64(tmp)>>52)<<52)
	t := &logT[int(m*256+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1 // exact: p in [0.99, 1.01]
	// log1p(r + pe) = r + pe(1 - r) + q(r), q = -r²/2 + r³/3 - ... - r^8/8
	r2 := r * r
	c0 := math.FMA(r, 1.0/3, -0.5)
	c1 := math.FMA(r, 1.0/5, -0.25)
	c2 := math.FMA(r, 1.0/7, -1.0/6)
	c2 = math.FMA(r2, -1.0/8, c2)
	c1 = math.FMA(r2, c2, c1)
	q := r2 * math.FMA(r2, c1, c0)
	kf := float64(ke)
	lh := kf*ln2Hi42 + t[1] // exact
	// hi = lh + r: Fast2Sum (|lh| >= |r| unless lh = 0).
	hi = lh + r
	lo = (r - (hi - lh)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q))
	return hi, lo
}
