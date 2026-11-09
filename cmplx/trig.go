package cmplx

import (
	"github.com/go-numerics/special/internal/libm"
	"math"
	"math/bits"

	"github.com/go-numerics/special/dd"
)

// Accurate real sine, cosine and tangent for the complex functions.
//
// math.Sin, Cos, Tan and Sincos reduce |y| < 2^29 by π/4 in three float64
// parts (Cephes), each subtraction rounded, and larger |y| by Payne–Hanek
// (math/trig_reduce.go) keeping a 64-bit fraction that is rounded to
// float64 and, for odd octants, decremented by one (z--) in float64. Both
// keep the reduced argument to an absolute, not relative, accuracy, so
// where sin or cos is near a zero the result loses up to hundreds of ulps
// (439 ulp measured, go-numerics/special#21). Here every |y| > π/4 is
// reduced by Payne–Hanek with a 256-bit window of 4/π, the odd-octant
// complement taken exactly in integer arithmetic, and the reduced argument
// kept as a double-double; sin and cos of it are math.Sincos on its high
// part (no reduction there) corrected by its low part.
//
// The reduction follows Go's trigReduce (BSD-3-Clause, The Go Authors),
// after K. C. Ng, "Argument reduction for huge arguments: good to the last
// bit", 1992, with the window widened from 128 to 256 bits.

// pi4 is π/4 as a double-double (error below 2^-160 relative).
var pi4 = dd.DD{Hi: pio2a / 2, Lo: pio2b / 2}

// reduce returns k and r with y = k·π/2 + r, |r| ≤ π/4 (to rounding), r as
// a double-double accurate to about 2^-100 relative, for finite y > π/4.
// Only k mod 4 is returned.
func reduce(y float64) (k uint64, r dd.DD) {
	b := math.Float64bits(y)
	exp := int(b>>52&0x7ff) - 1023 - 52
	ix := b&(1<<52-1) | 1<<52
	// Windows of 4/π aligned so that the product's leading word has the
	// integer part mod 8 in its top 3 bits (exp ≥ -53 as y > π/4).
	digit, shift := uint(exp+61)/64, uint(exp+61)%64
	var z [5]uint64
	for i := range z {
		z[i] = fourOverPi[digit+uint(i)]<<shift | fourOverPi[digit+uint(i)+1]>>(64-shift)
	}
	h4, l4 := bits.Mul64(z[4], ix)
	h3, l3 := bits.Mul64(z[3], ix)
	h2, l2 := bits.Mul64(z[2], ix)
	h1, l1 := bits.Mul64(z[1], ix)
	l0 := z[0] * ix
	_ = l4
	s3, c := bits.Add64(l3, h4, 0)
	s2, c := bits.Add64(l2, h3, c)
	s1, c := bits.Add64(l1, h2, c)
	s0, _ := bits.Add64(l0, h1, c)
	j := s0 >> 61
	// The fraction of y·4/π, 0.w0 w1 w2 w3.
	w := [4]uint64{s0<<3 | s1>>61, s1<<3 | s2>>61, s2<<3 | s3>>61, s3 << 3}
	neg := j&1 == 1
	if neg {
		// Odd octant: r = (f - 1)·π/4; 1 - f exactly (two's complement).
		j++
		var br uint64 = 1
		for i := 3; i >= 0; i-- {
			w[i], br = bits.Add64(^w[i], 0, br)
		}
	}
	f := dd.Mul(fracDD(w), pi4)
	if neg {
		f = dd.Neg(f)
	}
	return (j >> 1) & 3, f
}

// fracDD returns the nonzero 256-bit fraction 0.w0 w1 w2 w3 as a
// double-double (the leading 128 significant bits, rounded to about
// 2^-106). Leading zero words are skipped, so a fraction of any size
// keeps its relative accuracy.
func fracDD(w [4]uint64) dd.DD {
	var x [6]uint64 // w followed by two zero words
	copy(x[:], w[:])
	i := 0
	for i < 3 && x[i] == 0 {
		i++
	}
	lz := uint(bits.LeadingZeros64(x[i]))
	m1 := x[i]<<lz | x[i+1]>>(64-lz)
	m2 := x[i+1]<<lz | x[i+2]>>(64-lz)
	e := -64*i - int(lz)
	hi := math.Ldexp(float64(m1>>11), e-53)
	lo := math.Ldexp(float64(m1&0x7ff), e-64) + math.Ldexp(float64(m2), e-128)
	hi, lo = dd.FastTwoSum(hi, lo)
	return dd.DD{Hi: hi, Lo: lo}
}

// sincosSmall returns sin r and cos r for a double-double |r| ≤ π/4 + 2^-10
// to about 2^-66 relative: r = c + d with c = j/64 (sin c, cos c from a
// 300-bit mpmath table), |d| ≤ 1/128, sin d = d(1 - d²/6 + d⁴/120 - d⁶/5040)
// with the part past d in float64 (below 2^-15 of d), cos d = 1 - d²/2 in
// double-double plus d⁴/24 - d⁶/720 in float64.
func sincosSmall(r dd.DD) (s, c dd.DD) {
	neg := r.Hi < 0
	if neg {
		r = dd.Neg(r)
	}
	j := math.Round(r.Hi * 64)
	d := dd.AddF(r, float64(-j/64))
	t := float64(d.Hi * d.Hi)
	// d - d³/6 in double-double, d⁵/120 - d⁷/5040 (below 2^-30 of d) in float64.
	d3 := dd.Mul(dd.Sqr(d), d)
	ps := float64(float64(t*t)*(1.0/120)) - float64(float64(t*t)*float64(t*(1.0/5040)))
	sd := dd.AddF(dd.Sub(d, dd.DivF(d3, 6)), float64(d.Hi*ps))
	// cos d = 1 - d²/2 + d⁴/24 in double-double, - d⁶/720 in float64.
	d2 := dd.Sqr(d)
	cd := dd.Add(dd.Ldexp(dd.Neg(d2), -1), dd.DivF(dd.Sqr(d2), 24))
	cd = dd.AddF(dd.AddF(cd, -float64(float64(t*t)*float64(t*(1.0/720)))), 1)
	tab := sinCosT[int(j)]
	sc, cc := dd.DD{Hi: tab[0], Lo: tab[1]}, dd.DD{Hi: tab[2], Lo: tab[3]}
	s = dd.Add(dd.Mul(sc, cd), dd.Mul(cc, sd))
	c = dd.Sub(dd.Mul(cc, cd), dd.Mul(sc, sd))
	if neg {
		s = dd.Neg(s)
	}
	return s, c
}

// sincosDD returns sin y and cos y as double-doubles for finite y, to
// about 2^-66 relative whatever |y|: Payne–Hanek reduction (reduce), then
// sincosSmall and the quadrant.
func sincosDD(y float64) (s, c dd.DD) { return sincosW(dd.F(y)) }

// sincosW is sincosDD for a double-double argument (Pow's b·log a): the
// high part is reduced, the low part added to the reduced argument.
func sincosW(w dd.DD) (s, c dd.DD) {
	y := w.Hi
	aw := w
	if math.Signbit(y) {
		aw = dd.Neg(w)
	}
	var k uint64
	r := aw
	if aw.Hi > math.Pi/4 {
		k, r = reduce(aw.Hi)
		r = dd.AddF(r, aw.Lo)
	}
	s, c = sincosSmall(r)
	switch k {
	case 1:
		s, c = c, dd.Neg(s)
	case 2:
		s, c = dd.Neg(s), dd.Neg(c)
	case 3:
		s, c = dd.Neg(c), s
	}
	if math.Signbit(y) {
		s = dd.Neg(s)
	}
	return s, c
}

// sincos returns sin y and cos y for finite y, each correctly rounded but
// in rare cases within 2^-66 of a rounding boundary (sincosDD rounded once;
// the sign of a zero y is kept).
func sincos(y float64) (s, c float64) {
	if y == 0 {
		return y, 1
	}
	sd, cd := sincosDD(y)
	return sd.Hi + sd.Lo, cd.Hi + cd.Lo
}

// tan returns tan y to about 1 ulp for finite y: tanSmall on the reduced
// argument, corrected by its low part, and -1/tan r in odd quadrants.
func tan(y float64) float64 {
	ay := math.Abs(y)
	if ay <= math.Pi/4 {
		return tanSmall(y)
	}
	k, r := reduce(ay)
	th := tanSmall(r.Hi)
	t := math.FMA(r.Lo, math.FMA(th, th, 1), th)
	if k&1 == 1 {
		t = -1 / t
	}
	if y < 0 {
		t = -t
	}
	return t
}

// atanDD returns atan q for a double-double q in [0, 1], to about 2^-65
// relative: atan q = atan c + atan s with c = j/64 the nearest grid point
// (atan c from a 300-bit mpmath table), s = (q - c)/(1 + qc) in
// double-double, |s| ≤ 1/128, and atan s = s - s³/3 + s⁵/5 - s⁷/7 with the
// part past s in float64 (below 2^-14 of s; truncation below 2^-63).
// math.Atan (Cephes) is about 1.2 ulp, which a logarithm's imaginary part
// would inherit.
func atanDD(q dd.DD) dd.DD {
	j := math.Round(q.Hi * 64)
	c := float64(j / 64)
	s := dd.Div(dd.AddF(q, -c), dd.AddF(dd.MulF(q, c), 1))
	t := float64(s.Hi * s.Hi)
	p := -1.0/3 + float64(t*(1.0/5-float64(t*(1.0/7))))
	r := dd.AddF(s, float64(float64(s.Hi*t)*p))
	a := atanT[int(j)]
	return dd.Add(dd.DD{Hi: a[0], Lo: a[1]}, r)
}

// tanSmall is math.Tan except for |y| < 2^-20, where math.Tan (Cephes)
// returns y itself once y² < 1e-14, dropping y³/3: a relative error up to
// 3.3e-15, 15 ulp, for 1e-8 < |y| < 1e-7. There the series to y^5 is used
// (truncation y^6·17/315 relative, below 2^-120).
func tanSmall(y float64) float64 {
	if math.Abs(y) >= 0x1p-20 {
		return libm.Tan(y)
	}
	q := float64(y * y)
	c := 1.0/3 + float64(q*(2.0/15))
	return math.FMA(float64(y*q), c, y)
}
