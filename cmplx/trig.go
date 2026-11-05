package cmplx

import (
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

// sincos returns sin y and cos y for finite y, each to about 1 ulp
// whatever |y| (math.Sincos where |y| ≤ π/4, which it does not reduce).
func sincos(y float64) (s, c float64) {
	ay := math.Abs(y)
	if ay <= math.Pi/4 {
		return math.Sincos(y)
	}
	k, r := reduce(ay)
	sh, ch := math.Sincos(r.Hi)
	// sin(h+l) = sin h + l cos h, cos(h+l) = cos h - l sin h (l² below 2^-100).
	s = math.FMA(r.Lo, ch, sh)
	c = math.FMA(-r.Lo, sh, ch)
	switch k {
	case 1:
		s, c = c, -s
	case 2:
		s, c = -s, -c
	case 3:
		s, c = -c, s
	}
	if y < 0 {
		s = -s
	}
	return s, c
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

// tanSmall is math.Tan except for |y| < 2^-20, where math.Tan (Cephes)
// returns y itself once y² < 1e-14, dropping y³/3: a relative error up to
// 3.3e-15, 15 ulp, for 1e-8 < |y| < 1e-7. There the series to y^5 is used
// (truncation y^6·17/315 relative, below 2^-120).
func tanSmall(y float64) float64 {
	if math.Abs(y) >= 0x1p-20 {
		return math.Tan(y)
	}
	q := float64(y * y)
	c := 1.0/3 + float64(q*(2.0/15))
	return math.FMA(float64(y*q), c, y)
}
