package cmplx

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// zeroExp is the exponent given to an exact zero product: far enough below
// any other that scaling it is a no-op, small enough that sums of two
// exponents do not overflow a 32-bit int.
const zeroExp = -1 << 20

// prod returns a*b exactly as m * 2^e, with m a double-double in [1, 4) in
// magnitude: each factor is first scaled to [1, 2), so the product neither
// overflows nor underflows whatever the exponents of a and b.
func prod(a, b float64) (m dd.DD, e int) {
	if a == 0 || b == 0 {
		return dd.DD{}, zeroExp
	}
	ea, eb := math.Ilogb(a), math.Ilogb(b)
	return dd.MulFF(math.Ldexp(a, -ea), math.Ldexp(b, -eb)), ea + eb
}

// sum returns p*2^s + q*2^t as m * 2^e, the larger term unscaled. A term
// more than about 2^-1000 below the other loses its low bits, far below the
// last bit of the sum.
func sum(p dd.DD, s int, q dd.DD, t int) (dd.DD, int) {
	e := max(s, t)
	return dd.Add(dd.Ldexp(p, s-e), dd.Ldexp(q, t-e)), e
}

// dot returns a*b + c*d as m * 2^e, both products exact and the sum carried
// to about 2^-104 relative to the result (dd.Add, DWPlusDW).
func dot(a, b, c, d float64) (dd.DD, int) {
	p, s := prod(a, b)
	q, t := prod(c, d)
	return sum(p, s, q, t)
}

// round returns m * 2^e rounded to float64. naive is the same quantity
// computed with the builtin operators: where the exact value is zero, it
// carries the sign IEEE 754 gives an exact zero sum (and Annex G expects).
func round(m dd.DD, e int, naive float64) float64 {
	if m.Hi == 0 {
		return math.Copysign(0, naive)
	}
	r := math.Ldexp(m.Hi, e)
	if math.Abs(r) >= 0x1p-1022 {
		return r // normal or infinite: m.Lo is below half an ulp of m.Hi
	}
	// Subnormal result: Ldexp rounded m.Hi to a multiple of s = 2^(-1074-e)
	// (ties to even). |m.Lo| is at most half an ulp of m.Hi, so m.Lo can
	// change that rounding only where m.Hi was a tie: rem = ±s/2 exactly.
	half := math.Ldexp(0.5, -1074-e)
	switch rem := m.Hi - math.Ldexp(r, -e); {
	case rem == half && m.Lo > 0:
		r += 0x1p-1074
	case rem == -half && m.Lo < 0:
		r -= 0x1p-1074
	}
	return r
}

func finite(x float64) bool { return !math.IsInf(x, 0) && !math.IsNaN(x) }

// box returns ±1 with the sign of x where x is infinite, ±0 otherwise (the
// "box the infinity" step of Annex G G.5.1).
func box(x float64) float64 {
	if math.IsInf(x, 0) {
		return math.Copysign(1, x)
	}
	return math.Copysign(0, x)
}

// zeroNaN replaces a NaN by a zero with its sign bit, any other value
// unchanged.
func zeroNaN(x float64) float64 {
	if math.IsNaN(x) {
		return math.Copysign(0, x)
	}
	return x
}

// Mul returns x*y. Each part, ac - bd and ad + bc, is the sum of two exact
// products rounded once, so it is correct to about 1 ulp even where the two
// products cancel (the builtin * rounds each product first and loses every
// digit there) and where a product alone overflows or underflows.
// Infinities and NaNs follow C11 Annex G G.5.1 (the _Cmultd example): a
// product with an infinite factor is an infinity even where the naive
// formula gives NaN+NaN i.
func Mul(x, y complex128) complex128 {
	a, b, c, d := real(x), imag(x), real(y), imag(y)
	if finite(a) && finite(b) && finite(c) && finite(d) {
		re, er := dot(a, c, -b, d)
		im, ei := dot(a, d, b, c)
		return complex(round(re, er, float64(a*c)-float64(b*d)), round(im, ei, float64(a*d)+float64(b*c)))
	}
	ac, bd, ad, bc := float64(a*c), float64(b*d), float64(a*d), float64(b*c)
	re, im := ac-bd, ad+bc
	if !math.IsNaN(re) || !math.IsNaN(im) {
		return complex(re, im)
	}
	recalc := false
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		a, b = box(a), box(b)
		c, d = zeroNaN(c), zeroNaN(d)
		recalc = true
	}
	if math.IsInf(c, 0) || math.IsInf(d, 0) {
		c, d = box(c), box(d)
		a, b = zeroNaN(a), zeroNaN(b)
		recalc = true
	}
	if !recalc && (math.IsInf(ac, 0) || math.IsInf(bd, 0) || math.IsInf(ad, 0) || math.IsInf(bc, 0)) {
		// A finite product overflowed next to a NaN part: recover the
		// infinity.
		a, b, c, d = zeroNaN(a), zeroNaN(b), zeroNaN(c), zeroNaN(d)
		recalc = true
	}
	if recalc {
		re = float64(math.Inf(1) * (float64(a*c) - float64(b*d)))
		im = float64(math.Inf(1) * (float64(a*d) + float64(b*c)))
	}
	return complex(re, im)
}

// Div returns x/y. The numerators ac + bd and bc - ad and the denominator
// c² + d² are exact sums of exact products (scaled by powers of two, so
// nothing overflows or underflows on the way), the quotient is taken in
// double-double and rounded once: each part is correct to about 1 ulp.
// Infinities, zeros and NaNs follow C11 Annex G G.5.1 (the _Cdivd example).
func Div(x, y complex128) complex128 {
	a, b, c, d := real(x), imag(x), real(y), imag(y)
	if !finite(a) || !finite(b) || !finite(c) || !finite(d) || (c == 0 && d == 0) {
		return divAnnexG(a, b, c, d)
	}
	den, ed := dot(c, c, d, d)
	nre, er := dot(a, c, b, d)
	nim, ei := dot(b, c, -a, d)
	naive := divAnnexG(a, b, c, d)
	return complex(round(dd.Div(nre, den), er-ed, real(naive)), round(dd.Div(nim, den), ei-ed, imag(naive)))
}

// divAnnexG is the _Cdivd example of Annex G G.5.1: logb scaling of the
// divisor, the quotient, then the recovery of infinities and zeros.
func divAnnexG(a, b, c, d float64) complex128 {
	ilogbw := 0
	logbw := math.Logb(math.Max(math.Abs(c), math.Abs(d)))
	if finite(logbw) {
		ilogbw = int(logbw)
		c, d = math.Ldexp(c, -ilogbw), math.Ldexp(d, -ilogbw)
	}
	denom := float64(c*c) + float64(d*d)
	x := math.Ldexp((float64(a*c)+float64(b*d))/denom, -ilogbw)
	y := math.Ldexp((float64(b*c)-float64(a*d))/denom, -ilogbw)
	if math.IsNaN(x) && math.IsNaN(y) {
		switch {
		case denom == 0 && (!math.IsNaN(a) || !math.IsNaN(b)):
			x = float64(math.Copysign(math.Inf(1), c) * a)
			y = float64(math.Copysign(math.Inf(1), c) * b)
		case (math.IsInf(a, 0) || math.IsInf(b, 0)) && finite(c) && finite(d):
			a, b = box(a), box(b)
			x = float64(math.Inf(1) * (float64(a*c) + float64(b*d)))
			y = float64(math.Inf(1) * (float64(b*c) - float64(a*d)))
		case math.IsInf(logbw, 1) && finite(a) && finite(b):
			c, d = box(c), box(d)
			x = float64(0 * (float64(a*c) + float64(b*d)))
			y = float64(0 * (float64(b*c) - float64(a*d)))
		}
	}
	return complex(x, y)
}

// Proj returns the projection of x onto the Riemann sphere (C11 7.3.9.5):
// x itself, or +Inf + i·0 with the sign of imag(x) when either part is an
// infinity (even if the other is NaN).
func Proj(x complex128) complex128 {
	if math.IsInf(real(x), 0) || math.IsInf(imag(x), 0) {
		return complex(math.Inf(1), math.Copysign(0, imag(x)))
	}
	return x
}
