// Package dd is a minimal double-double ("float128-lite") arithmetic used
// where a special function must carry more than 53 bits through an
// intermediate step: the argument of an exponential, a long product, a
// cancellation. A value is Hi + Lo with |Lo| <= ulp(Hi)/2.
//
// The error-free transformations are the classical ones: TwoSum (Knuth,
// TAOCP vol. 2, 4.2.2), Fast2Sum (Dekker 1971) and TwoProd through a fused
// multiply-add (Ogita, Rump and Oishi, "Accurate sum and dot product", SIAM
// J. Sci. Comput. 26(6), 2005). Algorithms for the compound operations follow
// Joldes, Muller and Popescu, "Tight and rigorous error bounds for basic
// building blocks of double-word arithmetic", ACM TOMS 44(2), 2017
// (DWPlusDW: Algorithm 6, DWTimesDW: Algorithm 12, DWDivDW: Algorithm 18).
package dd

import "math"

// DD is an unevaluated sum Hi + Lo.
type DD struct{ Hi, Lo float64 }

// TwoSum returns s, e with s = fl(a+b) and s + e = a + b exactly.
func TwoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	e = (a - (s - bb)) + (b - bb)
	return
}

// FastTwoSum requires |a| >= |b| (or a == 0).
func FastTwoSum(a, b float64) (s, e float64) {
	s = a + b
	e = b - (s - a)
	return
}

// TwoProd returns p, e with p = fl(a*b) and p + e = a*b exactly (barring
// underflow).
//
// The explicit float64 conversion is load-bearing: the Go specification
// allows a*b to be fused with a later addition (an FMA on arm64, ppc64le,
// s390x, riscv64, loong64), which would break every error-free
// transformation built on p. A conversion forces the rounding.
func TwoProd(a, b float64) (p, e float64) {
	p = float64(a * b)
	e = math.FMA(a, b, -p)
	return
}

// F returns x as a DD.
func F(x float64) DD { return DD{x, 0} }

// Float rounds to the nearest float64.
func (a DD) Float() float64 { return a.Hi + a.Lo }

// Add returns a + b (DWPlusDW, relative error <= 3u^2).
func Add(a, b DD) DD {
	s, t := TwoSum(a.Hi, b.Hi)
	u, v := TwoSum(a.Lo, b.Lo)
	t += u
	s, t = FastTwoSum(s, t)
	t += v
	s, t = FastTwoSum(s, t)
	return DD{s, t}
}

// AddF returns a + b for a float64 b (DWPlusFP, relative error <= 2u^2).
func AddF(a DD, b float64) DD {
	s, t := TwoSum(a.Hi, b)
	t += a.Lo
	s, t = FastTwoSum(s, t)
	return DD{s, t}
}

// Neg returns -a.
func Neg(a DD) DD { return DD{-a.Hi, -a.Lo} }

// Sub returns a - b.
func Sub(a, b DD) DD { return Add(a, Neg(b)) }

// Mul returns a * b (DWTimesDW with FMA, relative error <= 5u^2).
func Mul(a, b DD) DD {
	p, e := TwoProd(a.Hi, b.Hi)
	t := a.Hi * b.Lo
	t = math.FMA(a.Lo, b.Hi, t)
	e += t
	p, e = FastTwoSum(p, e)
	return DD{p, e}
}

// MulF returns a * b for a float64 b (DWTimesFP, relative error <= 2u^2).
func MulF(a DD, b float64) DD {
	p, e := TwoProd(a.Hi, b)
	e = math.FMA(a.Lo, b, e)
	p, e = FastTwoSum(p, e)
	return DD{p, e}
}

// MulFF returns the exact product a*b of two float64.
func MulFF(a, b float64) DD {
	p, e := TwoProd(a, b)
	return DD{p, e}
}

// Div returns a / b (DWDivDW, relative error <= 15u^2 + 56u^3).
func Div(a, b DD) DD {
	q := a.Hi / b.Hi
	// r = a - q*b
	r := Sub(a, MulF(b, q))
	q2 := r.Hi / b.Hi
	r = Sub(r, MulF(b, q2))
	q3 := r.Hi / b.Hi
	s, e := FastTwoSum(q, q2)
	return AddF(DD{s, e}, q3)
}

// DivF returns a / b for float64 b.
func DivF(a DD, b float64) DD { return Div(a, DD{b, 0}) }

// Sqrt returns sqrt(a) (one Newton step from the float64 root).
func Sqrt(a DD) DD {
	if a.Hi <= 0 {
		return DD{math.Sqrt(a.Hi), 0}
	}
	s := math.Sqrt(a.Hi)
	// s + (a - s^2)/(2s)
	r := Sub(a, MulFF(s, s))
	return AddF(DD{s, 0}, r.Hi/(2*s))
}

// ln2 split as a double-double: Ln2Hi + Ln2Lo = ln 2 to 106 bits.
const (
	Ln2Hi = 6.93147180559945286227e-01
	Ln2Lo = 2.31904681384629955842e-17
)

// Exp returns exp(a.Hi + a.Lo) rounded to float64 with an error below one
// ulp (about 0.8 ulp): the argument is reduced by k*ln2 in double-double, the
// reduced exponential is formed as 1 + expm1(r) so that math.Expm1's error
// is scaled by |expm1(r)|/(1+expm1(r)) <= 0.3, and the low part of the
// argument enters through the first-order term.
func Exp(a DD) float64 {
	x := a.Hi
	if x != x {
		return x
	}
	if x > 709.8 {
		return math.Exp(x + a.Lo) // overflow region; math.Exp decides
	}
	if x < -745.2 {
		return 0
	}
	k := math.Floor(x/Ln2Hi + 0.5)
	// r = a - k*ln2 in double-double. k*Ln2Hi is exact through TwoProd.
	r := Sub(a, Add(MulFF(k, Ln2Hi), MulFF(k, Ln2Lo)))
	em := math.Expm1(r.Hi)
	// exp(r.Hi + r.Lo) = (1+em)(1+r.Lo) ~= 1 + em + r.Lo*(1+em)
	s, e := TwoSum(1, em)
	e += r.Lo * (1 + em)
	return math.Ldexp(s+e, int(k))
}

// Expm1 returns exp(a)-1 for |a| small enough that the result is not
// dominated by the -1 (callers use it for |a| < 1).
func Expm1(a DD) float64 {
	em := math.Expm1(a.Hi)
	return em + a.Lo*(1+em)
}

// Log returns log(x) for finite x > 0 as a double-double accurate to about
// 2^-68 relative: the reduced logarithm is the atanh series
// log m = 2s(1 + s^2/3 + s^4/5 + ...), s = (m-1)/(m+1), whose first three
// terms are carried in double-double and the remainder (< 3e-5 of the
// total) in float64; k*ln2 is added exactly.
func Log(x float64) DD {
	m, k := math.Frexp(x) // x = m * 2^k, m in [0.5, 1)
	if m < math.Sqrt2/2 {
		m *= 2
		k--
	}
	// s = (m-1)/(m+1); m-1 is exact (Sterbenz), m+1 is exact in DD.
	num := DD{m - 1, 0}
	h, l := TwoSum(m, 1)
	s := Div(num, DD{h, l})
	s2 := Mul(s, s)
	// |s| <= 0.1716, s^2 <= 0.02944.
	z := s2.Hi
	tail := 0.0
	for i := 31; i >= 7; i -= 2 {
		tail = tail*z + 1/float64(i)
	}
	// series = 1 + s2/3 + s2^2/5 + s2^3*tail
	s4 := Mul(s2, s2)
	t := Add(DD{1, 0}, Add(Div(s2, DD{3, 0}), Div(s4, DD{5, 0})))
	t = AddF(t, z*z*z*tail)
	lm := MulF(Mul(s, t), 2)
	kk := float64(k)
	return Add(Add(MulFF(kk, Ln2Hi), MulFF(kk, Ln2Lo)), lm)
}
