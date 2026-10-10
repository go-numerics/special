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

// Exp returns exp(a.Hi + a.Lo) rounded to float64: the double-double
// mantissa from ExpMant is accurate to about 2^-97, so the result is
// correctly rounded except in cases closer to a rounding boundary than that.
func Exp(a DD) float64 {
	m, k := ExpMant(a)
	return math.Ldexp(m.Hi+m.Lo, k)
}

// ExpMant returns m, k with exp(a) = m * 2^k, m in [1, 2) as a
// double-double accurate to about 2^-97 relative, so that a caller can
// multiply further factors in before the single final rounding. k is not
// limited to the float64 exponent range: a scaled result may be combined
// with others before math.Ldexp decides overflow or underflow. For
// |a.Hi| > 1e7 it returns m = +Inf or m = 0 with k = 0 (NaN for NaN).
//
// Method (Tang, "Table-driven implementation of the exponential function in
// IEEE floating-point arithmetic", ACM TOMS 15(2), 1989, carried to
// double-double): a = (64k + j) ln2/64 + r with ln2/64 split in three parts
// and every partial product exact, |r| <= ln2/128; exp(r) by its Taylor series
// with the terms through r⁴/24 in double-double and r⁵/120 + ... + r^10/10!
// (below 2^-44 relative) in float64; 2^(j/64) from a 300-bit
// mpmath table (zz_exptable.go).
func ExpMant(a DD) (m DD, k int) {
	x := a.Hi
	switch {
	case x != x:
		return DD{x, 0}, 0
	case x > 1e7:
		return DD{math.Inf(1), 0}, 0
	case x < -1e7:
		return DD{}, 0
	}
	n := math.Floor(x/ln2o64Hi + 0.5)
	// r = a - n ln2/64, each partial product exact (TwoProd).
	r := Sub(a, MulFF(n, ln2o64Hi))
	r = Sub(r, Add(MulFF(n, ln2o64Mid), MulFF(n, ln2o64Lo)))
	ni := int(n)
	j := ni & 63
	k = (ni - j) / 64
	// exp(r) = 1 + r + r²/2 + r³/6 + r⁴ Σ_{i>=0} r^i/(i+4)!; |r| < 0.0055
	// so r^11/11! < 2^-106 ends the series.
	r2 := Mul(r, r)
	rh := r.Hi
	p := 1.0 / 3628800
	for _, c := range [...]float64{1.0 / 362880, 1.0 / 40320, 1.0 / 5040, 1.0 / 720, 1.0 / 120} {
		p = p*rh + c
	}
	r4 := Mul(r2, r2)
	tail := r4.Hi * rh * p
	e := Add(r, MulF(r2, 0.5))
	e = Add(e, DivF(Mul(r2, r), 6))
	e = Add(e, DivF(r4, 24))
	e = AddF(e, tail)
	e = AddF(e, 1)
	return Mul(e, expTable[j]), k
}

// Ldexp returns a * 2^k for a double-double (both parts scaled).
func Ldexp(a DD, k int) DD {
	return DD{math.Ldexp(a.Hi, k), math.Ldexp(a.Lo, k)}
}

// Expm1 returns exp(a)-1 for |a| small enough that the result is not
// dominated by the -1 (callers use it for |a| < 1).
func Expm1(a DD) float64 {
	em := math.Expm1(a.Hi)
	return em + a.Lo*(1+em)
}

// Log returns log(x) for finite x > 0 as a double-double accurate to about
// 2^-100 relative.
//
// Method (table-driven, after Tang, "Table-driven implementation of the
// logarithm function in IEEE floating-point arithmetic", ACM TOMS 16(4),
// 1990): x = 2^k m with m in [1/sqrt2, sqrt2); c = j/128 is the grid point
// nearest m, and log m = log c + 2 atanh(s), s = (m-c)/(m+c). m-c is exact
// (Sterbenz) and |s| < 1/360, so the atanh series 2s(1 + s²/3 + s⁴/5 + ...)
// needs three terms in double-double and four more in float64. log c comes from a
// 300-bit mpmath table (zz_logtable.go); k ln2 is added exactly.
//
// Log(0) = -Inf, Log(+Inf) = +Inf, NaN for x < 0 or NaN (as math.Log).
func Log(x float64) DD {
	if !(x > 0) || math.IsInf(x, 1) {
		return DD{math.Log(x), 0}
	}
	m, k := math.Frexp(x) // x = m * 2^k, m in [0.5, 1)
	if m < math.Sqrt2/2 {
		m *= 2
		k--
	}
	j := math.Floor(m*128 + 0.5)
	c := j / 128
	num := m - c // exact
	h, l := TwoSum(m, c)
	s := Div(DD{num, 0}, DD{h, l})
	s2 := Mul(s, s)
	z := s2.Hi
	// series = 1 + s2/3 + s2^2/5 + s2^3 (1/7 + s2/9 + s2^2/11 + s2^3/13);
	// s2 < 7.7e-6, so the omitted s2^7/15 is below 2^-125.
	t := Add(Div(s2, DD{3, 0}), DivF(Mul(s2, s2), 5))
	t = AddF(t, z*z*z*(1.0/7+z*(1.0/9+z*(1.0/11+z/13))))
	t = AddF(t, 1)
	lm := MulF(Mul(s, t), 2)
	lm = Add(lm, logTable[int(j)-logTableLo])
	kk := float64(k)
	return Add(Add(MulFF(kk, Ln2Hi), MulFF(kk, Ln2Lo)), lm)
}
