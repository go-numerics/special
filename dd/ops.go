package dd

import (
	"fmt"
	"math"
	"math/big"
)

// Accumulation, comparison and conversion. The dot product and the sum are
// the "Dot2/Sum2" idea of Ogita, Rump and Oishi (2005) carried to a
// double-double accumulator: every product is formed exactly with TwoProd and
// added with Add, so the result is accurate to about the double-double
// precision (a few u^2 per term, u = 2^-53) times the condition number, not
// to u times it.

// AddProd returns acc + a*b with the product a*b taken exactly (relative
// error of the sum <= 3u^2 from Add).
func AddProd(acc DD, a, b float64) DD {
	p, e := TwoProd(a, b)
	return Add(acc, DD{p, e})
}

// AddProdDD returns acc + a*b for double-double a and b (Mul error <= 5u^2
// on the product, then Add).
func AddProdDD(acc, a, b DD) DD { return Add(acc, Mul(a, b)) }

// Sum returns the sum of x, accumulated in double-double.
func Sum(x []float64) DD {
	var s DD
	for _, v := range x {
		s = AddF(s, v)
	}
	return s
}

// Dot returns sum x[i]*y[i] with exact products and a double-double
// accumulator. It panics if the lengths differ.
func Dot(x, y []float64) DD {
	if len(x) != len(y) {
		panic("dd: Dot: length mismatch")
	}
	var s DD
	for i, v := range x {
		s = AddProd(s, v, y[i])
	}
	return s
}

// DotDD returns sum x[i]*y[i] for double-double operands. It panics if the
// lengths differ.
func DotDD(x, y []DD) DD {
	if len(x) != len(y) {
		panic("dd: DotDD: length mismatch")
	}
	var s DD
	for i, v := range x {
		s = AddProdDD(s, v, y[i])
	}
	return s
}

// Sqr returns a*a.
func Sqr(a DD) DD { return Mul(a, a) }

// Abs returns |a|.
func Abs(a DD) DD {
	if a.Hi < 0 || (a.Hi == 0 && math.Signbit(a.Hi)) {
		return Neg(a)
	}
	return a
}

// Sign returns -1, 0 or +1 (0 for zero and NaN).
func (a DD) Sign() int {
	switch {
	case a.Hi > 0:
		return 1
	case a.Hi < 0:
		return -1
	}
	return 0
}

// IsNaN reports whether a is NaN.
func (a DD) IsNaN() bool { return a.Hi != a.Hi || a.Lo != a.Lo }

// IsInf reports whether a is infinite.
func (a DD) IsInf() bool { return math.IsInf(a.Hi, 0) }

// Cmp compares a and b: -1, 0, +1. A normalised pair is compared by Hi then
// Lo; with a NaN operand it returns 0.
func Cmp(a, b DD) int {
	switch {
	case a.Hi < b.Hi:
		return -1
	case a.Hi > b.Hi:
		return 1
	case a.Lo < b.Lo:
		return -1
	case a.Lo > b.Lo:
		return 1
	}
	return 0
}

// FromInt64 returns v exactly (a 64-bit integer fits in two float64).
func FromInt64(v int64) DD {
	h := float64(v)
	if h == 0x1p63 { // v > 2^63 - 2^10: int64(h) would overflow
		return DD{h, -float64(uint64(1)<<63 - uint64(v))}
	}
	return DD{h, float64(v - int64(h))}
}

// FromBig rounds x to the nearest double-double (Hi = nearest float64, Lo =
// nearest float64 of the remainder). Infinities map to Inf, nil to NaN.
func FromBig(x *big.Float) DD {
	if x == nil {
		return DD{math.NaN(), 0}
	}
	if x.IsInf() {
		return DD{math.Inf(x.Sign()), 0}
	}
	h, _ := x.Float64()
	if math.IsInf(h, 0) {
		return DD{h, 0}
	}
	r := new(big.Float).SetPrec(x.Prec()+64).Sub(x, new(big.Float).SetFloat64(h))
	l, _ := r.Float64()
	return DD{h, l}
}

// Big returns Hi + Lo exactly. It panics on NaN (as big.Float does).
func (a DD) Big() *big.Float {
	z := new(big.Float).SetPrec(2200)
	z.SetFloat64(a.Hi)
	if a.Lo != 0 {
		z.Add(z, new(big.Float).SetFloat64(a.Lo))
	}
	return z
}

// Parse converts a decimal (or any big.Float-accepted) string to the nearest
// double-double, e.g. a NIST StRD certified value.
func Parse(s string) (DD, error) {
	x, _, err := big.ParseFloat(s, 10, 300, big.ToNearestEven)
	if err != nil {
		return DD{}, fmt.Errorf("dd: parse %q: %w", s, err)
	}
	return FromBig(x), nil
}

// String formats a with 32 significant digits.
func (a DD) String() string {
	if a.IsNaN() || a.IsInf() {
		return fmt.Sprint(a.Hi)
	}
	return a.Big().Text('g', 32)
}
