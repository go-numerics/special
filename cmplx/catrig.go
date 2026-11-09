package cmplx

import (
	"github.com/go-numerics/special/internal/libm"
	"math"

	"github.com/go-numerics/special/dd"
)

// The inverse trigonometric and hyperbolic functions: FreeBSD msun
// catrig.c r251404 (Stephen Montgomery-Smith, David Schultz, BSD-2-Clause),
// as NumPy's npymath carries it, after T. E. Hull, T. F. Fairgrieve and
// P. T. P. Tang, "Implementing the complex arcsine and arccosine functions
// using exception handling", ACM TOMS 23(3), 1997. The arguments of the
// real functions are arg (this package's double-double atan2) rather than
// math.Atan2, and every sum of products is written so that Go cannot fuse
// it (float64(a*b)), so the results are the same on every architecture.

const (
	epsilon      = 0x1p-52 // DBL_EPSILON
	recipEpsilon = 1 / epsilon
	sqrt6Eps     = 3.65002414998885671e-08 // sqrt(6 ε)
	sqrt3Eps     = 2.5809568279517849e-8   // sqrt(3 ε)
	aCrossover   = 10.0
	bCrossover   = 0.6417
	fourSqrtMin  = 5.9666725849601654e-154 // 4 sqrt(DBL_MIN)
	quarterSqMax = 3.3519519824856489e+153 // sqrt(DBL_MAX)/4
	sqrtMin      = 1.4916681462400413e-154 // sqrt(DBL_MIN)
)

// fHull returns f(a, b) = (hypot(a, b) - b)/2, given h = hypot(a, b).
func fHull(a, b, h float64) float64 {
	if b < 0 {
		return float64((h - b) / 2)
	}
	if b == 0 {
		return float64(a / 2)
	}
	return float64(a * a / (h + b) / 2)
}

// hardWork is msun's do_hard_work: for x, y ≥ 0 below 1/ε it returns
// rx = Re casinh(x+iy) and either B = y/A (usable) or sqrt(A² - y²) and a
// possibly rescaled y for atan2.
func hardWork(x, y float64) (rx float64, bUsable bool, b, sqrtA2my2, newY float64) {
	r := libm.Hypot(x, y+1)
	s := libm.Hypot(x, y-1)
	// A ≥ 1 mathematically; msun clamps it in case rounding says otherwise
	// (no float64 case was found in a search of 10^5 points; kept as a
	// guard).
	a := math.Max((r+s)/2, 1)
	switch {
	case a >= aCrossover:
		rx = libm.Log(a + math.Sqrt(float64(a*a)-1))
	case y == 1 && x < epsilon*epsilon/128:
		rx = math.Sqrt(x)
	case x >= epsilon*math.Abs(y-1):
		am1 := fHull(x, 1+y, r) + fHull(x, 1-y, s)
		rx = libm.Log1p(am1 + math.Sqrt(am1*(a+1)))
	case y < 1:
		rx = x / math.Sqrt((1-y)*(1+y))
	default:
		rx = libm.Log1p((y - 1) + math.Sqrt((y-1)*(y+1)))
	}
	newY = y
	if y < fourSqrtMin {
		return rx, false, 0, float64(a * (2 / epsilon)), float64(y * (2 / epsilon))
	}
	b = y / a
	if b <= bCrossover {
		return rx, true, b, 0, newY
	}
	switch {
	case y == 1 && x < epsilon/128:
		sqrtA2my2 = float64(math.Sqrt(x) * math.Sqrt((a+y)/2))
	case x >= epsilon*math.Abs(y-1):
		amy := fHull(x, y+1, r) + fHull(x, y-1, s)
		sqrtA2my2 = math.Sqrt(amy * (a + y))
	case y > 1:
		sqrtA2my2 = x * (4 / epsilon / epsilon) * y / math.Sqrt((y+1)*(y-1))
		newY = float64(y * (4 / epsilon / epsilon))
	default:
		sqrtA2my2 = math.Sqrt((1 - y) * (1 + y))
	}
	return rx, false, b, sqrtA2my2, newY
}

// clogLarge is msun's clog_for_large_values, for |z| beyond about 1/ε:
// log|z| (real part) and arg z.
func clogLarge(x, y float64) (float64, float64) {
	ax, ay := math.Abs(x), math.Abs(y)
	if ax < ay {
		ax, ay = ay, ax
	}
	var rr float64
	switch {
	case ax > math.MaxFloat64/2:
		rr = libm.Log(libm.Hypot(x/math.E, y/math.E)) + 1
	case ax > quarterSqMax || ay < sqrtMin:
		rr = libm.Log(libm.Hypot(x, y))
	default:
		rr = float64(libm.Log(float64(ax*ax)+float64(ay*ay)) / 2)
	}
	return rr, arg(y, x)
}

// Asinh returns the inverse hyperbolic sine of x. Special values: C11
// Annex G G.6.2.2.
func Asinh(z complex128) complex128 {
	x, y := real(z), imag(z)
	ax, ay := math.Abs(x), math.Abs(y)
	if math.IsNaN(x) || math.IsNaN(y) {
		switch {
		case math.IsInf(x, 0):
			return complex(x, y+y) // asinh(±Inf + i NaN) = ±Inf + i NaN
		case math.IsInf(y, 0):
			return complex(y, x+x) // asinh(NaN ± i Inf) = ±Inf + i NaN
		case y == 0:
			return complex(x+x, y) // asinh(NaN + i0) = NaN + i0
		}
		return complex(math.NaN(), math.NaN())
	}
	if ax > recipEpsilon || ay > recipEpsilon {
		var wx, wy float64
		if !math.Signbit(x) {
			wx, wy = clogLarge(x, y)
		} else {
			wx, wy = clogLarge(-x, -y)
		}
		return complex(math.Copysign(wx+math.Ln2, x), math.Copysign(wy, y))
	}
	if x == 0 && y == 0 {
		return z
	}
	if ax < sqrt6Eps/4 && ay < sqrt6Eps/4 {
		return z
	}
	rx, bUsable, b, sqrtA2my2, newY := hardWork(ax, ay)
	var ry float64
	if bUsable {
		ry = libm.Asin(b)
	} else {
		ry = arg(newY, sqrtA2my2)
	}
	return complex(math.Copysign(rx, x), math.Copysign(ry, y))
}

// Asin returns the inverse sine of x, -i Asinh(i x) computed as
// i conj(Asinh(i conj x)) (C11 G.6p7).
func Asin(x complex128) complex128 {
	w := Asinh(complex(imag(x), real(x)))
	return complex(imag(w), real(w))
}

// Acos returns the inverse cosine of x. Special values: C11 Annex G
// G.6.1.1.
func Acos(z complex128) complex128 {
	x, y := real(z), imag(z)
	sx, sy := math.Signbit(x), math.Signbit(y)
	ax, ay := math.Abs(x), math.Abs(y)
	if math.IsNaN(x) || math.IsNaN(y) {
		switch {
		case math.IsInf(x, 0):
			return complex(y+y, math.Inf(-1)) // acos(±Inf + i NaN) = NaN ∓ i Inf
		case math.IsInf(y, 0):
			return complex(x+x, -y) // acos(NaN ± i Inf) = NaN ∓ i Inf
		case x == 0:
			return complex(pio2a, y+y) // acos(±0 + i NaN) = π/2 + i NaN
		}
		return complex(math.NaN(), math.NaN())
	}
	if ax > recipEpsilon || ay > recipEpsilon {
		wx, wy := clogLarge(x, y)
		rx := math.Abs(wy)
		ry := wx + math.Ln2
		if !sy {
			ry = -ry
		}
		return complex(rx, ry)
	}
	if x == 1 && y == 0 {
		return complex(0, -y)
	}
	if ax < sqrt6Eps/4 && ay < sqrt6Eps/4 {
		return complex(pio2a-(x-pio2b), -y)
	}
	ry, bUsable, b, sqrtA2mx2, newX := hardWork(ay, ax)
	var rx float64
	switch {
	case bUsable && !sx:
		rx = libm.Acos(b)
	case bUsable:
		rx = libm.Acos(-b)
	case !sx:
		rx = arg(sqrtA2mx2, newX)
	default:
		rx = arg(sqrtA2mx2, -newX)
	}
	if !sy {
		ry = -ry
	}
	return complex(rx, ry)
}

// Acosh returns the inverse hyperbolic cosine of x, ±i Acos(x) with the
// sign that makes the real part non-negative. Special values: C11 Annex G
// G.6.2.1.
func Acosh(z complex128) complex128 {
	w := Acos(z)
	rx, ry := real(w), imag(w)
	switch {
	case math.IsNaN(rx) && math.IsNaN(ry):
		return complex(ry, rx)
	case math.IsNaN(rx):
		return complex(math.Abs(ry), rx) // acosh(NaN ± i Inf), acosh(±Inf + i NaN) = +Inf + i NaN
	case math.IsNaN(ry):
		return complex(ry, ry) // acosh(0 + i NaN) = NaN + i NaN
	}
	return complex(math.Abs(ry), math.Copysign(rx, imag(z)))
}

// realPartReciprocal is Re(1/(x+iy)) = x/(x² + y²) without spurious
// underflow, for x, y not NaN and one of them beyond 1/ε (msun, after
// C99 n1124 G.5.1 example 2).
func realPartReciprocal(x, y float64) float64 {
	const cutoff = 53/2 + 1 // DBL_MANT_DIG/2 + 1
	ix, iy := math.Ilogb(x), math.Ilogb(y)
	if y == 0 {
		iy = -1 << 20 // no int overflow in ix - iy, even with 32-bit int
	}
	switch {
	case ix-iy >= cutoff || math.IsInf(x, 0):
		return 1 / x
	case iy-ix >= cutoff:
		return x / y / y
	case ix <= 1024/2-cutoff:
		return x / (float64(x*x) + float64(y*y))
	}
	scale := math.Ldexp(1, 1-ix) // msun: 2^(1 - ilogb(x))
	x, y = float64(x*scale), float64(y*scale)
	return float64(x / (float64(x*x) + float64(y*y)) * scale)
}

// Atanh returns the inverse hyperbolic tangent of x. Special values: C11
// Annex G G.6.2.3.
func Atanh(z complex128) complex128 {
	x, y := real(z), imag(z)
	ax, ay := math.Abs(x), math.Abs(y)
	if y == 0 && ax <= 1 {
		return complex(libm.Atanh(x), y)
	}
	if x == 0 {
		return complex(x, libm.Atan(y))
	}
	if math.IsNaN(x) || math.IsNaN(y) {
		switch {
		case math.IsInf(x, 0):
			return complex(math.Copysign(0, x), y+y) // atanh(±Inf + i NaN) = ±0 + i NaN
		case math.IsInf(y, 0):
			return complex(math.Copysign(0, x), math.Copysign(pio2a, y)) // atanh(NaN ± i Inf) = ±0 ± iπ/2
		}
		return complex(math.NaN(), math.NaN())
	}
	if ax > recipEpsilon || ay > recipEpsilon {
		return complex(realPartReciprocal(x, y), math.Copysign(pio2a, y))
	}
	if ax < sqrt3Eps/2 && ay < sqrt3Eps/2 {
		return z
	}
	// Here 1/ε ≥ ax, ay and not both below sqrt(3ε)/2: msun's
	// log1p(4|x|/((|x|-1)² + y²))/4 and atan2(2|y|, 1 - x² - y²)/2, with
	// (|x|-1)² + y² and 1 - x² - y² formed from exact squares in
	// double-double (msun rounds each square, and the second difference
	// cancels near |z| = 1), the log1p in double-double, the atan2 on the
	// double-double denominator.
	var rx float64
	if ax == 1 && ay < epsilon {
		rx = float64((math.Ln2 - libm.Log(ay)) / 2)
	} else {
		s, e := dd.TwoSum(ax, -1)
		den := dd.Add(dd.Sqr(dd.DD{Hi: s, Lo: e}), dd.MulFF(ay, ay))
		l := log1pDD(dd.Div(dd.F(float64(4*ax)), den))
		rx = round(l, -2, l.Hi)
	}
	r := atan2X(float64(2*ay), dd.AddF(dd.Neg(dd.Add(dd.MulFF(ax, ax), dd.MulFF(ay, ay))), 1))
	ry := round(r, -1, r.Hi)
	return complex(math.Copysign(rx, x), math.Copysign(ry, y))
}

// Atan returns the inverse tangent of x, -i Atanh(i x) computed as
// i conj(Atanh(i conj x)) (C11 G.6p7).
func Atan(x complex128) complex128 {
	w := Atanh(complex(imag(x), real(x)))
	return complex(imag(w), real(w))
}
