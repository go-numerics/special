package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Bessel functions of order 0 and 1 of the first and second kind.
//
// On (0, 8] these are the minimax rational approximations of Boost.Math
// (bessel_j0.hpp, bessel_j1.hpp, bessel_y0.hpp, bessel_y1.hpp; Xiaogang
// Zhang and John Maddock, Boost Software License 1.0), in which the zeros of
// the function in the interval are factored out exactly, so the relative
// error stays small next to them.
//
// Between the power series near 0 and x = 100 the function is summed as a
// Taylor series about the nearest zero of the function or of its
// derivative, tabulated in double-double with the value and the derivative
// there; the higher coefficients follow from Bessel's equation (the method
// of J. Harrison, "Fast and accurate Bessel function computation", ARITH
// 19, 2009). With h = x - z formed exactly, the leading term d*h carries
// the value with full relative accuracy however close x is to a zero.
//
// From 100 on, besselAsymJY: Hankel's expansion in modulus-phase form with
// the phase in double-double.

// besselZeroTaylorMax is where besselAsymJY takes over from the series about
// the tabulated zeros.
const besselZeroTaylorMax = 100

// besselPointTaylor evaluates the solution of Bessel's equation of order n,
// x*x*y2 + x*y1 + (x*x - n*n)*y = 0 (y1, y2 the derivatives), as a Taylor
// series about the tabulated point nearest to x. The points are the zeros
// of the function and of its derivative, tab[i] = {z, y(z), y'(z)}, each a
// double-double, so |x - z| <= ~pi/4 beyond the first point.
func besselPointTaylor(tab [][6]float64, n2, x float64, singular bool) float64 {
	i := int((x-tab[0][0])/(math.Pi/2) + 0.5)
	if i < 0 {
		i = 0
	}
	if i >= len(tab) {
		i = len(tab) - 1
	}
	// For Y, whose singularity at 0 limits the radius of convergence about
	// z to z, the point is chosen by |x - z|/z; for J by |x - z|.
	dist := func(j int) float64 {
		d := math.Abs(x - tab[j][0])
		if singular {
			d /= tab[j][0]
		}
		return d
	}
	for i+1 < len(tab) && dist(i+1) < dist(i) {
		i++
	}
	for i > 0 && dist(i-1) < dist(i) {
		i--
	}
	z := tab[i]
	hs, he := dd.TwoSum(x, -z[0])
	h := dd.AddF(dd.DD{Hi: hs, Lo: he}, -z[1])
	z0 := z[0]
	// a[k] are the Taylor coefficients about z0; from the equation, for
	// k >= 0, z0^2 (k+1)(k+2) a[k+2] = -[z0 (k+1)(2k+1) a[k+1]
	//   + (k^2 + z0^2 - n^2) a[k] + 2 z0 a[k-1] + a[k-2]].
	var a [64]float64
	a[0], a[1] = z[2], z[4]
	hh := math.Abs(h.Hi)
	lead := math.Max(math.Abs(a[0]), math.Abs(a[1]*hh))
	K := len(a) - 1
	small := 0
	pw := hh
	for k := 0; k+2 < len(a); k++ {
		fk := float64(k)
		s := z0*(fk+1)*(2*fk+1)*a[k+1] + (fk*fk+z0*z0-n2)*a[k]
		if k >= 1 {
			s += 2 * z0 * a[k-1]
		}
		if k >= 2 {
			s += a[k-2]
		}
		a[k+2] = -s / (z0 * z0 * (fk + 1) * (fk + 2))
		pw *= hh
		if math.Abs(a[k+2])*pw <= 0x1p-60*lead {
			small++
			if small == 2 {
				K = k + 2
				break
			}
		} else {
			small = 0
		}
	}
	tail := 0.0
	for k := K; k >= 2; k-- {
		tail = tail*h.Hi + a[k]
	}
	tail *= h.Hi * h.Hi
	r := dd.Add(dd.DD{Hi: z[2], Lo: z[3]}, dd.Mul(dd.DD{Hi: z[4], Lo: z[5]}, h))
	r = dd.AddF(r, tail)
	return r.Hi + r.Lo
}

// besselRational evaluates (p[0] + p[1] z + ...)/(q[0] + q[1] z + ...)
// (coefficients in ascending order, as in Boost's evaluate_rational).
// For |z| > 1 both polynomials are evaluated in 1/z from the other end,
// as Boost does (same degree here, so the powers of z cancel).
func besselRational(p, q []float64, z float64) float64 {
	n, d := 0.0, 0.0
	if math.Abs(z) <= 1 || len(p) != len(q) {
		for i := len(p) - 1; i >= 0; i-- {
			n = n*z + p[i]
		}
		for i := len(q) - 1; i >= 0; i-- {
			d = d*z + q[i]
		}
		return n / d
	}
	w := 1 / z
	for i := 0; i < len(p); i++ {
		n = n*w + p[i]
	}
	for i := 0; i < len(q); i++ {
		d = d*w + q[i]
	}
	return n / d
}

// besselPoly evaluates p[0] + p[1] z + ... (ascending order).
func besselPoly(p []float64, z float64) float64 {
	s := 0.0
	for i := len(p) - 1; i >= 0; i-- {
		s = s*z + p[i]
	}
	return s
}

// besselEulerDD is Euler's constant gamma as a double-double.
var besselEulerDD = dd.DD{Hi: 0.5772156649015329, Lo: -4.942915152430645e-18}

// besselJ0Series is J0(x) = sum (-x^2/4)^k / (k!)^2 (DLMF 10.2.2) for
// 0 <= x <= 1, where the terms fall by a factor 4 at least.
func besselJ0Series(x float64) float64 {
	m := -x * x / 4
	t, s := 1.0, 0.0
	for k := 1; k < 30; k++ {
		t *= m / float64(k*k)
		s += t
		if math.Abs(t) < 0x1p-60 {
			break
		}
	}
	return 1 + s
}

// besselJ1Series is J1(x) = (x/2) sum (-x^2/4)^k / (k! (k+1)!) for
// 0 <= x <= 1.5.
func besselJ1Series(x float64) float64 {
	m := -x * x / 4
	t, s := 1.0, 0.0
	for k := 1; k < 30; k++ {
		t *= m / float64(k*(k+1))
		s += t
		if math.Abs(t) < 0x1p-60 {
			break
		}
	}
	return x / 2 * (1 + s)
}

// besselY0Series is Y0(x) for 0 < x <= 0.6 by DLMF 10.8.2:
// Y0 = (2/pi) [(log(x/2) + gamma) J0(x) + sum_{k>=1} (-1)^(k+1) H_k (x^2/4)^k / (k!)^2],
// with the logarithm and the constants in double-double.
func besselY0Series(x float64) float64 {
	q := x * x / 4
	t, s, H := 1.0, 0.0, 0.0
	for k := 1; k < 30; k++ {
		fk := float64(k)
		H += 1 / fk
		t *= -q / (fk * fk)
		s -= H * t
		if math.Abs(t) < 0x1p-60 {
			break
		}
	}
	L := dd.Add(dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo}), besselEulerDD)
	r := dd.AddF(dd.MulF(L, besselJ0Series(x)), s)
	return dd.Mul(r, besselTwoOverPiDD).Float()
}

// besselY1Series is Y1(x) for 0 < x <= 1.5 by DLMF 10.8.1 with n = 1:
// Y1 = -2/(pi x) + (2/pi) log(x/2) J1(x)
//   - (x/(2 pi)) sum (psi(k+1) + psi(k+2)) (-x^2/4)^k / (k! (k+1)!).
func besselY1Series(x float64) float64 {
	m := -x * x / 4
	g := besselEulerDD.Hi
	t, H := 1.0, 0.0
	s := 1 - 2*g // k = 0: psi(1) + psi(2) = 1 - 2 gamma
	for k := 1; k < 30; k++ {
		fk := float64(k)
		H += 1 / fk
		t *= m / (fk * (fk + 1))
		s += t * (2*H + 1/(fk+1) - 2*g)
		if math.Abs(t) < 0x1p-60 {
			break
		}
	}
	lx := dd.Sub(dd.Log(x), dd.DD{Hi: dd.Ln2Hi, Lo: dd.Ln2Lo})
	// (2/pi) [ -1/x + log(x/2) J1(x) - (x/4) s ]
	inv := dd.Neg(recipDD(x))
	if math.IsInf(inv.Hi, 0) {
		return math.Inf(-1)
	}
	r := dd.Add(inv, dd.MulF(lx, besselJ1Series(x)))
	r = dd.AddF(r, -x/4*s)
	if x < 0x1p-1000 {
		// keep 2/(pi x) from overflowing in the double-double product
		return r.Hi * besselTwoOverPiDD.Hi
	}
	return dd.Mul(r, besselTwoOverPiDD).Float()
}

// J0 returns the Bessel function of the first kind of order zero,
//
//	J0(x) = (1/pi) * integral_0^pi cos(x sin t) dt,
//
// for real x (even in x). J0(0) = 1, J0(±Inf) = 0, J0(NaN) = NaN.
//
// Algorithm: Boost.Math's minimax rational approximations with the first
// two zeros factored out on [0, 8]; a Taylor series about the nearest zero
// (Harrison 2009) on [8, 100); Hankel's expansion in modulus-phase form with
// the phase in double-double and exact (Payne-Hanek) reduction beyond. The
// error stays within a few ulps of the value also next to the zeros.
func J0(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 0):
		return 0
	case x <= 1.5:
		return besselJ0Series(x)
	case x < besselZeroTaylorMax:
		return besselPointTaylor(besselJ0Points[:], 0, x, false)
	}
	j, _, _ := besselAsymJY(0, x, true, false)
	return j
}

// J1 returns the Bessel function of the first kind of order one,
// J1(x) = (1/pi) * integral_0^pi cos(t - x sin t) dt, for real x (odd in x).
// J1(0) = 0, J1(±Inf) = 0, J1(NaN) = NaN.
//
// Algorithm: as J0 (Boost.Math bessel_j1.hpp on [0, 8], Taylor series about
// the nearest zero on [8, 100), modulus-phase Hankel expansion beyond).
func J1(x float64) float64 {
	sign := 1.0
	if x < 0 {
		x, sign = -x, -1
	}
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 0):
		return 0
	case x <= 2:
		return sign * besselJ1Series(x)
	case x < besselZeroTaylorMax:
		return sign * besselPointTaylor(besselJ1Points[:], 1, x, false)
	}
	j, _, _ := besselAsymJY(1, x, true, false)
	return sign * j
}

// Y0 returns the Bessel function of the second kind of order zero,
// Y0(x) = (2/pi)(log(x/2) + gamma) J0(x) + ..., defined for x > 0.
// Y0(0) = -Inf, Y0(+Inf) = 0, Y0(x < 0) = NaN, Y0(NaN) = NaN.
//
// Algorithm: Boost.Math bessel_y0.hpp on (0, 8] with the three zeros
// factored out (and the logarithm taken as log1p of the exact distance to
// the zero), a Taylor series about the nearest zero on [8, 100), and the
// modulus-phase Hankel expansion beyond.
func Y0(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	case math.IsInf(x, 1):
		return 0
	case x <= 0.6:
		return besselY0Series(x)
	case x < besselZeroTaylorMax:
		return besselPointTaylor(besselY0Points[:], 0, x, true)
	}
	_, y, _ := besselAsymJY(0, x, false, true)
	return y
}

// Y1 returns the Bessel function of the second kind of order one, defined
// for x > 0. Y1(0) = -Inf, Y1(+Inf) = 0, Y1(x < 0) = NaN, Y1(NaN) = NaN;
// Y1(x) overflows to -Inf for x below about 3.5e-309.
//
// Algorithm: as Y0 (Boost.Math bessel_y1.hpp on (0, 8]).
func Y1(x float64) float64 {
	switch {
	case math.IsNaN(x) || x < 0:
		return math.NaN()
	case x == 0:
		return math.Inf(-1)
	case math.IsInf(x, 1):
		return 0
	case x <= 1.5:
		return besselY1Series(x)
	case x < besselZeroTaylorMax:
		return besselPointTaylor(besselY1Points[:], 1, x, true)
	}
	_, y, _ := besselAsymJY(1, x, false, true)
	return y
}
