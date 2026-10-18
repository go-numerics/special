package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// Bessel functions of order 0 and 1 of the first and second kind.
//
// Near 0 the power series of DLMF 10.2.2 (J) and 10.8.1-2 (Y, with the
// logarithm in double-double) are summed; they converge quickly there.
//
// Below x = 100 (from 0 for J, from the end of the series for Y) the
// function is one polynomial per cell (zz_besselCells_coeffs.go,
// go -C tools run ./genfit bessel-cells): the cells are centred on the zeros of
// the function and of its derivative (the points of J. Harrison, "Fast and
// accurate Bessel function computation", ARITH 19, 2009) and on the
// midpoints between them, and on each f(z + h) = a0 + a1 h + h² R(h) with
// a0, a1 exact to double-double and R fitted to mpmath. With h = x - z
// formed exactly, the leading term a1 h carries the value with full
// relative accuracy however close x is to a zero.
//
// From 100 on, besselAsymJY: Hankel's expansion in modulus-phase form with
// the phase in double-double.

// besselZeroTaylorMax is where besselAsymJY takes over from the cell
// polynomials.
const besselZeroTaylorMax = 100

// besselCell is one cell of zz_besselCells_coeffs.go: on [lo, next lo)
// the function is f(z + h) = a0 + a1 h + h² R(h), h = x - z, with z, a0, a1
// double-doubles (a0 = 0 when z is a zero of f) and R a degree-11
// polynomial (r descending).
type besselCell struct {
	lo, z, zlo, a0, a0lo, a1, a1lo float64
	r                              [12]float64
}

// besselCellEval evaluates the cell table at x in [cells[0].lo, 100).
//
// The cell is found from the uniform table lut (8 entries per unit of x,
// each the cell holding the left end of its interval), then by stepping
// right while x is past the next cell's edge. h = x - z is exact
// (Sterbenz: every cell lies within [z/2, 2z]); z has a low part only at a
// zero of f, where h = x - z - zlo is formed as a double-double.
// R is summed in float64; w = a1 + h R and then a0 + h w are formed with
// error-free transformations, so that the only roundings that reach the
// result are those of h R (|h² R| is a small fraction of |f| on every
// cell, or vanishes with h next to a zero) and the final one.
func besselCellEval(cells []besselCell, lut []uint16, x float64) float64 {
	i := int(lut[int(x*8)])
	for i+1 < len(cells) && x >= cells[i+1].lo {
		i++
	}
	c := &cells[i]
	// x - (z + zlo) = he + hl exactly: x - z is exact, and the TwoSum
	// keeps the cancellation between it and zlo next to a zero.
	he, hl := dd.TwoSum(x-c.z, -c.zlo)
	r := &c.r
	t := r[0]
	t = t*he + r[1]
	t = t*he + r[2]
	t = t*he + r[3]
	t = t*he + r[4]
	t = t*he + r[5]
	t = t*he + r[6]
	t = t*he + r[7]
	t = t*he + r[8]
	t = t*he + r[9]
	t = t*he + r[10]
	t = t*he + r[11]
	// w = a1 + h R(h); f = a0 + h w
	wh, wl := dd.TwoSum(c.a1, float64(he*t))
	wl += c.a1lo
	p, pe := dd.TwoProd(he, wh)
	s, se := dd.TwoSum(c.a0, p)
	return s + (se + pe + he*wl + hl*wh + c.a0lo)
}

// besselRational evaluates (p[0] + p[1] z + ...)/(q[0] + q[1] z + ...)
// (coefficients in ascending order, as in Boost's evaluate_rational).
func besselRational(p, q []float64, z float64) float64 {
	return besselPoly(p, z) / besselPoly(q, z)
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
// for real x (even in x). J0(0) = 1, J0(±Inf) = 0 (the limit; SciPy returns
// NaN), J0(NaN) = NaN.
//
// Algorithm: below 100, one polynomial per cell about the zeros of J0, of
// J1 and the midpoints between them, with the value and the derivative at
// the centre in double-double (Harrison 2009); Hankel's expansion in
// modulus-phase form with the phase in double-double and exact
// (Payne-Hanek) reduction beyond. The error stays within a few ulps of the
// value, also next to the zeros.
func J0(x float64) float64 {
	x = math.Abs(x)
	switch {
	case math.IsNaN(x):
		return x
	case math.IsInf(x, 0):
		return 0
	case x < besselZeroTaylorMax:
		return besselCellEval(besselJ0Cells[:], besselJ0CellsLUT[:], x)
	}
	j, _, _ := besselAsymJY(0, x, true, false)
	return j
}

// J1 returns the Bessel function of the first kind of order one,
// J1(x) = (1/pi) * integral_0^pi cos(t - x sin t) dt, for real x (odd in x).
// J1(0) = 0, J1(±Inf) = 0, J1(NaN) = NaN.
//
// Algorithm: as J0 (cell polynomials about the zeros of J1 and of its
// derivative up to 100, modulus-phase Hankel expansion beyond).
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
	case x < besselZeroTaylorMax:
		return sign * besselCellEval(besselJ1Cells[:], besselJ1CellsLUT[:], x)
	}
	j, _, _ := besselAsymJY(1, x, true, false)
	return sign * j
}

// Y0 returns the Bessel function of the second kind of order zero,
// Y0(x) = (2/pi)(log(x/2) + gamma) J0(x) + ..., defined for x > 0.
// Y0(0) = -Inf, Y0(+Inf) = 0, Y0(x < 0) = NaN, Y0(NaN) = NaN.
//
// Algorithm: the series DLMF 10.8.2 for x <= 0.6; cell polynomials about
// the zeros of Y0 and Y1 and the midpoints between them up to 100 (cells
// split further near the logarithmic singularity at 0); the
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
		return besselCellEval(besselY0Cells[:], besselY0CellsLUT[:], x)
	}
	_, y, _ := besselAsymJY(0, x, false, true)
	return y
}

// Y1 returns the Bessel function of the second kind of order one, defined
// for x > 0. Y1(0) = -Inf, Y1(+Inf) = 0, Y1(x < 0) = NaN, Y1(NaN) = NaN;
// Y1(x) overflows to -Inf for x below about 3.5e-309.
//
// Algorithm: as Y0 (series DLMF 10.8.1 for x <= 1.5).
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
		return besselCellEval(besselY1Cells[:], besselY1CellsLUT[:], x)
	}
	_, y, _ := besselAsymJY(1, x, false, true)
	return y
}
