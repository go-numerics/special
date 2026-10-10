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
// On [8, 100) the function is summed as a Taylor series about its nearest
// zero z, stored in double-double together with the derivative there; the
// higher coefficients follow from Bessel's equation (the method of
// J. Harrison, "Fast and accurate Bessel function computation", ARITH 19,
// 2009). With h = x - z formed exactly, the leading term d*h carries the
// value with full relative accuracy however close x is to the zero.
//
// From 100 on, besselAsymJY: Hankel's expansion in modulus-phase form with
// the phase in double-double.

// besselZeroTaylorMax is where besselAsymJY takes over from the series about
// the tabulated zeros.
const besselZeroTaylorMax = 100

// besselZeroTaylor evaluates the solution of x^2 y'' + x y' + (x^2 - n^2) y = 0
// that vanishes at the tabulated zero nearest to x, with the tabulated
// derivative there: tab[i] = {z.hi, z.lo, d.hi, d.lo}.
func besselZeroTaylor(tab [][4]float64, n2, x float64) float64 {
	i := int((x-tab[0][0])/math.Pi + 0.5)
	if i < 0 {
		i = 0
	}
	if i >= len(tab) {
		i = len(tab) - 1
	}
	for i+1 < len(tab) && math.Abs(x-tab[i+1][0]) < math.Abs(x-tab[i][0]) {
		i++
	}
	for i > 0 && math.Abs(x-tab[i-1][0]) < math.Abs(x-tab[i][0]) {
		i--
	}
	z := tab[i]
	hs, he := dd.TwoSum(x, -z[0])
	h := dd.AddF(dd.DD{Hi: hs, Lo: he}, -z[1])
	z0 := z[0]
	// a[k] are the Taylor coefficients about z0: a0 = 0, a1 = d; from the
	// equation, for k >= 0,
	// z0^2 (k+1)(k+2) a[k+2] = -[z0 (k+1)(2k+1) a[k+1] + (k^2 + z0^2 - n^2) a[k]
	//                            + 2 z0 a[k-1] + a[k-2]].
	var a [48]float64
	a[1] = z[2]
	hh := math.Abs(h.Hi)
	lead := math.Abs(a[1] * hh)
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
	r := dd.AddF(dd.Mul(dd.DD{Hi: z[2], Lo: z[3]}, h), tail)
	return r.Hi + r.Lo
}

// besselRational evaluates (p[0] + p[1] z + ...)/(q[0] + q[1] z + ...)
// (coefficients in ascending order, as in Boost's evaluate_rational).
func besselRational(p, q []float64, z float64) float64 {
	n, d := 0.0, 0.0
	for i := len(p) - 1; i >= 0; i-- {
		n = n*z + p[i]
	}
	for i := len(q) - 1; i >= 0; i-- {
		d = d*z + q[i]
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

// Boost.Math bessel_j0.hpp coefficients.
var (
	besselJ0P1 = []float64{-4.1298668500990866786e+11, 2.7282507878605942706e+10, -6.2140700423540120665e+08, 6.6302997904833794242e+06, -3.6629814655107086448e+04, 1.0344222815443188943e+02, -1.2117036164593528341e-01}
	besselJ0Q1 = []float64{2.3883787996332290397e+12, 2.6328198300859648632e+10, 1.3985097372263433271e+08, 4.5612696224219938200e+05, 9.3614022392337710626e+02, 1.0, 0.0}
	besselJ0P2 = []float64{-1.8319397969392084011e+03, -1.2254078161378989535e+04, -7.2879702464464618998e+03, 1.0341910641583726701e+04, 1.1725046279757103576e+04, 4.4176707025325087628e+03, 7.4321196680624245801e+02, 4.8591703355916499363e+01}
	besselJ0Q2 = []float64{-3.5783478026152301072e+05, 2.4599102262586308984e+05, -8.4055062591169562211e+04, 1.8680990008359188352e+04, -2.9458766545509337327e+03, 3.3307310774649071172e+02, -2.5258076240801555057e+01, 1.0}
)

// besselJ0Small is J0 on [0, 8] (Boost.Math bessel_j0.hpp).
func besselJ0Small(x float64) float64 {
	const (
		x1  = 2.4048255576957727686e+00
		x2  = 5.5200781102863106496e+00
		x11 = 6.160e+02
		x12 = -1.42444230422723137837e-03
		x21 = 1.4130e+03
		x22 = 5.46860286310649596604e-04
	)
	if x <= 4 {
		y := x * x
		r := besselRational(besselJ0P1, besselJ0Q1, y)
		return (x + x1) * ((x - x11/256) - x12) * r
	}
	y := 1 - (x*x)/64
	r := besselRational(besselJ0P2, besselJ0Q2, y)
	return (x + x2) * ((x - x21/256) - x22) * r
}

// Boost.Math bessel_j1.hpp coefficients.
var (
	besselJ1P1 = []float64{-1.4258509801366645672e+11, 6.6781041261492395835e+09, -1.1548696764841276794e+08, 9.8062904098958257677e+05, -4.4615792982775076130e+03, 1.0650724020080236441e+01, -1.0767857011487300348e-02}
	besselJ1Q1 = []float64{4.1868604460820175290e+12, 4.2091902282580133541e+10, 2.0228375140097033958e+08, 5.9117614494174794095e+05, 1.0742272239517380498e+03, 1.0, 0.0}
	besselJ1P2 = []float64{-1.7527881995806511112e+16, 1.6608531731299018674e+15, -3.6658018905416665164e+13, 3.5580665670910619166e+11, -1.8113931269860667829e+09, 5.0793266148011179143e+06, -7.5023342220781607561e+03, 4.6179191852758252278e+00}
	besselJ1Q2 = []float64{1.7253905888447681194e+18, 1.7128800897135812012e+16, 8.4899346165481429307e+13, 2.7622777286244082666e+11, 6.4872502899596389593e+08, 1.1267125065029138050e+06, 1.3886978985861357615e+03, 1.0}
)

// besselJ1Small is J1 on [0, 8] (Boost.Math bessel_j1.hpp).
func besselJ1Small(x float64) float64 {
	const (
		x1  = 3.8317059702075123156e+00
		x2  = 7.0155866698156187535e+00
		x11 = 9.810e+02
		x12 = -3.2527979248768438556e-04
		x21 = 1.7960e+03
		x22 = -3.8330184381246462950e-05
	)
	if x < 0x1p-30 {
		return x / 2 // J1(x) = x/2 (1 - x^2/8 + ...)
	}
	y := x * x
	if x <= 4 {
		r := besselRational(besselJ1P1, besselJ1Q1, y)
		return x * (x + x1) * ((x - x11/256) - x12) * r
	}
	r := besselRational(besselJ1P2, besselJ1Q2, y)
	return x * (x + x2) * ((x - x21/256) - x22) * r
}

// Boost.Math bessel_y0.hpp coefficients.
var (
	besselY0P1 = []float64{1.0723538782003176831e+11, -8.3716255451260504098e+09, 2.0422274357376619816e+08, -2.1287548474401797963e+06, 1.0102532948020907590e+04, -1.8402381979244993524e+01}
	besselY0Q1 = []float64{5.8873865738997033405e+11, 8.1617187777290363573e+09, 5.5662956624278251596e+07, 2.3889393209447253406e+05, 6.6475986689240190091e+02, 1.0}
	besselY0P2 = []float64{-2.2213976967566192242e+13, -5.5107435206722644429e+11, 4.3600098638603061642e+10, -6.9590439394619619534e+08, 4.6905288611678631510e+06, -1.4566865832663635920e+04, 1.7427031242901594547e+01}
	besselY0Q2 = []float64{4.3386146580707264428e+14, 5.4266824419412347550e+12, 3.4015103849971240096e+10, 1.3960202770986831075e+08, 4.0669982352539552018e+05, 8.3030857612070288823e+02, 1.0}
	besselY0P3 = []float64{-8.0728726905150210443e+15, 6.7016641869173237784e+14, -1.2829912364088687306e+11, -1.9363051266772083678e+11, 2.1958827170518100757e+09, -1.0085539923498211426e+07, 2.1363534169313901632e+04, -1.7439661319197499338e+01}
	besselY0Q3 = []float64{3.4563724628846457519e+17, 3.9272425569640309819e+15, 2.2598377924042897629e+13, 8.6926121104209825246e+10, 2.4727219475672302327e+08, 5.3924739209768057030e+05, 8.7903362168128450017e+02, 1.0}
)

// besselLogRatio returns log(x/z) for z = zh + zl, where d = x - z has been
// formed exactly: near z through log1p(d/z), so it keeps its relative
// accuracy as x approaches the zero z.
func besselLogRatio(x, z, d float64) float64 {
	if math.Abs(d) < 0.5*z {
		return math.Log1p(d / z)
	}
	return math.Log(x / z)
}

// besselY0Small is Y0 on (0, 8] (Boost.Math bessel_y0.hpp; the logarithm
// near each zero is taken through log1p of the exact difference).
func besselY0Small(x float64) float64 {
	const (
		x1  = 8.9357696627916752158e-01
		x2  = 3.9576784193148578684e+00
		x3  = 7.0860510603017726976e+00
		x11 = 2.280e+02
		x12 = 2.9519662791675215849e-03
		x21 = 1.0130e+03
		x22 = 6.4716931485786837568e-04
		x31 = 1.8140e+03
		x32 = 1.1356030177269762362e-04
	)
	y := x * x
	switch {
	case x <= 3:
		d := (x - x11/256) - x12
		z := 2 / math.Pi * besselLogRatio(x, x1, d) * besselJ0Small(x)
		return z + (x+x1)*d*besselRational(besselY0P1, besselY0Q1, y)
	case x <= 5.5:
		d := (x - x21/256) - x22
		z := 2 / math.Pi * besselLogRatio(x, x2, d) * besselJ0Small(x)
		return z + (x+x2)*d*besselRational(besselY0P2, besselY0Q2, y)
	default:
		d := (x - x31/256) - x32
		z := 2 / math.Pi * besselLogRatio(x, x3, d) * besselJ0Small(x)
		return z + (x+x3)*d*besselRational(besselY0P3, besselY0Q3, y)
	}
}

// Boost.Math bessel_y1.hpp coefficients.
var (
	besselY1P1 = []float64{4.0535726612579544093e+13, 5.4708611716525426053e+12, -3.7595974497819597599e+11, 7.2144548214502560419e+09, -5.9157479997408395984e+07, 2.2157953222280260820e+05, -3.1714424660046133456e+02}
	besselY1Q1 = []float64{3.0737873921079286084e+14, 4.1272286200406461981e+12, 2.7800352738690585613e+10, 1.2250435122182963220e+08, 3.8136470753052572164e+05, 8.2079908168393867438e+02, 1.0}
	besselY1P2 = []float64{1.1514276357909013326e+19, -5.6808094574724204577e+18, -2.3638408497043134724e+16, 4.0686275289804744814e+15, -5.9530713129741981618e+13, 3.7453673962438488783e+11, -1.1957961912070617006e+09, 1.9153806858264202986e+06, -1.2337180442012953128e+03}
	besselY1Q2 = []float64{5.3321844313316185697e+20, 5.6968198822857178911e+18, 3.0837179548112881950e+16, 1.1187010065856971027e+14, 3.0221766852960403645e+11, 6.3550318087088919566e+08, 1.0453748201934079734e+06, 1.2855164849321609336e+03, 1.0}
)

// besselY1Small is Y1 on (0, 8] (Boost.Math bessel_y1.hpp, logarithm as in
// besselY0Small).
func besselY1Small(x float64) float64 {
	const (
		x1  = 2.1971413260310170351e+00
		x2  = 5.4296810407941351328e+00
		x11 = 5.620e+02
		x12 = 1.8288260310170351490e-03
		x21 = 1.3900e+03
		x22 = -6.4592058648672279948e-06
	)
	if x < 0x1p-60 {
		// Y1(x) = -2/(pi x) + O(x log x)
		return -2 / math.Pi / x
	}
	y := x * x
	if x <= 4 {
		d := (x - x11/256) - x12
		z := 2 / math.Pi * besselLogRatio(x, x1, d) * besselJ1Small(x)
		return z + (x+x1)*d/x*besselRational(besselY1P1, besselY1Q1, y)
	}
	d := (x - x21/256) - x22
	z := 2 / math.Pi * besselLogRatio(x, x2, d) * besselJ1Small(x)
	return z + (x+x2)*d/x*besselRational(besselY1P2, besselY1Q2, y)
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
	case x <= 8:
		return besselJ0Small(x)
	case x < besselZeroTaylorMax:
		return besselZeroTaylor(besselJ0Zeros[:], 0, x)
	}
	j, _, _ := besselAsymJY(0, x)
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
	case x <= 8:
		return sign * besselJ1Small(x)
	case x < besselZeroTaylorMax:
		return sign * besselZeroTaylor(besselJ1Zeros[:], 1, x)
	}
	j, _, _ := besselAsymJY(1, x)
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
	case x <= 8:
		return besselY0Small(x)
	case x < besselZeroTaylorMax:
		return besselZeroTaylor(besselY0Zeros[:], 0, x)
	}
	_, y, _ := besselAsymJY(0, x)
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
	case x <= 8:
		return besselY1Small(x)
	case x < besselZeroTaylorMax:
		return besselZeroTaylor(besselY1Zeros[:], 1, x)
	}
	_, y, _ := besselAsymJY(1, x)
	return y
}
