package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// r = x0 - 1 where x0 = 1.46163... is the positive zero of ψ, as a
// double-double (mpmath, 400 bits).
var digammaRoot = dd.DD{Hi: 0.46163214496836236, Lo: -1.5522348162858677e-17}

// digamma1p returns ψ(1+t) for t in [0, 1] as a double-double whose high
// part carries the error: ψ(1+t) = (t - r) D(t), with t - r formed exactly
// in double-double so that the relative accuracy holds at the zero, and D a
// piecewise Chebyshev interpolant fitted to mpmath
// (zz_digamma1pPieces_coeffs.go). The form follows Boost.Math's
// digamma_imp_1_2 (ψ(x) = (x - root)(Y + R(x-1)), root split in parts).
func digamma1p(t float64) dd.DD {
	h, l := dd.TwoSum(t, -digammaRoot.Hi)
	d := dd.DD{Hi: h, Lo: l - digammaRoot.Lo}
	return dd.Mul(d, evalPiecesDD(digamma1pPieces[:], t))
}

// recipDD returns 1/x as a double-double.
func recipDD(x float64) dd.DD {
	q := 1 / x
	if math.IsInf(q, 0) {
		return dd.F(q)
	}
	return dd.DD{Hi: q, Lo: math.FMA(-q, x, 1) / x}
}

// digammaAsym returns ψ(x) for x >= 10 by the asymptotic series
// ψ(x) = log x - 1/(2x) - Σ B_2k / (2k x^2k) (DLMF 5.11.2), with the
// logarithm and 1/(2x) in double-double. Nine terms bring the truncation
// error at x = 10 below 1e-19 relative.
func digammaAsym(x float64) dd.DD {
	l := dd.Log(x)
	h := recipDD(x)
	h = dd.DD{Hi: h.Hi / 2, Lo: h.Lo / 2}
	r := dd.Sub(l, h)
	if x > 1e17 {
		return r
	}
	z := 1 / (x * x)
	// B_2k/(2k), k = 1..9
	const (
		b1 = 1.0 / 12
		b2 = -1.0 / 120
		b3 = 1.0 / 252
		b4 = -1.0 / 240
		b5 = 1.0 / 132
		b6 = -691.0 / 32760
		b7 = 1.0 / 12
		b8 = -3617.0 / 8160
		b9 = 43867.0 / 14364
	)
	s := z * (b1 + z*(b2+z*(b3+z*(b4+z*(b5+z*(b6+z*(b7+z*(b8+z*b9))))))))
	return dd.AddF(r, -s)
}

// Digamma returns the digamma function ψ(x) = d/dx log Γ(x).
//
// Special values: ψ(+Inf) = +Inf; ψ(+0) = -Inf, ψ(-0) = +Inf; NaN at the
// poles x = -1, -2, ..., at -Inf and for NaN.
//
// Algorithm: on [1, 2] ψ(x) = (x - x0) D(x-1) with x0 the positive zero in
// double-double and D fitted to mpmath (the form of Boost.Math's
// digamma_imp_1_2); (0, 10) is reduced to [1, 2] by ψ(x+1) = ψ(x) + 1/x
// with the reciprocals in double-double; x >= 10 uses the asymptotic series
// (DLMF 5.11.2). For -40 < x < 0, ψ(x) = ψ(t) - Σ 1/(x+k) with t = x+n in
// [0, 1) and exact x+k, summed in double-double; below, the reflection
// ψ(x) = ψ(-x) - 1/x - π cot(πx) (DLMF 5.5.4) with exact reduction of πx.
func Digamma(x float64) float64 {
	switch {
	case math.IsNaN(x) || math.IsInf(x, -1):
		return math.NaN()
	case math.IsInf(x, 1):
		return x
	case x == 0:
		return math.Copysign(math.Inf(1), -x)
	case math.Abs(x) < 0x1p-60:
		// ψ(x) = -1/x - γ + O(x)
		r := recipDD(x)
		if math.IsInf(r.Hi, 0) {
			return -r.Hi
		}
		return dd.AddF(dd.Neg(r), -euler).Float()
	}
	if x > 0 {
		return digammaPos(x).Float()
	}
	if x == math.Trunc(x) {
		return math.NaN()
	}
	if x > -0.5 {
		// ψ(x) = ψ(1+x) - 1/x with ψ(1+x) from t = x directly... t must
		// be in [0, 1]; for x in (-0.5, 0) use ψ(x) = ψ(2+x) - 1/x - 1/(1+x)
		// would round 1+x, so go through the reflection-free recurrence
		// below instead with n = 1: t = x + 1 is inexact here, hence the
		// series around 1: ψ(1+x) for x in (-0.5, 0).
		return dd.Sub(digamma1pNeg(x), recipDD(x)).Float()
	}
	if x > -40 {
		n := math.Ceil(-x)
		t := x + n // exact, in (0, 1)
		// ψ(t) = ψ(1+t) - 1/t
		p := digamma1p(t)
		s := dd.Sub(p, recipDD(t))
		mag := math.Abs(p.Hi) + 1/t
		for k := 0.0; k < n; k++ {
			r := recipDD(x + k)
			s = dd.Sub(s, r)
			mag += math.Abs(r.Hi)
		}
		if math.Abs(s.Hi) < 0x1p-12*mag {
			// Near a zero of ψ the terms cancel and the fitted core's
			// 2^-56 is not enough: redo ψ(t) to about 2^-100.
			s = digammaFrac(t)
			for k := 0.0; k < n; k++ {
				s = dd.Sub(s, recipDD(x+k))
			}
		}
		return s.Float()
	}
	// ψ(x) = ψ(-x) - 1/x - π cot(πx), with cot(πx) in double-double so
	// that the cancellation near the zeros of ψ costs as little as the
	// asymptotic series' own accuracy allows.
	c := dd.Mul(piDD, dd.Div(cosPiDD(x), sinPiDD(x)))
	r := dd.Sub(digammaAsym(-x), recipDD(x))
	return dd.Sub(r, c).Float()
}

// digammaFrac returns ψ(t) for 0 < t < 1 to about 2^-100 relative to the
// terms involved: ψ(t) = ψ(t+40) - Σ_{k<40} 1/(t+k), the reciprocals and
// the asymptotic series at t+40 (DLMF 5.11.2, its first three terms in
// double-double) all carried in double-double. It is slow and used only
// where Digamma's terms cancel.
func digammaFrac(t float64) dd.DD {
	var sum dd.DD
	for k := 0.0; k < 40; k++ {
		sum = dd.Add(sum, dd.Div(dd.F(1), sumDD(t, k)))
	}
	w := sumDD(t, 40)
	iw := dd.Div(dd.F(1), w)
	iw2 := dd.Mul(iw, iw)
	a := dd.Sub(dd.AddF(dd.LogAccurate(w.Hi), w.Lo/w.Hi), dd.MulF(iw, 0.5))
	a = dd.Sub(a, dd.DivF(iw2, 12))
	iw4 := dd.Mul(iw2, iw2)
	a = dd.Add(a, dd.DivF(iw4, 120))
	a = dd.Sub(a, dd.DivF(dd.Mul(iw4, iw2), 252))
	// remaining B_2k/(2k w^2k), k = 4..10, in float64 (each < 1e-15)
	z := iw2.Hi
	tail := z * z * z * z * (-1.0/240 + z*(1.0/132+z*(-691.0/32760+z*(1.0/12+z*(-3617.0/8160+z*(43867.0/14364+z*(-174611.0/6600)))))))
	a = dd.AddF(a, -tail)
	return dd.Sub(a, sum)
}

// digamma1pNeg returns ψ(1+x) for x in (-0.5, 0) as ψ(2+x) - 1/(1+x), with
// 2+x and 1+x carried as exact double-doubles.
func digamma1pNeg(x float64) dd.DD {
	// ψ(1+x) = ψ(1 + (1+x)) - 1/(1+x); 1+x = h + l exactly.
	h, l := dd.TwoSum(1, x)
	// ψ(1+s) for s = h + l in (0.5, 1): evaluate the core at h and correct
	// to first order with ψ'(1+h) ~ D + (h - r)D'; the correction l is
	// below 2^-53, so ψ'(1+h)*l is ample.
	core := digamma1p(h)
	core = dd.AddF(core, l*trigamma1p(h))
	one := dd.DD{Hi: h, Lo: l}
	return dd.Sub(core, dd.Div(dd.F(1), one))
}

// trigamma1p returns ψ'(1+t) for t in [0.5, 1] to a few ulps, enough for a
// first-order correction: ψ'(1+t) = Σ_{k>=1} 1/(t+k)^2, summed to 8 terms
// with the Euler–Maclaurin tail 1/(t+8.5) (error < 1e-5 relative).
func trigamma1p(t float64) float64 {
	s := 0.0
	for k := 1.0; k <= 8; k++ {
		s += 1 / ((t + k) * (t + k))
	}
	y := t + 9
	return s + 1/y + 1/(2*y*y) + 1/(6*y*y*y)
}

func digammaPos(x float64) dd.DD {
	switch {
	case x < 1:
		// ψ(x) = ψ(1+x) - 1/x with ψ(1+x) from the core at t = x.
		return dd.Sub(digamma1p(x), recipDD(x))
	case x <= 2:
		return digamma1p(x - 1) // exact
	case x < 10:
		n := math.Ceil(x - 2) // z = x - n in (1, 2]
		s := digamma1p(x - n - 1)
		for j := 1.0; j <= n; j++ {
			s = dd.Add(s, recipDD(x-j))
		}
		return s
	}
	return digammaAsym(x)
}
