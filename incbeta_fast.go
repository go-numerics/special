package special

import (
	"math"

	"github.com/go-numerics/special/dd"
)

// Float64 prefix of the incomplete beta function.
//
// TOMS 708's power prefix x^a y^b / B(a, b) was formed as a double-double
// logarithm (three dd gamma evaluations for log B, two dd logarithms, one dd
// exponential): about a third of BetaInc's time for a, b in (0.5, 50). Here,
// for 2^-20 <= a, b <= 85 (so a + b <= 170), the exponent
//
//	E = a log x + b log y - log Γ(a) - log Γ(b) + log Γ(a + b)
//
// is summed as hi + lo from the float64 kernels (log: 2^-68 absolute; log Γ:
// 2^-61 absolute up to 171.6; exp: 2^-68 relative), the rounding of a + b
// entering to first order through ψ(a + b). The error of E, times the
// multipliers a, b <= 85 on the logarithms, is below 2^-58 absolute, so the
// prefix, hence the result, is within 2^-58 relative of the double-double
// prefix (0.03 ulp); the series and fractions are unchanged. Results are no
// longer the bits of the double-double prefix: the contract is the error
// against the oracle (docs/accuracy.md), measured by the fixtures and by
// incbeta_fast_test.go.

// ibetaNoFast turns the float64 prefix off (the tests' double-double
// reference); never set outside tests.
var ibetaNoFast bool

// ibetaFastOK reports whether the float64 prefix covers (a, b).
func ibetaFastOK(a, b float64) bool {
	return !ibetaNoFast && a >= 0x1p-20 && b >= 0x1p-20 && a <= 85 && b <= 85
}

// ibetaScaleFast is the largest a + b whose ibetaXY skips the double-double
// logarithms (they are formed on demand, by withLogs, if the float64 prefix
// declines).
const ibetaScaleFast = 170

// hlAcc accumulates an unevaluated sum hi + lo.
type hlAcc struct{ h, l float64 }

func (e *hlAcc) add(h, l float64) {
	s := e.h + h // TwoSum
	bb := s - e.h
	e.l += ((e.h - (s - bb)) + (h - bb)) + l
	e.h = s
}

// addProd adds a·(h + l).
func (e *hlAcc) addProd(a, h, l float64) {
	p := float64(a * h)
	e.add(p, math.FMA(a, h, -p)+float64(a*l))
}

// ibetaLogGammaHL returns log Γ(x) as hi + lo for 2^-20 <= x <= 170.
func ibetaLogGammaHL(x float64) (hi, lo float64) {
	switch {
	case x < 0.5:
		// log(Γ(1+x)/x)
		gh, gl := gamma1pFast(x)
		lh, ll := logHLn(gh)
		xh, xl := logHLn(x)
		s := lh - xh // TwoSum
		bb := s - lh
		e := (lh - (s - bb)) + (-xh - bb)
		return s, e + ((ll + gl/gh) - xl)
	case x < 10:
		return logGammaMidHL(x)
	}
	return logGammaLargeHL(x)
}

// ibetaPowMantFast returns m, k with x^a y^b / B(a, b) = m·2^k, m in
// [0.99, 2.02) as hi + lo, and whether (a, b, x) are in its region.
func ibetaPowMantFast(a, b float64, v ibetaXY) (m dd.DD, k int, ok bool) {
	if !ibetaFastOK(a, b) || v.xd.Hi < 0x1p-1000 || v.yd.Hi < 0x1p-1000 {
		return dd.DD{}, 0, false
	}
	lxh, lxl := logHLn(v.xd.Hi)
	lxl += v.xd.Lo / v.xd.Hi
	lyh, lyl := logHLn(v.yd.Hi)
	lyl += v.yd.Lo / v.yd.Hi
	var e hlAcc
	e.addProd(a, lxh, lxl)
	e.addProd(b, lyh, lyl)
	h, l := ibetaLogGammaHL(a)
	e.add(-h, -l)
	h, l = ibetaLogGammaHL(b)
	e.add(-h, -l)
	// + log Γ(s + sl), s + sl = a + b exactly: ψ(s)·sl to first order, with
	// ψ to 2^-12 below 10 (sl < 2^-50) and log s - 1/2s - 1/12s² above
	// (error 2^-20; sl < 2^-45).
	s := a + b
	sl := b - (s - a) // Fast2Sum remainder, a >= b
	if b > a {
		sl = a - (s - b)
	}
	if s < 10 {
		h, l = ibetaLogGammaHL(s)
		l += float64(ibetaPsiApprox(s) * sl)
	} else {
		var lh float64
		h, l, lh, _ = logGammaLargeHLL(s)
		psi := lh - 0.5/s - float64(1/(12*float64(s*s)))
		l += float64(psi * sl)
	}
	e.add(h, l)
	if !(math.Abs(e.h) <= 700) {
		return dd.DD{}, 0, false
	}
	xh := e.h + e.l
	xl := e.l - (xh - e.h)
	hi, lo, k := expHL(xh, xl)
	return dd.DD{Hi: hi, Lo: lo}, k, true
}
