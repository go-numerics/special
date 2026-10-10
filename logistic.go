package special

import (
	"math"

	"github.com/go-numerics/special/internal/dd"
)

// expDD returns exp(x) as a double-double (dd.ExpMant scaled).
func expDD(x float64) dd.DD {
	m, k := dd.ExpMant(dd.F(x))
	return dd.Ldexp(m, k)
}

// Expit returns the logistic sigmoid 1/(1 + exp(-x)).
//
// The exponential of the smaller magnitude side is formed as a
// double-double and the quotient taken in double-double, so the result is
// within about half an ulp everywhere, including the lower tail where it
// is exp(x) to full relative accuracy. Expit(-Inf) = 0, Expit(+Inf) = 1,
// NaN for NaN.
func Expit(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		return 1 // 1 - e^-40 rounds to 1
	case x < -746:
		return 0
	}
	if x >= 0 {
		t := expDD(-x)
		return dd.Div(dd.F(1), dd.AddF(t, 1)).Float()
	}
	t := expDD(x)
	return dd.Div(t, dd.AddF(t, 1)).Float()
}

// LogExpit returns log(Expit(x)) = -log(1 + exp(-x)) without underflow for
// large negative x (where it is x - log1p(exp(x)) ~ x) and with full
// relative accuracy for large positive x (where it is -exp(-x)).
func LogExpit(x float64) float64 {
	return -Log1pExp(-x)
}

// Log1pExp returns log(1 + exp(x)), the softplus function, accurate in both
// tails: x + log1p(exp(-x)) for x > 0 (no overflow) and log1p(exp(x)) with
// exp(x) in double-double for x <= 0 (full relative accuracy as it tends to
// exp(x)). Log1pExp(-Inf) = 0, Log1pExp(+Inf) = +Inf, NaN for NaN.
func Log1pExp(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		return x + math.Exp(-x) // exact sum to within the last bit
	case x < -746:
		return 0
	}
	if x > 0 {
		return dd.AddF(log1pDD(expDD(-x)), x).Float()
	}
	return log1pDD(expDD(x)).Float()
}

// Logit returns the log-odds log(p/(1-p)), the inverse of Expit, for
// 0 <= p <= 1: -Inf at 0, +Inf at 1, NaN outside or for NaN. 1-p and the
// quotient are carried in double-double and the logarithm keeps its
// relative accuracy as p/(1-p) approaches 1, so the result is accurate
// around p = 1/2 where it crosses zero.
func Logit(p float64) float64 {
	switch {
	case math.IsNaN(p) || p < 0 || p > 1:
		return math.NaN()
	case p == 0:
		return math.Inf(-1)
	case p == 1:
		return math.Inf(1)
	}
	q := sumDD(1, -p)
	return logDD(dd.Div(dd.F(p), q)).Float()
}

// Xlogy returns x*log(y), defined as 0 when x = 0 and y is not NaN (so that
// 0*log(0) terms of entropies vanish), as SciPy's xlogy. The product is
// formed with log(y) in double-double: one rounding.
func Xlogy(x, y float64) float64 {
	if x == 0 && !math.IsNaN(y) {
		return 0
	}
	if math.IsNaN(x) || math.IsNaN(y) || y < 0 {
		return math.NaN()
	}
	if y == 0 || math.IsInf(y, 1) || math.IsInf(x, 0) {
		return x * math.Log(y)
	}
	return dd.MulF(dd.Log(y), x).Float()
}

// Xlog1py returns x*log1p(y), defined as 0 when x = 0 and y is not NaN, as
// SciPy's xlog1py, with log1p(y) in double-double: one rounding.
func Xlog1py(x, y float64) float64 {
	if x == 0 && !math.IsNaN(y) {
		return 0
	}
	if math.IsNaN(x) || math.IsNaN(y) || y < -1 {
		return math.NaN()
	}
	if y == -1 || math.IsInf(y, 1) || math.IsInf(x, 0) {
		return x * math.Log1p(y)
	}
	return dd.MulF(log1pDD(dd.F(y)), x).Float()
}

// LogSumExp returns log(Σ exp(x_i)) without overflow or underflow: the
// maximum m is factored out and log1p of Σ_{i≠argmax} exp(x_i - m) is taken
// in double-double (each x_i - m exact, each exponential to about 2^-54),
// so the result is within about half an ulp even when one term dominates.
// It returns -Inf for an empty slice or when every x_i is -Inf, +Inf if any
// x_i is +Inf, NaN if any is NaN.
func LogSumExp(x []float64) float64 {
	m, im := math.Inf(-1), -1
	for i, v := range x {
		if math.IsNaN(v) {
			return v
		}
		if v > m {
			m, im = v, i
		}
	}
	if math.IsInf(m, 0) {
		return m // +Inf, or -Inf for empty / all -Inf
	}
	var s dd.DD
	for i, v := range x {
		if i == im {
			continue
		}
		d := sumDD(v, -m)
		if d.Hi < -746 {
			continue
		}
		e, k := dd.ExpMant(d)
		s = dd.Add(s, dd.Ldexp(e, k))
	}
	return dd.AddF(log1pDD(s), m).Float()
}

// Softmax writes exp(x_i) / Σ_j exp(x_j) into dst (allocated when nil or
// too short) and returns it; the maximum is subtracted first and the sum
// is accumulated in double-double. An empty x gives an empty result; if x
// holds NaN or +Inf the results are NaN (as SciPy's softmax).
func Softmax(dst, x []float64) []float64 {
	if cap(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	if len(x) == 0 {
		return dst
	}
	m := math.Inf(-1)
	bad := false
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 1) {
			bad = true
		}
		if v > m {
			m = v
		}
	}
	if bad || math.IsInf(m, -1) {
		for i := range dst {
			dst[i] = math.NaN()
		}
		return dst
	}
	es := make([]dd.DD, len(x))
	var s dd.DD
	for i, v := range x {
		d := sumDD(v, -m)
		e, k := dd.ExpMant(d)
		es[i] = dd.Ldexp(e, k)
		s = dd.Add(s, es[i])
	}
	for i := range dst {
		dst[i] = dd.Div(es[i], s).Float()
	}
	return dst
}

// LogSoftmax writes x_i - LogSumExp(x) into dst (allocated when nil or too
// short) and returns it.
func LogSoftmax(dst, x []float64) []float64 {
	if cap(dst) < len(x) {
		dst = make([]float64, len(x))
	}
	dst = dst[:len(x)]
	if len(x) == 0 {
		return dst
	}
	// x_i - lse with lse kept in double-double: lse = m + log1p(s).
	m, im := math.Inf(-1), -1
	for i, v := range x {
		if v > m || math.IsNaN(v) {
			m, im = v, i
		}
		if math.IsNaN(v) {
			break
		}
	}
	if math.IsNaN(m) || math.IsInf(m, 0) {
		for i := range dst {
			dst[i] = math.NaN()
		}
		return dst
	}
	var s dd.DD
	for i, v := range x {
		if i == im {
			continue
		}
		d := sumDD(v, -m)
		e, k := dd.ExpMant(d)
		s = dd.Add(s, dd.Ldexp(e, k))
	}
	lse := dd.AddF(log1pDD(s), m)
	for i, v := range x {
		if math.IsInf(v, -1) {
			dst[i] = v
			continue
		}
		dst[i] = dd.Sub(dd.F(v), lse).Float()
	}
	return dst
}
