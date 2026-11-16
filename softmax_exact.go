package special

import (
	"math"
	"sync"

	"github.com/go-numerics/special/dd"
)

// The double-double evaluations of the log-sum-exp family, as in v0.19.0: each
// x_i - m exact, each exponential to about 2^-54 (dd.ExpMant), the sum, the
// quotient and log1p in double-double. They are the exact option next to the
// float64 kernels of softmax_fast.go: about 4.5 times slower, within half an
// ulp where their conditions hold, and bit-identical across architectures.
//
// Where they hold: a term exp(x_i - m) below 2^-1022 is subnormal or lost, with
// an absolute error up to 2^-1075, so when the sum of the terms of the others
// is itself below about n·2^-1000 the relative accuracy of LogSumExp and
// LogSoftmax is not stated (x = [0, 2^20 × -720] gives 26298 u); and where
// m + log Σ cancels (m < 0, |LogSumExp| below log Σ) the relative error of
// log1pDD (about 2^-90) is magnified by the cancellation (x = [-ln 2, -ln 2]
// gives 4.5e6 u in the relative sense). Both are open (see the issue linked from
// the pull request that added the fast path).

// softmaxDivErr bounds the relative error of softmaxDivFast against the exact
// quotient: the residual r = a - q b is formed from exact pieces (the Sterbenz
// difference a.Hi - p, the FMA remainder pe) and two roundings of size
// 2^-53 |r| < 2^-106 |a|, q b.Lo is rounded at 2^-106 |a| and r/b.Hi at
// 2^-106 |q|: under 2^-104 in all. The reference dd.Div is within 15u² =
// 2^-100.1 of the exact quotient; 2^-98 covers both.
const softmaxDivErr = 0x1p-98

// softmaxDiv returns dd.Div(a, b).Float() for a, b > 0 normal: a float64
// evaluation of a/b as hi + lo with the Ziv test of gamma_fast.go (zivRound),
// the double-double division for the points it cannot decide (about 2^-44).
func softmaxDiv(a, b dd.DD) float64 {
	if v, ok := softmaxDivCertified(a, b); ok {
		return v
	}
	return dd.Div(a, b).Float()
}

// softmaxDivCertified is the float64 evaluation of softmaxDiv and whether the
// Ziv test decided its rounding.
func softmaxDivCertified(a, b dd.DD) (float64, bool) {
	q := a.Hi / b.Hi
	p := float64(q * b.Hi)
	pe := math.FMA(q, b.Hi, -p)
	r := ((a.Hi - p) - pe) + (a.Lo - float64(q*b.Lo))
	q2 := r / b.Hi
	h := q + q2
	l := q2 - (h - q)
	return h, zivRound(h, l, float64(softmaxDivErr*h))
}

// softmaxScratch holds the double-double exponentials of SoftmaxExact.
var softmaxScratch sync.Pool

// SoftmaxExact is Softmax with every exponential, the sum and the quotient in
// double-double: the result is the correctly rounded exp(x_i) / Σ_j exp(x_j) up
// to the conditions in the comment above (no subnormal terms that matter). Use
// it where the last bit matters or where the bits must not depend on the
// architecture; Softmax is several times faster, within 8·2^-53 relative. It
// writes into dst (allocated when nil or too short) and returns it; dst may be
// x itself. An empty x gives an empty result; if x holds NaN or +Inf the
// results are NaN (as SciPy's softmax).
func SoftmaxExact(dst, x []float64) []float64 {
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
	// The exponentials are kept in double-double for the division; the
	// scratch is pooled (a fresh 16 bytes per element costs more than the
	// division).
	bp, _ := softmaxScratch.Get().(*[]dd.DD)
	if bp == nil {
		bp = new([]dd.DD)
	}
	if cap(*bp) < len(x) {
		*bp = make([]dd.DD, len(x))
	}
	es := (*bp)[:len(x)]
	var s dd.DD
	for i, v := range x {
		d := sumDD(v, -m)
		e, k := dd.ExpMant(d)
		es[i] = dd.Ldexp(e, k)
		s = dd.Add(s, es[i])
	}
	for i := range dst {
		dst[i] = softmaxDiv(es[i], s)
	}
	softmaxScratch.Put(bp)
	return dst
}

// LogSoftmaxExact is LogSoftmax with x_i - lse in double-double, lse = m +
// log1p(Σ_{x_j < m} exp(x_j - m)): half an ulp where the conditions in the
// comment above hold. It writes into dst (allocated when nil or too short) and
// returns it; dst may be x itself.
func LogSoftmaxExact(dst, x []float64) []float64 {
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

// LogSumExpExact is LogSumExp with the exponentials, the sum and log1p in
// double-double: within half an ulp where the conditions in the comment above
// hold (it has no guard: where m + log Σ cancels it carries log1pDD's 2^-90
// relative error magnified, as the fast path's fallback does). It returns -Inf
// for an empty slice or when every x_i is -Inf, +Inf if any x_i is +Inf, NaN
// if any is NaN.
func LogSumExpExact(x []float64) float64 {
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
