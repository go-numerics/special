package special

import (
	"math"

	"github.com/go-numerics/special/dd"
	"github.com/go-simd/floats"
)

// The log-sum-exp family on float64 kernels.
//
// Softmax, LogSoftmax and LogSumExp evaluate Σ_i exp(x_i - m), m = max x, once
// for all of x, and every output is a function of that sum and of x_i - m. The
// earlier code took each exponential in double-double (dd.ExpMant, then
// dd.Ldexp, dd.Add: 70 % of the profile of Softmax in #33) to return the
// correctly rounded quotient; this one keeps the two steps that float64 cannot
// do and lets float64 do the rest.
//
//   - x_i - m is formed exactly as h + l (Knuth's TwoSum). exp(h) alone would
//     carry the relative error |h| 2^-53 of the subtraction, up to 2^-43 for
//     |h| ~ 745 (the error of NumPy's and SciPy's softmax; 199 u measured on
//     the "wide" input of the tests). exp(h + l) = exp(h) (1 + l + l²/2 ...) with
//     |l| <= 2^-53 |h|, so one FMA, e + e l, is exact to the l²/2 term, below 2^-88.
//   - exp(h) is floats.Exp (Arm's optimized-routines exp, 0.511 ulp).
//   - the sum is compensated (four interleaved TwoSum accumulators): it carries
//     no error of its own, n 2^-106 relative, whatever n.
//
// The bounds below are stated with u = 2^-53; docs/accuracy.md derives them and
// the oracle tests assert them against the double-double evaluation these
// functions replaced (kept in the tests).
const (
	// softmaxErr bounds |Softmax_i - exact| / exact for exact >= 2^-1022:
	// e_i = exp(h_i) (1 + l_i) carries the exponential's 0.511 ulp = 1.022u and
	// the FMA's u, 2.03u; the compensated sum S of these carries 2.03u and the
	// rounding of S to float64 u; r = 1/S another u; the product e_i r another
	// u: 2.03 + 3.03 + 1 + 1 = 7.06u (second order terms are below 2^-100).
	softmaxErr = 8 * 0x1p-53

	// logSoftmaxErr bounds |LogSoftmax_i - exact| / |exact|: the sum's
	// 2.03u acts through log1p with a gain of at most 1 (s/(1+s) <= log1p(s)),
	// and |y_i| = |h_i + l_i| + L >= L (h_i <= 0), so the error on L is below
	// 2.03u |y_i|; two roundings of the final sum add 2u: 4.03u.
	logSoftmaxErr = 5 * 0x1p-53

	// logSumExpErr bounds |LogSumExp - exact| / |exact| on the fast path: 2.03u L
	// from the sum plus the rounding u |lse|, with L = log Σ exp(x_i - m) <=
	// |lse| (the guard of LogSumExp): 3.03u.
	logSumExpErr = 4 * 0x1p-53

	// smSmall and smBig are the chunk sizes of the scratch arrays on the stack:
	// a short x does not pay for zeroing the large ones; a long x runs the
	// three passes of a chunk on 4 KB that stay in the L1 cache.
	smSmall = 64
	smBig   = 512
)

// smShift returns the exact difference v - m as h + l, with l = 0 where h is
// not finite or v is NaN (the exponential of such an h is 0 or NaN and takes
// no correction).
func smShift(v, m float64) (h, l float64) {
	h = v - m
	bb := h - v
	l = (v - (h - bb)) + (-m - bb)
	if l != l {
		l = 0
	}
	return h, l
}

// smAdd is one step of a compensated sum, Fast2Sum (Dekker): s + c holds the
// sum so far to 2^-106 relative. It needs s >= t, which holds here: every
// accumulator starts at 1 and every summand is at most 1 (exp(h) <= 1 for
// h <= 0, and the correction e l moves a value below 1 by a relative 2^-53 at
// most).
func smAdd(s, c, t float64) (float64, float64) {
	a := s + t
	return a, c + (t - (a - s))
}

// smChunk writes exp(x_i - m) into buf (the exponentials of the corrected
// differences) and returns their sum as a double-double and the number of x_i
// equal to m. An element equal to the maximum gets the low part mark: 0 keeps
// its exponential, exactly 1; -1 turns it to 1 + 1·(-1) = 0, so that the sum
// is Σ_{x_i < m} exp(x_i - m), which log1p needs in full relative accuracy
// when one term dominates. len(buf) = len(x) <= len(lo).
func smChunk(buf, x []float64, m, mark float64, lo []float64) (dd.DD, int) {
	n := len(x)
	buf = buf[:n]
	lo = lo[:n]
	k := 0
	for i, v := range x {
		h, l := smShift(v, m)
		if h == 0 {
			l = mark
			k++
		}
		buf[i] = h
		lo[i] = l
	}
	floats.Exp(buf, buf)
	s0, s1, s2, s3 := 1.0, 1.0, 1.0, 1.0
	var c0, c1, c2, c3 float64
	for g := range n / 4 {
		i := 4 * g
		e0 := math.FMA(buf[i], lo[i], buf[i])
		e1 := math.FMA(buf[i+1], lo[i+1], buf[i+1])
		e2 := math.FMA(buf[i+2], lo[i+2], buf[i+2])
		e3 := math.FMA(buf[i+3], lo[i+3], buf[i+3])
		buf[i], buf[i+1], buf[i+2], buf[i+3] = e0, e1, e2, e3
		s0, c0 = smAdd(s0, c0, e0)
		s1, c1 = smAdd(s1, c1, e1)
		s2, c2 = smAdd(s2, c2, e2)
		s3, c3 = smAdd(s3, c3, e3)
	}
	for j := range n % 4 {
		i := 4*(n/4) + j
		e := math.FMA(buf[i], lo[i], buf[i])
		buf[i] = e
		s0, c0 = smAdd(s0, c0, e)
	}
	var s dd.DD
	// s_j - 1 is exact (s_j in [1, 2^53)): the ones the accumulators started
	// with leave no rounding error, even when the sum of the others is tiny.
	for _, f := range [...]float64{s0 - 1, s1 - 1, s2 - 1, s3 - 1, c0, c1, c2, c3} {
		s = dd.AddF(s, f)
	}
	return s, k
}

// smSum returns the sum of smChunk over x and the number of maxima, for finite
// m = max x. The exponentials go to dst when it is not nil (len(dst) = len(x);
// x may be dst itself, element i of x is read before element i of dst is
// written) and to a scratch on the stack otherwise.
func smSum(dst, x []float64, m, mark float64) (dd.DD, int) {
	if len(x) <= smSmall {
		var lo, tmp [smSmall]float64
		return smRun(dst, x, m, mark, lo[:], tmp[:])
	}
	var lo, tmp [smBig]float64
	return smRun(dst, x, m, mark, lo[:], tmp[:])
}

func smRun(dst, x []float64, m, mark float64, lo, tmp []float64) (dd.DD, int) {
	var s dd.DD
	k := 0
	size := len(lo)
	for ci := range (len(x) + size - 1) / size {
		start := ci * size
		c := min(len(x)-start, size)
		buf := tmp[:c]
		if dst != nil {
			buf = dst[start : start+c]
		}
		cs, ck := smChunk(buf, x[start:start+c], m, mark, lo)
		s = dd.Add(s, cs)
		k += ck
	}
	return s, k
}

// softmaxFast is Softmax for finite m = max x.
func softmaxFast(dst, x []float64, m float64) {
	s, _ := smSum(dst, x, m, 0)
	// One reciprocal and a SIMD scale instead of n divisions (0.35 against
	// 0.12 ns per element): one more rounding, u, on every result.
	floats.Scale(1/s.Float(), dst)
}

// smLog returns log Σ_i exp(x_i - m) = log1p((k - 1) + Σ_{x_i < m} exp(x_i - m))
// in double-double, for finite m = max x.
func smLog(x []float64, m float64) dd.DD {
	s, k := smSum(nil, x, m, -1)
	return log1pDD(dd.AddF(s, float64(k-1)))
}

// logSoftmaxFast is LogSoftmax for finite m = max x.
func logSoftmaxFast(dst, x []float64, m float64) {
	l := smLog(x, m)
	for i, v := range x {
		h, lo := smShift(v, m)
		dst[i] = (h - l.Hi) + (lo - l.Lo)
	}
}

// logSumExpFast is LogSumExp for finite m = max x, and whether the fast
// path's bound holds (|lse| is not below the log of the sum).
func logSumExpFast(x []float64, m float64) (float64, bool) {
	l := smLog(x, m)
	lse := dd.AddF(l, m).Float()
	return lse, l.Hi <= math.Abs(lse)
}
