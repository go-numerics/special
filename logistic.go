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
// e = exp(-|x|) comes from the float64 kernel as hi + lo (about 2^-68
// relative); 1 + e is formed exactly (Fast2Sum) and the quotient 1/(1+e)
// or e/(1+e) is corrected by its FMA residual, so the result is within
// about half an ulp everywhere, including the lower tail where it is
// exp(x) to full relative accuracy. Below -660, where the low part of e
// would be subnormal, the double-double path is used. Expit(-Inf) = 0,
// Expit(+Inf) = 1, NaN for NaN.
func Expit(x float64) float64 {
	if !(x >= -660 && x <= 40) {
		return expitTail(x)
	}
	// e = exp(-|x|): expHL written out (a call costs a tenth of Expit)
	xh := -math.Abs(x)
	kd := math.RoundToEven(xh * expInvLn2N)
	ki := int(kd)
	r := xh - kd*expC1
	dr := -kd * expC2
	u := r + dr
	ue := dr - (u - r)
	u2 := u * u
	pa := math.FMA(u, 1.0/6, 0.5)
	pb := math.FMA(u, 1.0/120, 1.0/24)
	pb = math.FMA(u2, 1.0/720, pb)
	pu := u2 * math.FMA(u2, pb, pa)
	t := &expT[ki&127]
	th, tl := t[0], t[1]
	tu := float64(th * u)
	h0 := th + tu
	l := ((tu - (h0 - th)) + math.FMA(th, u, -tu)) + math.FMA(th, ue+pu, math.FMA(tl, u, tl))
	h := h0 + l
	l -= h - h0
	s := pow2(ki >> 7) // >= -953 for x >= -660
	h *= s
	l *= s
	// d = 1 + e exactly (e <= 1).
	dh := 1 + h
	dl := (1 - dh) + h + l
	nh, nl := 1.0, 0.0
	if x < 0 {
		nh, nl = h, l
	}
	inv := 1 / dh
	y := nh * inv
	res := math.FMA(-y, dh, nh) + (nl - y*dl) // n - y·d
	return math.FMA(res, inv, y)
}

// expitTail is Expit outside [-660, 40] (and NaN).
func expitTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		return 1 // 1 - e^-40 rounds to 1
	case x < -746:
		return 0
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
// tails: x + log1p(exp(-x)) for x > 0 (no overflow) and log1p(exp(x)) for
// x <= 0 (full relative accuracy as it tends to exp(x)). e = exp(-|x|)
// comes from the float64 kernel as hi + lo, w = 1 + e is formed exactly
// and log w = log(w.hi) + w.lo/w.hi from the log kernel (2^-68 absolute);
// x is added with TwoSum before the single final rounding.
// Log1pExp(-Inf) = 0, Log1pExp(+Inf) = +Inf, NaN for NaN.
func Log1pExp(x float64) float64 {
	if !(x >= -660 && x <= 40) {
		return log1pExpTail(x)
	}
	h, l, k := expHL(-math.Abs(x), 0)
	s := pow2(k)
	h *= s
	l *= s
	if h < 0x1p-27 {
		// log1p(e) = e - e²/2 + e³/3 - ..., e⁴/4 below 2^-80 e. (Through
		// w = 1 + e the second-order term of log(w.hi + w.lo) would be
		// lost, and it is not negligible when e is this small.)
		t := h + (l + h*h*math.FMA(h, 1.0/3, -0.5))
		if x > 0 {
			return x + t
		}
		return t
	}
	wh := 1 + h
	wl := (1 - wh) + h + l
	lh, ll := logHL(wh)
	ll += wl / wh
	if x <= 0 {
		return lh + ll
	}
	sh := x + lh
	bb := sh - x
	se := (x - (sh - bb)) + (lh - bb)
	return sh + (se + ll)
}

// log1pExpTail is Log1pExp outside [-660, 40] (and NaN).
func log1pExpTail(x float64) float64 {
	switch {
	case math.IsNaN(x):
		return x
	case x > 40:
		return x + math.Exp(-x) // exact sum to within the last bit
	case x < -746:
		return 0
	}
	return log1pDD(expDD(x)).Float()
}

// Logit returns the log-odds log(p/(1-p)), the inverse of Expit, for
// 0 <= p <= 1: -Inf at 0, +Inf at 1, NaN outside or for NaN. 1-p is formed
// exactly (TwoSum), the quotient p/(1-p) is carried as hi + lo through its
// FMA residual, and log(hi) comes from the log kernel, whose error is
// relative to hi - 1 near 1: the result is accurate around p = 1/2 where it
// crosses zero.
func Logit(p float64) float64 {
	if !(p > 0 && p < 1) {
		switch {
		case p == 0:
			return math.Inf(-1)
		case p == 1:
			return math.Inf(1)
		}
		return math.NaN()
	}
	qh := 1 - p
	bb := qh - 1
	ql := (1 - (qh - bb)) + (-p - bb) // 1 - p = qh + ql exactly
	if p < 0.25 || p > 0.75 {
		// |logit| > log 3: first-order corrections suffice.
		y := p / qh
		r := (math.FMA(-y, qh, p) - y*ql) / qh // p/(qh+ql) - y
		lh, ll := logHL(y)
		return lh + (ll + r/y)
	}
	// log1p(t), t = (2p - 1)/(1 - p); 2p - 1 is exact here. Carrying t
	// itself keeps the relative accuracy as the result crosses zero.
	num := 2*p - 1
	th := num / qh
	tl := (math.FMA(-th, qh, num) - th*ql) / qh
	return log1pHL(th, tl)
}

// log1pHL returns log1p(th + tl) rounded to float64 for th > -1 with
// |tl| <= ulp(th): the series for |th| < 2^-20 (terms to th⁴/4, the next
// below 2^-80 relative); otherwise w = 1 + t formed exactly and
// log(w.hi) + w.lo/w.hi, the dropped second-order term below 2^-86 of the
// result.
func log1pHL(th, tl float64) float64 {
	if math.Abs(th) < 0x1p-20 {
		s := th * th * math.FMA(th, math.FMA(th, -0.25, 1.0/3), -0.5)
		return th + (math.FMA(tl, -th, tl) + s)
	}
	wh := 1 + th
	bb := wh - 1
	wl := (1 - (wh - bb)) + (th - bb) + tl
	lh, ll := logHL(wh)
	return lh + (ll + wl/wh)
}

// Xlogy returns x*log(y), defined as 0 when x = 0 and y is not NaN (so that
// 0*log(0) terms of entropies vanish), as SciPy's xlogy. log(y) comes from
// the log kernel as hi + lo (2^-68 absolute) and the product is rounded
// once by an FMA. The kernel is written out here (the same operations as
// logHL, without its subnormal branch): at a few ns per call, the spill of
// a call under Go's ABI is a tenth of the cost.
func Xlogy(x, y float64) float64 {
	iy := math.Float64bits(y)
	if iy-1<<52 >= 0x7ff0000000000000-1<<52 || x == 0 || x-x != 0 {
		// y not normal positive finite (or NaN), x zero, infinite or NaN
		return xlogyTail(x, y)
	}
	tmp := iy - 0x3fe8000000000000
	ke := int(int64(tmp) >> 52)
	m := math.Float64frombits(iy - uint64(int64(tmp)>>52)<<52)
	t := &logT[int(m*256+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1
	r2 := r * r
	c0 := math.FMA(r, 1.0/3, -0.5)
	c1 := math.FMA(r, 1.0/5, -0.25)
	c2 := math.FMA(r, 1.0/7, -1.0/6)
	c2 = math.FMA(r2, -1.0/8, c2)
	c1 = math.FMA(r2, c2, c1)
	q := r2 * math.FMA(r2, c1, c0)
	kf := float64(ke)
	lh := kf*ln2Hi42 + t[1]
	hi := lh + r
	lo := (r - (hi - lh)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q))
	return math.FMA(x, hi, x*lo)
}

// xlogyTail is Xlogy for x = 0, y <= 0, subnormal, infinite or NaN, and x
// infinite or NaN.
func xlogyTail(x, y float64) float64 {
	if x == 0 && !math.IsNaN(y) {
		return 0
	}
	if math.IsNaN(x) || math.IsNaN(y) || y < 0 {
		return math.NaN()
	}
	if y > 0 && y < 0x1p-1022 && !math.IsInf(x, 0) {
		lh, ll := logHL(y)
		return math.FMA(x, lh, x*ll)
	}
	return x * math.Log(y)
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
