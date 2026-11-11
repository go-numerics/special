package special

import (
	"github.com/go-numerics/special/internal/libm"
	"math"
	"sync"

	"github.com/go-numerics/special/dd"
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
	r := xh - float64(kd*expC1)
	dr := float64(-kd * expC2)
	u := r + dr
	ue := dr - (u - r)
	u2 := float64(u * u)
	pa := math.FMA(u, 1.0/6, 0.5)
	pb := math.FMA(u, 1.0/120, 1.0/24)
	pb = math.FMA(u2, 1.0/720, pb)
	pu := float64(u2 * math.FMA(u2, pb, pa))
	t := &expT[ki&127]
	th, tl := t[0], t[1]
	tu := float64(th * u)
	h0 := th + tu
	l := ((tu - (h0 - th)) + math.FMA(th, u, -tu)) + math.FMA(th, ue+pu, math.FMA(tl, u, tl))
	h := h0 + l
	l -= h - h0
	s := pow2(ki >> 7) // >= -953 for x >= -660
	h = float64(h * (s))
	l = float64(l * (s))
	// d = 1 + e exactly (e <= 1).
	dh := 1 + h
	dl := (1 - dh) + h + l
	nh, nl := 1.0, 0.0
	if x < 0 {
		nh, nl = h, l
	}
	inv := 1 / dh
	y := float64(nh * inv)
	res := math.FMA(-y, dh, nh) + (nl - float64(y*dl)) // n - y·d
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
	// e = exp(-|x|) and log(1 + e): expHL and logHL written out (two
	// calls would cost a sixth of the function).
	xh := -math.Abs(x)
	kd := math.RoundToEven(xh * expInvLn2N)
	ki := int(kd)
	er := xh - float64(kd*expC1)
	dr := float64(-kd * expC2)
	u := er + dr
	ue := dr - (u - er)
	u2 := float64(u * u)
	pa := math.FMA(u, 1.0/6, 0.5)
	pb := math.FMA(u, 1.0/120, 1.0/24)
	pb = math.FMA(u2, 1.0/720, pb)
	pu := float64(u2 * math.FMA(u2, pb, pa))
	et := &expT[ki&127]
	th, tl := et[0], et[1]
	tu := float64(th * u)
	h0 := th + tu
	l := ((tu - (h0 - th)) + math.FMA(th, u, -tu)) + math.FMA(th, ue+pu, math.FMA(tl, u, tl))
	h := h0 + l
	l -= h - h0
	sc := pow2(ki >> 7)
	h = float64(h * (sc))
	l = float64(l * (sc))
	if h < 0x1p-27 {
		// log1p(e) = e - e²/2 + e³/3 - ..., e⁴/4 below 2^-80 e. (Through
		// w = 1 + e the second-order term of log(w.hi + w.lo) would be
		// lost, and it is not negligible when e is this small.)
		t := h + (l + float64(h*h*math.FMA(h, 1.0/3, -0.5)))
		if x > 0 {
			return x + t
		}
		return t
	}
	wh := 1 + h
	wl := (1 - wh) + h + l
	// log wh, wh in (1, 2]
	iw := math.Float64bits(wh)
	tmp := iw - 0x3fe8000000000000
	kf := float64(int64(tmp) >> 52) // 0 or 1
	m := math.Float64frombits(iw - uint64(int64(tmp)>>52)<<52)
	t := &logT[int(float64(m*256)+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1
	r2 := float64(r * r)
	q0 := math.FMA(r, 1.0/3, -0.5)
	q1 := math.FMA(r, 1.0/5, -0.25)
	q2 := math.FMA(r, 1.0/7, -1.0/6)
	q2 = math.FMA(r2, -1.0/8, q2)
	q1 = math.FMA(r2, q2, q1)
	q := float64(r2 * math.FMA(r2, q1, q0))
	l0 := float64(kf*ln2Hi42) + t[1]
	lh := l0 + r
	// wl/wh = wl invc 2^-k (1 - r + r² ...): the r² dropped is below 2^-61
	// of the result (|wl| < 2^-53, and the result is about r when c = 1,
	// above 2^-9 otherwise).
	ll := (r - (lh - l0)) + (math.FMA(kf, ln2Lo42, t[2]) + (math.FMA(pe, -r, pe) + q) + float64(wl*invc*(1-float64(0.5*kf))*(1-r)))
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
		return x + libm.Exp(-x) // exact sum to within the last bit
	case x < -746:
		return 0
	}
	return log1pDD(expDD(x)).Float()
}

// Logit returns the log-odds log(p/(1-p)), the inverse of Expit, for
// 0 <= p <= 1: -Inf at 0, +Inf at 1, NaN outside or for NaN.
//
// Algorithm: for q = min(p, 1-p) (1-p is exact for p >= 1/2) in [2^-29, 1/2),
// Logit(q) = (q - 1/2) G(q) with G fitted directly on 8 pieces per octave of
// q (logitPieces; the piece is the top 15 bits of q's float64 pattern, as for
// Ndtri), G and q - 1/2 each carried as hi + lo and multiplied before the
// single rounding: no logarithm or division, and the result is accurate
// where it crosses zero. Logit(p) = -Logit(1-p) for p > 1/2. Below 2^-29
// (and at the special values), logitTail: 1-p is formed exactly (TwoSum),
// the quotient p/(1-p) is carried as hi + lo through its FMA residual and
// log of it comes from the log kernel.
func Logit(p float64) float64 {
	q := min(p, 1-p) // NaN stays NaN; p outside [0, 1] or 0, 1 fail the range test below
	i := math.Float64bits(q)>>49 - ndtriBase
	if i >= ndtriN {
		return logitTail(p)
	}
	e := &logitEst[i&255]
	s := q - e.c // exact: q and c are in the same binade, c has 4 significant bits
	k := &e.q
	s2 := float64(s * s)
	s4 := float64(s2 * s2)
	a0 := math.FMA(k[5], s, k[4])
	a1 := math.FMA(k[7], s, k[6])
	a2 := math.FMA(k[9], s, k[8])
	a3 := math.FMA(k[11], s, k[10])
	a4 := math.FMA(k[13], s, k[12])
	a5 := math.FMA(k[15], s, k[14])
	b0 := math.FMA(a1, s2, a0)
	b1 := math.FMA(a3, s2, a2)
	b2 := math.FMA(a5, s2, a4)
	r := math.FMA(b2, s4*s4, math.FMA(b1, s4, b0))
	r = math.FMA(math.FMA(math.FMA(r, s, k[3]), s, k[2]), s, k[1]) // Q(s)
	r1 := float64(s * k[0])                                        // float64(): not fused into gh
	r1l := math.FMA(s, k[0], -r1) + float64(s*e.c1lo)
	gh := e.c0 + r1 // Fast2Sum: |s c1| < |c0| on every piece
	gl := (r1 - (gh - e.c0)) + (r1l + math.FMA(s2, r, e.c0lo))
	mh := q - 0.5 // q - 1/2 = mh + ml, Fast2Sum
	ml := q - (mh + 0.5)
	v := float64(mh * gh)
	lo := math.FMA(mh, gh, -v) + math.FMA(mh, gl, float64(ml*gh))
	// p > 1/2: negate (the sign of 1/2 - p; +0 at p = 1/2 where the result is +0)
	return math.Float64frombits(math.Float64bits(v+lo) ^ (math.Float64bits(0.5-p) & (1 << 63)))
}

// logitEst holds logitPieces, the last octave padded with the last piece.
var logitEst = func() (t [256]estPiece) {
	p := estPieces(logitPieces[:])
	for i := range t {
		t[i] = p[min(i, len(p)-1)]
	}
	return t
}()

// logitTail is Logit for p < 2^-29, p > 1 - 2^-29 and the special values.
func logitTail(p float64) float64 {
	switch {
	case p == 0:
		return math.Inf(-1)
	case p == 1:
		return math.Inf(1)
	case !(p > 0 && p < 1):
		return math.NaN()
	}
	qh := 1 - p
	bb := qh - 1
	ql := (1 - (qh - bb)) + (-p - bb) // 1 - p = qh + ql exactly
	y := p / qh
	r := (math.FMA(-y, qh, p) - float64(y*ql)) / qh // p/(qh+ql) - y
	lh, ll := logHL(y)
	return lh + (ll + r/y)
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
	t := &logT[int(float64(m*256)+0.5)-192]
	invc := t[0]
	p := float64(m * invc) // float64(): not fused into r below
	pe := math.FMA(m, invc, -p)
	r := p - 1
	r2 := float64(r * r)
	c0 := math.FMA(r, 1.0/3, -0.5)
	c1 := math.FMA(r, 1.0/5, -0.25)
	c2 := math.FMA(r, 1.0/7, -1.0/6)
	c2 = math.FMA(r2, -1.0/8, c2)
	c1 = math.FMA(r2, c2, c1)
	q := float64(r2 * math.FMA(r2, c1, c0))
	kf := float64(ke)
	lh := float64(kf*ln2Hi42) + t[1]
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
	return float64(x * libm.Log(y))
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
		return float64(x * libm.Log1p(y))
	}
	if v, ok := xlog1pyFast(x, y); ok {
		return v
	}
	return dd.MulF(log1pDD(dd.F(y)), x).Float()
}

// xlog1pyErr bounds |h + l - D| / |h| of xlog1pyFast, D the double-double
// value of Xlog1py's reference: the log kernel's relative error for the
// logarithm of w = 1 + y (below 2^-62, logHL), the reference's own 2^-90, and
// the product's roundings (2^-100), all under 2^-61.5; 2^-60 is used.
const xlog1pyErr = 0x1p-60

// xlog1pyFast returns x log1p(y) for 2^-20 <= |y| and y > -1 finite, and
// whether it is certified equal to the double-double reference's rounding
// (the Ziv test of gamma_fast.go): w = 1 + y is formed exactly as wh + wl,
// log(wh) comes from the log kernel as hi + lo, log1p(y) = log(wh) + wl/wh
// to 2^-86, and the product x (hi + lo) is kept as an unevaluated sum.
func xlog1pyFast(x, y float64) (float64, bool) {
	if !(math.Abs(y) >= 0x1p-20) || math.IsInf(y, 0) || math.IsInf(x, 0) {
		return 0, false
	}
	wh := 1 + y
	bb := wh - 1
	wl := (1 - (wh - bb)) + (y - bb) // 1 + y = wh + wl exactly
	lh, ll := logHLn(wh)
	ll += wl / wh
	h := float64(x * lh)
	l := math.FMA(x, lh, -h) + float64(x*ll)
	s := h + l // Fast2Sum: |l| < 2^-14 |h| (ll is not normalised)
	l -= s - h
	if !zivRound(s, l, float64(xlog1pyErr*math.Abs(s))) {
		return 0, false
	}
	return s, true
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

// softmaxScratch holds the double-double exponentials of Softmax.
var softmaxScratch sync.Pool

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
